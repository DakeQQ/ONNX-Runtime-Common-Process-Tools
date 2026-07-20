"""Discover generic beam-state contracts from an ONNX model interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import onnx
import torch
from onnx import TensorProto

from .exporter import BeamStateSpec


_TORCH_DTYPES = {
    TensorProto.FLOAT: torch.float32,
    TensorProto.FLOAT16: torch.float16,
    TensorProto.BFLOAT16: torch.bfloat16,
    TensorProto.INT8: torch.int8,
    TensorProto.UINT8: torch.uint8,
    TensorProto.INT16: torch.int16,
    TensorProto.UINT16: torch.uint16,
    TensorProto.INT32: torch.int32,
    TensorProto.INT64: torch.int64,
}


@dataclass(frozen=True)
class MainGraphContract:
    """State and logits interface discovered from one autoregressive Main graph."""

    state_specs: tuple[BeamStateSpec, ...]
    logits_output_name: str
    vocab_size: int
    logits_dtype: torch.dtype
    opset: int


def _torch_dtype(value_info) -> torch.dtype:
    elem_type = value_info.type.tensor_type.elem_type
    try:
        return _TORCH_DTYPES[elem_type]
    except KeyError as exc:
        raise TypeError(
            f"Unsupported ONNX tensor type "
            f"{TensorProto.DataType.Name(elem_type)} for {value_info.name!r}."
        ) from exc


def _shape(value_info) -> tuple[int | None, ...]:
    dimensions = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dimensions.append(int(dim.dim_value))
        else:
            dimensions.append(None)
    return tuple(dimensions)


def _infer_sequence_axis(value_info) -> int:
    symbolic_axes = [
        axis
        for axis, dim in enumerate(value_info.type.tensor_type.shape.dim)
        if axis != 0 and dim.dim_param
    ]
    if len(symbolic_axes) != 1:
        raise ValueError(
            f"Cannot infer one sequence axis for {value_info.name!r}; "
            f"symbolic non-batch axes are {symbolic_axes}. Pass sequence_axes."
        )
    return symbolic_axes[0]


def _infer_state_input_name(output_name: str, model_input_names: set[str]) -> str:
    candidates = []
    if output_name.startswith("out_"):
        candidates.append("in_" + output_name[4:])
    candidates.append(output_name)
    for candidate in candidates:
        if candidate in model_input_names:
            return candidate
    raise ValueError(
        f"Cannot infer the state input paired with output {output_name!r}. "
        "Pass state_input_names explicitly."
    )


def _validate_state_pair(input_info, output_info, sequence_axis: int) -> None:
    input_shape = _shape(input_info)
    output_shape = _shape(output_info)
    input_dtype = input_info.type.tensor_type.elem_type
    output_dtype = output_info.type.tensor_type.elem_type
    if input_dtype != output_dtype:
        raise ValueError(
            f"State pair {input_info.name!r} -> {output_info.name!r} has "
            f"different dtypes: {TensorProto.DataType.Name(input_dtype)} and "
            f"{TensorProto.DataType.Name(output_dtype)}."
        )
    if len(input_shape) != len(output_shape):
        raise ValueError(
            f"State pair {input_info.name!r} -> {output_info.name!r} has "
            f"different ranks: {input_shape} and {output_shape}."
        )
    if not 0 < sequence_axis < len(output_shape):
        raise ValueError(
            f"State pair {input_info.name!r} -> {output_info.name!r} has "
            f"sequence axis {sequence_axis} outside rank {len(output_shape)}."
        )

    input_symbolic_axes = [
        axis
        for axis, dim in enumerate(input_info.type.tensor_type.shape.dim)
        if axis != 0 and dim.dim_param
    ]
    if input_symbolic_axes and sequence_axis not in input_symbolic_axes:
        raise ValueError(
            f"State input {input_info.name!r} has symbolic non-batch axes "
            f"{input_symbolic_axes}, which do not include output sequence axis "
            f"{sequence_axis}."
        )

    conflicts = [
        (axis, input_dim, output_dim)
        for axis, (input_dim, output_dim) in enumerate(
            zip(input_shape, output_shape)
        )
        if axis != sequence_axis
        and input_dim is not None
        and output_dim is not None
        and input_dim != output_dim
    ]
    if conflicts:
        raise ValueError(
            f"State pair {input_info.name!r} -> {output_info.name!r} has "
            f"incompatible static dimensions: {conflicts}."
        )


def discover_main_graph_contract(
    model_path: str | Path,
    *,
    state_count: int | None = None,
    logits_output_name: str | None = None,
    state_input_names: Sequence[str] | None = None,
    state_output_names: Sequence[str] | None = None,
    sequence_axes: Mapping[str, int] | Sequence[int] | None = None,
) -> MainGraphContract:
    """Inspect a Main graph without loading its external weight data.

    By default the final graph output is treated as logits and all preceding
    outputs are treated as opaque recurrent states. Explicit names override
    positional discovery for models with a different interface.
    """
    model = onnx.load(str(Path(model_path)), load_external_data=False)
    outputs_by_name = {value.name: value for value in model.graph.output}
    inputs_by_name = {value.name: value for value in model.graph.input}
    model_input_names = set(inputs_by_name)

    if logits_output_name is None:
        if not model.graph.output:
            raise ValueError("Main graph has no outputs.")
        logits_info = model.graph.output[-1]
        logits_output_name = logits_info.name
    else:
        try:
            logits_info = outputs_by_name[logits_output_name]
        except KeyError as exc:
            raise ValueError(
                f"Logits output {logits_output_name!r} is not a graph output."
            ) from exc

    if state_output_names is None:
        candidate_outputs = [
            value for value in model.graph.output if value.name != logits_output_name
        ]
        if state_count is None:
            state_count = len(candidate_outputs)
        if state_count <= 0:
            raise ValueError(f"state_count must be positive, got {state_count}.")
        state_infos = candidate_outputs[:state_count]
        if len(state_infos) != state_count:
            raise ValueError(
                f"Main graph exposes {len(candidate_outputs)} non-logits outputs, "
                f"fewer than state_count={state_count}."
            )
    else:
        missing = [name for name in state_output_names if name not in outputs_by_name]
        if missing:
            raise ValueError(f"State outputs are missing from Main: {missing}")
        state_infos = [outputs_by_name[name] for name in state_output_names]
        if state_count is not None and state_count != len(state_infos):
            raise ValueError(
                f"state_count={state_count} does not match "
                f"{len(state_infos)} explicit state outputs."
            )

    if state_input_names is None:
        resolved_input_names = [
            _infer_state_input_name(info.name, model_input_names)
            for info in state_infos
        ]
    else:
        if len(state_input_names) != len(state_infos):
            raise ValueError(
                "state_input_names and state outputs must have the same length."
            )
        missing = [name for name in state_input_names if name not in model_input_names]
        if missing:
            raise ValueError(f"State inputs are missing from Main: {missing}")
        resolved_input_names = list(state_input_names)

    if sequence_axes is None:
        resolved_sequence_axes = [
            _infer_sequence_axis(info) for info in state_infos
        ]
    elif isinstance(sequence_axes, Mapping):
        missing = [info.name for info in state_infos if info.name not in sequence_axes]
        if missing:
            raise ValueError(f"sequence_axes is missing state outputs: {missing}")
        resolved_sequence_axes = [sequence_axes[info.name] for info in state_infos]
    else:
        if len(sequence_axes) != len(state_infos):
            raise ValueError("sequence_axes and state outputs must have the same length.")
        resolved_sequence_axes = list(sequence_axes)

    state_specs = []
    for input_name, info, sequence_axis in zip(
        resolved_input_names, state_infos, resolved_sequence_axes
    ):
        _validate_state_pair(inputs_by_name[input_name], info, sequence_axis)
        state_specs.append(
            BeamStateSpec(
                input_name=input_name,
                output_name=info.name,
                shape=_shape(info),
                dtype=_torch_dtype(info),
                sequence_axis=sequence_axis,
            )
        )

    logits_shape = _shape(logits_info)
    if len(logits_shape) != 2 or logits_shape[-1] is None:
        raise ValueError(
            f"Logits output {logits_output_name!r} must have rank 2 with a "
            f"static vocabulary dimension, got shape {logits_shape}."
        )
    opset_versions = [
        item.version for item in model.opset_import if item.domain in ("", "ai.onnx")
    ]
    if not opset_versions:
        raise ValueError("Main graph has no default ONNX opset import.")

    return MainGraphContract(
        state_specs=tuple(state_specs),
        logits_output_name=logits_output_name,
        vocab_size=logits_shape[-1],
        logits_dtype=_torch_dtype(logits_info),
        opset=max(opset_versions),
    )