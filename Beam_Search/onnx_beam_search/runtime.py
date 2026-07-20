"""Tokenizer-free ONNX Runtime beam-search engine.

The runner consumes a merged prefill graph and a merged decode graph. It owns
beam state, saved token IDs, cumulative scores, and I/O-binding ping-pong; model
adapters own prompt construction, tokenization, graph composition, and any
model-specific scalar inputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto
from onnxruntime.capi import _pybind_state as C


_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)

_NUMPY_DTYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(uint16)": np.uint16,
    "tensor(int16)": np.int16,
    "tensor(uint32)": np.uint32,
    "tensor(int32)": np.int32,
    "tensor(uint64)": np.uint64,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


@dataclass(frozen=True)
class BeamGraphIO:
    """Names of non-state values in merged prefill and decode graphs.

    State tensors must be the leading ``state_count`` inputs and outputs. The
    output tail order is fixed to saved IDs, cumulative scores, next beam token
    IDs, best token ID, and sequence length.
    """

    prefill_input_ids: str = "input_ids"
    prefill_save_ids: str = "save_id_in"
    prefill_beam_size: str = "beam_size"
    decode_input_ids: str = "input_ids"
    decode_sequence_length: str = "kv_seq_len"
    decode_previous_scores: str = "previous_prob"
    decode_save_ids: tuple[str, ...] = ("save_id_in",)
    decode_beam_size: str = "beam_size"
    decode_top_k: str = "topK"

    def __post_init__(self):
        name_fields = (
            "prefill_input_ids",
            "prefill_save_ids",
            "prefill_beam_size",
            "decode_input_ids",
            "decode_sequence_length",
            "decode_previous_scores",
            "decode_beam_size",
            "decode_top_k",
        )
        for field_name in name_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"BeamGraphIO.{field_name} must be a non-empty string."
                )
        if isinstance(self.decode_save_ids, str) or not isinstance(
            self.decode_save_ids, Sequence
        ):
            raise ValueError(
                "BeamGraphIO.decode_save_ids must be a non-empty sequence of names."
            )
        decode_save_ids = tuple(self.decode_save_ids)
        if not decode_save_ids or any(
            not isinstance(name, str) or not name for name in decode_save_ids
        ):
            raise ValueError(
                "BeamGraphIO.decode_save_ids must be a non-empty sequence of "
                "non-empty strings."
            )
        if len(set(decode_save_ids)) != len(decode_save_ids):
            raise ValueError("BeamGraphIO.decode_save_ids names must be unique.")
        object.__setattr__(self, "decode_save_ids", decode_save_ids)


@dataclass(frozen=True)
class BeamModelSpec:
    """Files and graph contract needed by the generic runtime."""

    prefill_model: Path
    decode_model: Path
    state_count: int
    max_sequence_length: int
    io: BeamGraphIO = field(default_factory=BeamGraphIO)
    shared_initializers: Path | None = None
    activations_fp16: bool = False

    def __post_init__(self):
        if self.state_count <= 0:
            raise ValueError(f"state_count must be positive, got {self.state_count}.")
        if self.max_sequence_length <= 0:
            raise ValueError(
                f"max_sequence_length must be positive, got {self.max_sequence_length}."
            )


@dataclass(frozen=True)
class OrtProviderConfig:
    """Execution-provider settings independent from any LLM adapter."""

    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    provider_options: tuple[Mapping[str, object], ...] | None = None
    device_type: str | None = None
    device_id: int = 0
    threads: int = 0
    log: bool = False

    def __post_init__(self):
        if not self.providers:
            raise ValueError("At least one ONNX Runtime provider is required.")
        if self.device_type not in (None, "cpu", "cuda", "dml"):
            raise ValueError(
                "device_type must be one of None, 'cpu', 'cuda', or 'dml'."
            )
        if self.provider_options is not None and len(self.provider_options) != len(
            self.providers
        ):
            raise ValueError("provider_options must align one-to-one with providers.")


@dataclass(frozen=True)
class BeamSearchResult:
    """Generated token IDs and phase timings returned by the generic runtime."""

    token_ids: tuple[int, ...]
    prompt_tokens: int
    generated_tokens: int
    decode_steps: int
    prefill_seconds: float
    decode_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds

    @property
    def prefill_tokens_per_second(self) -> float:
        return (
            self.prompt_tokens / self.prefill_seconds
            if self.prefill_seconds > 0
            else 0.0
        )

    @property
    def decode_tokens_per_second(self) -> float:
        return (
            self.decode_steps / self.decode_seconds
            if self.decode_seconds > 0
            else 0.0
        )


@dataclass(frozen=True)
class _IOPlan:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    state_inputs: tuple[str, ...]
    state_outputs: tuple[str, ...]
    save_ids_output: str
    scores_output: str
    next_tokens_output: str
    best_token_output: str
    sequence_length_output: str


def _external_data_map(initializer):
    return {entry.key: entry.value for entry in initializer.external_data}


def attach_shared_initializers(session_options, shared_model_path):
    """Memory-map a generic shared-initializer ONNX bundle into SessionOptions."""
    shared_model_path = Path(shared_model_path)
    shared_model = onnx.load(str(shared_model_path), load_external_data=False)
    arrays = {}
    ort_values = []
    for initializer in shared_model.graph.initializer:
        if initializer.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        external = _external_data_map(initializer)
        location = external.get("location")
        if not location:
            raise RuntimeError(
                f"Shared initializer {initializer.name!r} has no external data."
            )
        data_path = shared_model_path.parent / location
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        array = np.memmap(
            data_path,
            dtype=onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type),
            mode="r",
            offset=int(external.get("offset", "0")),
            shape=tuple(int(dim) for dim in initializer.dims),
        )
        arrays[initializer.name] = array
        ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(initializer.name, ort_value)
    return arrays, ort_values


def _np_dtype(type_name: str):
    if type_name == "tensor(bfloat16)":
        raise ValueError(
            "ONNX Runtime BF16 inputs cannot be constructed through this "
            "NumPy-backed runtime. Use a model with NumPy-compatible recurrent "
            "state inputs."
        )
    try:
        return _NUMPY_DTYPES[type_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported ONNX Runtime tensor type: {type_name}"
        ) from exc


def _input_dtypes(session):
    return {value.name: _np_dtype(value.type) for value in session.get_inputs()}


def _state_sequence_axis(value_meta) -> int:
    symbolic_axes = [
        axis
        for axis, dim in enumerate(value_meta.shape)
        if axis != 0 and not isinstance(dim, int)
    ]
    if len(symbolic_axes) != 1:
        raise RuntimeError(
            f"Expected one dynamic sequence axis for {value_meta.name!r}, "
            f"got {symbolic_axes}. Supply initial_states explicitly."
        )
    return symbolic_axes[0]


def _zero_state(value_meta):
    shape = list(value_meta.shape)
    sequence_axis = _state_sequence_axis(value_meta)
    for axis, dim in enumerate(shape):
        if axis == 0:
            shape[axis] = 1
        elif axis == sequence_axis:
            shape[axis] = 0
        elif not isinstance(dim, int):
            shape[axis] = 1
    return np.zeros(tuple(shape), dtype=_np_dtype(value_meta.type))


def _plan_io(session, state_count: int) -> _IOPlan:
    inputs = tuple(value.name for value in session.get_inputs())
    outputs = tuple(value.name for value in session.get_outputs())
    if len(inputs) < state_count:
        raise RuntimeError(
            f"Graph has {len(inputs)} inputs, fewer than state_count={state_count}."
        )
    if len(outputs) != state_count + 5:
        raise RuntimeError(
            f"Graph has {len(outputs)} outputs; expected {state_count} state "
            "outputs followed by exactly five beam-control outputs."
        )
    tail = outputs[state_count:]
    return _IOPlan(
        inputs=inputs,
        outputs=outputs,
        state_inputs=inputs[:state_count],
        state_outputs=outputs[:state_count],
        save_ids_output=tail[0],
        scores_output=tail[1],
        next_tokens_output=tail[2],
        best_token_output=tail[3],
        sequence_length_output=tail[4],
    )


def _validate_value_compatibility(source, target, relationship: str) -> None:
    if source.type != target.type:
        raise RuntimeError(
            f"{relationship} dtype mismatch: output {source.name!r} is "
            f"{source.type}, but input {target.name!r} is {target.type}."
        )
    source_shape = tuple(source.shape)
    target_shape = tuple(target.shape)
    if len(source_shape) != len(target_shape):
        raise RuntimeError(
            f"{relationship} rank mismatch: output {source.name!r} has shape "
            f"{source_shape}, but input {target.name!r} has shape {target_shape}."
        )
    conflicts = [
        (axis, source_dim, target_dim)
        for axis, (source_dim, target_dim) in enumerate(
            zip(source_shape, target_shape)
        )
        if isinstance(source_dim, int)
        and isinstance(target_dim, int)
        and source_dim != target_dim
    ]
    if conflicts:
        raise RuntimeError(
            f"{relationship} static-shape mismatch between output "
            f"{source.name!r} {source_shape} and input {target.name!r} "
            f"{target_shape}; conflicting axes: {conflicts}."
        )


def _inferred_device_type(provider: str) -> str:
    if provider == "CUDAExecutionProvider":
        return "cuda"
    if provider == "DmlExecutionProvider":
        return "dml"
    return "cpu"


def _ort_device(device_type: str, device_id: int):
    factories = {
        "cpu": C.OrtDevice.cpu,
        "cuda": C.OrtDevice.cuda,
        "dml": C.OrtDevice.dml,
    }
    return C.OrtDevice(
        factories[device_type](), C.OrtDevice.default_memory(), device_id
    )


class BeamSearchRunner:
    """Run merged beam graphs using dynamic-output I/O-binding ping-pong."""

    def __init__(
        self,
        model: BeamModelSpec,
        provider: OrtProviderConfig | None = None,
    ):
        self.model = model
        self.provider = provider or OrtProviderConfig()
        available = onnxruntime.get_available_providers()
        missing = [name for name in self.provider.providers if name not in available]
        if missing:
            raise RuntimeError(
                f"Unavailable ONNX Runtime providers: {missing}; installed: {available}"
            )
        for path in (model.prefill_model, model.decode_model):
            if not Path(path).exists():
                raise FileNotFoundError(path)
        if model.shared_initializers is not None and not Path(
            model.shared_initializers
        ).exists():
            raise FileNotFoundError(model.shared_initializers)

        self.device_type = provider_device = (
            self.provider.device_type
            or _inferred_device_type(self.provider.providers[0])
        )
        self.state_input_device = "cpu" if provider_device == "dml" else provider_device
        self.ort_device = _ort_device(provider_device, self.provider.device_id)
        self.run_options = self._create_run_options()
        self.prefill_session = self._create_session(model.prefill_model)
        self.decode_session = self._create_session(model.decode_model)
        self.prefill_plan = _plan_io(self.prefill_session, model.state_count)
        self.decode_plan = _plan_io(self.decode_session, model.state_count)
        self.prefill_dtypes = _input_dtypes(self.prefill_session)
        self.decode_dtypes = _input_dtypes(self.decode_session)
        self._validate_contract()

    def _create_session_options(self):
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 0 if self.provider.log else 4
        options.log_verbosity_level = 4
        options.inter_op_num_threads = self.provider.threads
        options.intra_op_num_threads = self.provider.threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.add_session_config_entry("session.set_denormal_as_zero", "1")
        options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        options.add_session_config_entry("session.inter_op.allow_spinning", "1")
        if self.model.activations_fp16:
            options.add_session_config_entry(
                "optimization.disable_specified_optimizers",
                "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer",
            )
        return options

    def _create_run_options(self):
        options = onnxruntime.RunOptions()
        options.log_severity_level = 0 if self.provider.log else 4
        options.log_verbosity_level = 4
        options.add_run_config_entry("disable_synchronize_execution_providers", "0")
        return options

    def _create_session(self, model_path):
        options = self._create_session_options()
        shared_refs = None
        if self.model.shared_initializers is not None:
            shared_refs = attach_shared_initializers(
                options, self.model.shared_initializers
            )
        disabled_optimizers = (
            ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
            if self.model.activations_fp16
            else None
        )
        session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=list(self.provider.providers),
            provider_options=(
                list(self.provider.provider_options)
                if self.provider.provider_options is not None
                else None
            ),
            disabled_optimizers=disabled_optimizers,
        )
        if shared_refs is not None:
            session._beam_search_shared_initializers = shared_refs
        return session

    def _validate_contract(self):
        io = self.model.io
        required_prefill = {
            io.prefill_input_ids,
            io.prefill_save_ids,
            io.prefill_beam_size,
        }
        required_decode = {
            io.decode_input_ids,
            io.decode_sequence_length,
            io.decode_previous_scores,
            io.decode_beam_size,
            io.decode_top_k,
            *io.decode_save_ids,
        }
        missing_prefill = required_prefill - set(self.prefill_plan.inputs)
        missing_decode = required_decode - set(self.decode_plan.inputs)
        if missing_prefill:
            raise RuntimeError(
                f"Prefill graph is missing configured inputs: {sorted(missing_prefill)}"
            )
        if missing_decode:
            raise RuntimeError(
                f"Decode graph is missing configured inputs: {sorted(missing_decode)}"
            )

        prefill_outputs = {
            value.name: value for value in self.prefill_session.get_outputs()
        }
        decode_inputs = {
            value.name: value for value in self.decode_session.get_inputs()
        }
        decode_outputs = {
            value.name: value for value in self.decode_session.get_outputs()
        }
        for index in range(self.model.state_count):
            decode_input = decode_inputs[self.decode_plan.state_inputs[index]]
            _validate_value_compatibility(
                prefill_outputs[self.prefill_plan.state_outputs[index]],
                decode_input,
                f"Recurrent state {index} from prefill to decode",
            )
            _validate_value_compatibility(
                decode_outputs[self.decode_plan.state_outputs[index]],
                decode_input,
                f"Recurrent state {index} across decode steps",
            )

        control_links = (
            (
                self.prefill_plan.scores_output,
                self.decode_plan.scores_output,
                io.decode_previous_scores,
                "cumulative scores",
            ),
            (
                self.prefill_plan.next_tokens_output,
                self.decode_plan.next_tokens_output,
                io.decode_input_ids,
                "next token IDs",
            ),
            (
                self.prefill_plan.sequence_length_output,
                self.decode_plan.sequence_length_output,
                io.decode_sequence_length,
                "sequence length",
            ),
        )
        for prefill_output, decode_output, decode_input, label in control_links:
            target = decode_inputs[decode_input]
            _validate_value_compatibility(
                prefill_outputs[prefill_output],
                target,
                f"Prefill-to-decode {label}",
            )
            _validate_value_compatibility(
                decode_outputs[decode_output],
                target,
                f"Decode-step {label}",
            )
        for name in io.decode_save_ids:
            target = decode_inputs[name]
            _validate_value_compatibility(
                prefill_outputs[self.prefill_plan.save_ids_output],
                target,
                "Prefill-to-decode saved token IDs",
            )
            _validate_value_compatibility(
                decode_outputs[self.decode_plan.save_ids_output],
                target,
                "Decode-step saved token IDs",
            )

    def _ort_value(self, array, device=None):
        return onnxruntime.OrtValue.ortvalue_from_numpy(
            np.ascontiguousarray(array),
            device or self.device_type,
            self.provider.device_id,
        )

    def _typed_value(self, name, value, dtypes, device=None):
        array = np.asarray(value).astype(dtypes[name], copy=False)
        return self._ort_value(array, device)

    def _bind_device_outputs(self, binding, names):
        for name in names:
            binding._iobinding.bind_output(name, self.ort_device)

    @staticmethod
    def _validate_extra_inputs(required_names, managed_names, extras, phase):
        extras = extras or {}
        overlap = set(extras) & managed_names
        if overlap:
            raise ValueError(
                f"{phase} extras cannot override managed inputs: {sorted(overlap)}"
            )
        missing = set(required_names) - managed_names - set(extras)
        if missing:
            raise ValueError(
                f"{phase} graph requires adapter inputs that were not supplied: "
                f"{sorted(missing)}"
            )
        unknown = set(extras) - set(required_names)
        if unknown:
            raise ValueError(
                f"{phase} extras contain names absent from the graph: {sorted(unknown)}"
            )
        return extras

    def generate(
        self,
        input_ids,
        *,
        beam_size: int = 3,
        top_k: int = 3,
        stop_token_ids: Sequence[int] = (),
        max_new_tokens: int | None = None,
        prefill_inputs: Mapping[str, object] | None = None,
        decode_inputs: Mapping[str, object] | None = None,
        initial_states: Mapping[str, object] | None = None,
    ) -> BeamSearchResult:
        """Generate token IDs without tokenizing or decoding text."""
        if beam_size < 2:
            raise ValueError(f"beam_size must be at least 2, got {beam_size}.")
        if top_k < beam_size:
            raise ValueError(f"top_k ({top_k}) must be >= beam_size ({beam_size}).")
        input_ids = np.asarray(input_ids)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must have shape (1, sequence), got {input_ids.shape}."
            )
        prompt_tokens = int(input_ids.shape[1])
        generation_limit = max(0, self.model.max_sequence_length - prompt_tokens)
        if max_new_tokens is not None:
            generation_limit = min(generation_limit, max(0, int(max_new_tokens)))
        stop_set = {int(token) for token in stop_token_ids}
        io = self.model.io

        prefill_managed = {
            *self.prefill_plan.state_inputs,
            io.prefill_input_ids,
            io.prefill_save_ids,
            io.prefill_beam_size,
        }
        decode_managed = {
            *self.decode_plan.state_inputs,
            io.decode_input_ids,
            io.decode_sequence_length,
            io.decode_previous_scores,
            io.decode_beam_size,
            io.decode_top_k,
            *io.decode_save_ids,
        }
        prefill_inputs = self._validate_extra_inputs(
            self.prefill_plan.inputs,
            prefill_managed,
            prefill_inputs,
            "Prefill",
        )
        decode_inputs = self._validate_extra_inputs(
            self.decode_plan.inputs,
            decode_managed,
            decode_inputs,
            "Decode",
        )

        prefill_input_meta = {
            value.name: value for value in self.prefill_session.get_inputs()
        }
        initial_states = initial_states or {}
        unknown_states = set(initial_states) - set(self.prefill_plan.state_inputs)
        if unknown_states:
            raise ValueError(
                f"initial_states contains unknown names: {sorted(unknown_states)}"
            )
        prefill_values = []
        prefill_binding = self.prefill_session.io_binding()

        def bind_prefill(name, value, device=None):
            ort_value = self._typed_value(
                name, value, self.prefill_dtypes, device=device
            )
            prefill_values.append(ort_value)
            prefill_binding.bind_ortvalue_input(name, ort_value)

        bind_prefill(io.prefill_input_ids, input_ids)
        bind_prefill(io.prefill_save_ids, np.zeros((beam_size, 0)))
        bind_prefill(io.prefill_beam_size, np.array([beam_size]))
        for name in self.prefill_plan.state_inputs:
            state = initial_states.get(name)
            if state is None:
                state = _zero_state(prefill_input_meta[name])
            bind_prefill(name, state, self.state_input_device)
        for name, value in prefill_inputs.items():
            bind_prefill(name, value)
        self._bind_device_outputs(prefill_binding, self.prefill_plan.outputs)

        prefill_start = time.time()
        self.prefill_session.run_with_iobinding(
            prefill_binding, run_options=self.run_options
        )
        prefill_seconds = time.time() - prefill_start
        prefill_outputs = prefill_binding.get_outputs()
        prefill_positions = {
            name: index for index, name in enumerate(self.prefill_plan.outputs)
        }

        cached_states = prefill_outputs[:self.model.state_count]
        saved_ids = prefill_outputs[
            prefill_positions[self.prefill_plan.save_ids_output]
        ]
        final_saved_ids = saved_ids
        beam_scores = prefill_outputs[
            prefill_positions[self.prefill_plan.scores_output]
        ]
        next_tokens = prefill_outputs[
            prefill_positions[self.prefill_plan.next_tokens_output]
        ]
        selected_token = int(
            prefill_outputs[
                prefill_positions[self.prefill_plan.best_token_output]
            ].numpy().flat[0]
        )
        sequence_length = prefill_outputs[
            prefill_positions[self.prefill_plan.sequence_length_output]
        ]
        generated_count = int(
            generation_limit > 0 and selected_token not in stop_set
        )

        decode_positions = {
            name: index for index, name in enumerate(self.decode_plan.outputs)
        }
        dynamic_output_names = list(self.decode_plan.state_outputs) + [
            self.decode_plan.save_ids_output
        ]
        static_values = [
            (
                io.decode_beam_size,
                self._typed_value(
                    io.decode_beam_size,
                    np.array([beam_size]),
                    self.decode_dtypes,
                ),
            ),
            (
                io.decode_top_k,
                self._typed_value(
                    io.decode_top_k,
                    np.array([top_k]),
                    self.decode_dtypes,
                ),
            ),
        ]
        for name, value in decode_inputs.items():
            static_values.append(
                (name, self._typed_value(name, value, self.decode_dtypes))
            )

        decode_bindings = [
            self.decode_session.io_binding(),
            self.decode_session.io_binding(),
        ]
        for binding in decode_bindings:
            for name, value in static_values:
                binding.bind_ortvalue_input(name, value)
            self._bind_device_outputs(binding, self.decode_plan.outputs)

        control_rebinds_left = [2, 2]
        decode_step = 0
        decode_start = time.time()
        while generated_count < generation_limit and selected_token not in stop_set:
            binding_index = decode_step & 1
            binding = decode_bindings[binding_index]
            if control_rebinds_left[binding_index]:
                control_rebinds_left[binding_index] -= 1
                binding.bind_ortvalue_input(
                    io.decode_sequence_length, sequence_length
                )
                binding.bind_ortvalue_input(io.decode_input_ids, next_tokens)
                binding.bind_ortvalue_input(
                    io.decode_previous_scores, beam_scores
                )
            for name, value in zip(self.decode_plan.state_inputs, cached_states):
                binding.bind_ortvalue_input(name, value)
            for name in io.decode_save_ids:
                binding.bind_ortvalue_input(name, saved_ids)

            # Preserve prior buffers as inputs before rebinding growing outputs.
            self._bind_device_outputs(binding, dynamic_output_names)
            self.decode_session.run_with_iobinding(
                binding, run_options=self.run_options
            )
            outputs = binding.get_outputs()

            cached_states = outputs[:self.model.state_count]
            saved_ids = outputs[
                decode_positions[self.decode_plan.save_ids_output]
            ]
            final_saved_ids = saved_ids
            selected_token = int(
                outputs[
                    decode_positions[self.decode_plan.best_token_output]
                ].numpy().flat[0]
            )
            if any(control_rebinds_left):
                sequence_length = outputs[
                    decode_positions[self.decode_plan.sequence_length_output]
                ]
                next_tokens = outputs[
                    decode_positions[self.decode_plan.next_tokens_output]
                ]
                beam_scores = outputs[
                    decode_positions[self.decode_plan.scores_output]
                ]
            if selected_token not in stop_set:
                generated_count += 1
            decode_step += 1
        decode_seconds = time.time() - decode_start

        generated = []
        if generation_limit > 0:
            for token in final_saved_ids.numpy()[0][:generated_count]:
                token = int(token)
                if token in stop_set:
                    break
                generated.append(token)
        return BeamSearchResult(
            token_ids=tuple(generated),
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated),
            decode_steps=decode_step,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
        )