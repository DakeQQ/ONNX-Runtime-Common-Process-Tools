"""Model-agnostic beam-search operators and ONNX exporter.

The exporter knows only about logits and opaque state tensors. LLM cache
discovery, graph composition, metadata, tokenization, and prompt formatting
belong in adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch


DEFAULT_GRAPH_FILE_NAMES = {
    "first_beam": "BeamSearch_First.onnx",
    "next_beam": "BeamSearch_Next.onnx",
    "gather_first_beam": "BeamSearch_GatherFirst.onnx",
    "concat_first_beam": "BeamSearch_ConcatFirst.onnx",
}


@dataclass(frozen=True)
class BeamStateSpec:
    """Describe one opaque state tensor carried through beam selection."""

    input_name: str
    output_name: str
    shape: tuple[int | None, ...]
    dtype: torch.dtype
    sequence_axis: int

    def __post_init__(self):
        if len(self.shape) < 2:
            raise ValueError(f"State {self.input_name!r} must have rank >= 2.")
        if not 0 < self.sequence_axis < len(self.shape):
            raise ValueError(
                f"State {self.input_name!r} sequence_axis must be in "
                f"[1, {len(self.shape) - 1}], got {self.sequence_axis}."
            )
        invalid = [dim for dim in self.shape if dim is not None and dim <= 0]
        if invalid:
            raise ValueError(
                f"State {self.input_name!r} has non-positive dimensions: {invalid}."
            )

    def example(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        shape = [1 if dimension is None else dimension for dimension in self.shape]
        shape[0] = batch_size
        shape[self.sequence_axis] = sequence_length
        return torch.zeros(tuple(shape), dtype=self.dtype)


class FirstBeamSearch(torch.nn.Module):
    """Expand one state row and select the first ``beam_size`` tokens."""

    def __init__(
        self, num_states: int, token_dtype: torch.dtype = torch.int32
    ):
        super().__init__()
        self.num_states = num_states
        self.token_dtype = token_dtype
        self.save_states = [None] * num_states

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(
            logits, dim=-1, k=beam_size, sorted=True, largest=True
        )
        top_beam_prob = top_beam_logits - row_logsumexp

        for index in range(self.num_states):
            state = all_inputs[index]
            self.save_states[index] = state.repeat(
                beam_size, *((1,) * (state.dim() - 1))
            )

        top_beam_indices = top_beam_indices.transpose(0, 1).to(self.token_dtype)
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        return (
            *self.save_states,
            save_id,
            top_beam_prob.transpose(0, 1),
            top_beam_indices,
            top_beam_indices[[0]],
        )


class NextBeamSearch(torch.nn.Module):
    """Select global candidates and gather every state from its parent beam."""

    def __init__(
        self, num_states: int, token_dtype: torch.dtype = torch.int32
    ):
        super().__init__()
        self.num_states = num_states
        self.token_dtype = token_dtype
        self.save_states = [None] * num_states

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size = all_inputs[-2]
        top_k = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(
            logits, k=top_k, dim=-1, largest=True, sorted=True
        )
        current_prob = (top_k_logits - row_logsumexp + previous_prob).reshape(-1)
        top_beam_prob, flat_beam_indices = torch.topk(
            current_prob, k=beam_size, dim=-1, largest=True, sorted=True
        )
        beam_index = flat_beam_indices // top_k
        top_beam_indices = top_k_indices.reshape(-1)[flat_beam_indices]

        for index in range(self.num_states):
            self.save_states[index] = torch.index_select(
                all_inputs[index], dim=0, index=beam_index
            )

        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).to(self.token_dtype)
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)
        return (
            *self.save_states,
            save_id,
            top_beam_prob.unsqueeze(-1),
            top_beam_indices,
            top_beam_indices[[0]],
        )


class GatherFirstBeam(torch.nn.Module):
    """Select state row zero when leaving a multi-beam path."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "first_beam_row", torch.tensor([0], dtype=torch.int64), persistent=False
        )

    def forward(self, *states):
        return tuple(
            torch.index_select(state, dim=0, index=self.first_beam_row)
            for state in states
        )


class ConcatFirstBeam(torch.nn.Module):
    """Join a one-row prefix to beam row zero, then broadcast over all beams."""

    def __init__(self, concat_axes: Sequence[int]):
        super().__init__()
        self.concat_axes = tuple(concat_axes)
        self.register_buffer(
            "first_beam_row", torch.tensor([0], dtype=torch.int64), persistent=False
        )

    def forward(self, *all_inputs):
        num_states = len(self.concat_axes)
        prefixes = all_inputs[:num_states]
        suffixes = all_inputs[num_states:]
        outputs = []
        for prefix, suffix, axis in zip(prefixes, suffixes, self.concat_axes):
            suffix_top1 = torch.index_select(
                suffix, dim=0, index=self.first_beam_row
            )
            full_top1 = torch.cat([prefix, suffix_top1], dim=axis)
            outputs.append(
                full_top1.expand(suffix.shape[0], *([-1] * (full_top1.dim() - 1)))
            )
        return tuple(outputs)


def _validate_specs(state_specs: Sequence[BeamStateSpec]) -> None:
    if not state_specs:
        raise ValueError("state_specs must contain at least one state tensor.")
    input_names = [spec.input_name for spec in state_specs]
    output_names = [spec.output_name for spec in state_specs]
    if len(set(input_names)) != len(input_names):
        raise ValueError("State input names must be unique.")
    if len(set(output_names)) != len(output_names):
        raise ValueError("State output names must be unique.")


def _export(
    module,
    args,
    path,
    input_names,
    output_names,
    dynamic_axes,
    opset,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.with_name(path.name + ".data").unlink(missing_ok=True)
    torch.onnx.export(
        module.eval(),
        tuple(args),
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )
    return path


def export_beam_search_graphs(
    state_specs: Sequence[BeamStateSpec],
    vocab_size: int,
    output_folder: str | Path,
    *,
    beam_size: int = 3,
    top_k: int = 3,
    opset: int = 20,
    token_dtype: torch.dtype = torch.int32,
    logits_dtype: torch.dtype = torch.float32,
    file_names: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Export four generic beam-state graphs from explicit tensor contracts."""
    _validate_specs(state_specs)
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be at least 2, got {vocab_size}.")
    if beam_size < 2:
        raise ValueError(f"beam_size must be at least 2, got {beam_size}.")
    if top_k < beam_size:
        raise ValueError(f"top_k ({top_k}) must be >= beam_size ({beam_size}).")
    if top_k > vocab_size:
        raise ValueError(
            f"top_k ({top_k}) cannot exceed vocab_size ({vocab_size})."
        )
    if opset < 11:
        raise ValueError(f"opset must be at least 11, got {opset}.")
    if token_dtype not in (torch.int32, torch.int64):
        raise ValueError("token_dtype must be torch.int32 or torch.int64.")
    if logits_dtype not in (torch.float16, torch.float32):
        raise ValueError("logits_dtype must be torch.float16 or torch.float32.")

    names = dict(DEFAULT_GRAPH_FILE_NAMES)
    if file_names:
        unknown = set(file_names) - set(names)
        if unknown:
            raise ValueError(f"Unknown beam graph file-name roles: {sorted(unknown)}")
        names.update(file_names)

    output_folder = Path(output_folder)
    input_names = [spec.input_name for spec in state_specs]
    output_names = [spec.output_name for spec in state_specs]
    first_states = [spec.example(1, 4) for spec in state_specs]
    beam_states = [spec.example(beam_size, 4) for spec in state_specs]
    first_logits = torch.zeros((1, vocab_size), dtype=logits_dtype)
    beam_logits = torch.zeros((beam_size, vocab_size), dtype=logits_dtype)
    save_ids = torch.zeros((beam_size, 4), dtype=token_dtype)
    beam_size_tensor = torch.tensor([beam_size], dtype=torch.int64)
    top_k_tensor = torch.tensor([top_k], dtype=torch.int32)

    first_axes = {}
    next_axes = {}
    gather_axes = {}
    concat_axes = {}
    prefix_names = []
    suffix_names = []
    for spec in state_specs:
        first_axes[spec.input_name] = {
            0: "prefill_batch",
            spec.sequence_axis: "history_len",
        }
        first_axes[spec.output_name] = {
            0: "beam_size",
            spec.sequence_axis: "kv_seq_len",
        }
        next_axes[spec.input_name] = {
            0: "beam_size",
            spec.sequence_axis: "history_len",
        }
        next_axes[spec.output_name] = {
            0: "beam_size",
            spec.sequence_axis: "kv_seq_len",
        }
        gather_axes[spec.input_name] = {
            0: "beam_size",
            spec.sequence_axis: "history_len",
        }
        gather_axes[spec.output_name] = {spec.sequence_axis: "history_len"}
        suffix = (
            spec.output_name[4:]
            if spec.output_name.startswith("out_")
            else spec.output_name
        )
        prefix_name = f"in_prefix_{suffix}"
        suffix_name = f"in_beam_{suffix}"
        prefix_names.append(prefix_name)
        suffix_names.append(suffix_name)
        concat_axes[prefix_name] = {spec.sequence_axis: "prefix_len"}
        concat_axes[suffix_name] = {
            0: "beam_size",
            spec.sequence_axis: "suffix_len",
        }
        concat_axes[spec.output_name] = {
            0: "beam_size",
            spec.sequence_axis: "concat_len",
        }

    first_axes.update({
        "logits": {0: "prefill_batch"},
        "save_id_in": {0: "beam_size", 1: "history_len"},
        "save_id_out": {0: "beam_size", 1: "history_len"},
        "top_beam_prob": {0: "beam_size"},
        "top_beam_indices": {0: "beam_size"},
        "max_logits_idx": {0: "best_beam_batch"},
    })
    next_axes.update({
        "logits": {0: "beam_size"},
        "save_id_in": {0: "beam_size", 1: "history_len"},
        "previous_prob": {0: "beam_size"},
        "save_id_out": {0: "beam_size", 1: "history_len"},
        "top_beam_prob": {0: "beam_size"},
        "top_beam_indices": {0: "beam_size"},
        "max_logits_idx": {0: "best_beam_batch"},
    })
    output_tail = [
        "save_id_out",
        "top_beam_prob",
        "top_beam_indices",
        "max_logits_idx",
    ]

    return {
        "first_beam": _export(
            FirstBeamSearch(len(state_specs), token_dtype),
            first_states + [first_logits, save_ids, beam_size_tensor],
            output_folder / names["first_beam"],
            input_names + ["logits", "save_id_in", "beam_size"],
            output_names + output_tail,
            first_axes,
            opset,
        ),
        "next_beam": _export(
            NextBeamSearch(len(state_specs), token_dtype),
            beam_states + [
                beam_logits,
                save_ids,
                torch.zeros((beam_size, 1), dtype=logits_dtype),
                beam_size_tensor,
                top_k_tensor,
            ],
            output_folder / names["next_beam"],
            input_names + [
                "logits",
                "save_id_in",
                "previous_prob",
                "beam_size",
                "topK",
            ],
            output_names + output_tail,
            next_axes,
            opset,
        ),
        "gather_first_beam": _export(
            GatherFirstBeam(),
            beam_states,
            output_folder / names["gather_first_beam"],
            input_names,
            output_names,
            gather_axes,
            opset,
        ),
        "concat_first_beam": _export(
            ConcatFirstBeam([spec.sequence_axis for spec in state_specs]),
            [spec.example(1, 2) for spec in state_specs]
            + [spec.example(beam_size, 2) for spec in state_specs],
            output_folder / names["concat_first_beam"],
            prefix_names + suffix_names,
            output_names,
            concat_axes,
            opset,
        ),
    }