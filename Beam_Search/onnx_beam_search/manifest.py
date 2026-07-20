"""JSON manifest support for model-agnostic beam-search runtimes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .runtime import BeamGraphIO, BeamModelSpec


@dataclass(frozen=True)
class BeamSearchManifest:
    """Runtime model plus adapter-supplied inputs loaded from JSON."""

    model: BeamModelSpec
    prefill_inputs: Mapping[str, object]
    decode_inputs: Mapping[str, object]
    stop_token_ids: tuple[int, ...]

    def resolved_prefill_inputs(self, prompt_length: int) -> dict[str, object]:
        def resolve(value):
            if value == "$prompt_length":
                return prompt_length
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in value.items()}
            return value

        return {
            name: resolve(value) for name, value in self.prefill_inputs.items()
        }


def _required(data, key):
    if key not in data:
        raise ValueError(f"Beam manifest is missing required key {key!r}.")
    return data[key]


def _required_integer(data, key, field_name):
    value = _required(data, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Beam manifest {field_name} must be an integer.")
    return value


def _resolve_path(base: Path, value, field_name, *, required=True):
    if value is None:
        if required:
            raise ValueError(f"Beam manifest {field_name} cannot be null.")
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Beam manifest {field_name} must be a non-empty path string."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_beam_search_manifest(path: str | Path) -> BeamSearchManifest:
    """Load a generic runtime manifest with paths relative to the JSON file."""
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)
    if not isinstance(data, dict):
        raise ValueError("Beam manifest root must be a JSON object.")

    model_data = _required(data, "model")
    if not isinstance(model_data, dict):
        raise ValueError("Beam manifest 'model' must be a JSON object.")
    io_data = model_data.get("io", {})
    if not isinstance(io_data, dict):
        raise ValueError("Beam manifest model.io must be a JSON object.")
    valid_io_fields = set(BeamGraphIO.__dataclass_fields__)
    unknown_io = set(io_data) - valid_io_fields
    if unknown_io:
        raise ValueError(f"Unknown BeamGraphIO fields: {sorted(unknown_io)}")
    if "decode_save_ids" in io_data:
        io_data = dict(io_data)
        decode_save_ids = io_data["decode_save_ids"]
        if not isinstance(decode_save_ids, list):
            raise ValueError(
                "Beam manifest model.io.decode_save_ids must be a JSON array."
            )
        io_data["decode_save_ids"] = tuple(decode_save_ids)

    activations_fp16 = model_data.get("activations_fp16", False)
    if not isinstance(activations_fp16, bool):
        raise ValueError(
            "Beam manifest model.activations_fp16 must be a boolean."
        )

    model = BeamModelSpec(
        prefill_model=_resolve_path(
            path.parent,
            _required(model_data, "prefill_model"),
            "model.prefill_model",
        ),
        decode_model=_resolve_path(
            path.parent,
            _required(model_data, "decode_model"),
            "model.decode_model",
        ),
        state_count=_required_integer(
            model_data, "state_count", "model.state_count"
        ),
        max_sequence_length=_required_integer(
            model_data,
            "max_sequence_length",
            "model.max_sequence_length",
        ),
        io=BeamGraphIO(**io_data),
        shared_initializers=_resolve_path(
            path.parent,
            model_data.get("shared_initializers"),
            "model.shared_initializers",
            required=False,
        ),
        activations_fp16=activations_fp16,
    )
    prefill_inputs = data.get("prefill_inputs", {})
    decode_inputs = data.get("decode_inputs", {})
    if not isinstance(prefill_inputs, dict) or not isinstance(decode_inputs, dict):
        raise ValueError("prefill_inputs and decode_inputs must be JSON objects.")
    stop_token_ids = data.get("stop_token_ids", [])
    if not isinstance(stop_token_ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int)
        for token in stop_token_ids
    ):
        raise ValueError(
            "Beam manifest stop_token_ids must be a JSON array of integers."
        )
    return BeamSearchManifest(
        model=model,
        prefill_inputs=prefill_inputs,
        decode_inputs=decode_inputs,
        stop_token_ids=tuple(stop_token_ids),
    )