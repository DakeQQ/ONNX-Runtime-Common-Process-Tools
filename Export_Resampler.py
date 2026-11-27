"""Unified ONNX audio resampler exporter, optimizer, and verifier.

Consolidates all three resampler export targets into a single tool with shared
ONNX hygiene (`OPSET=20`, `IR_VERSION=9`, unset default domain, zero `DOUBLE`,
minimal `INT64`, `onnxslim` + deterministic LCA initializer re-hoisting, atomic saves):

- `--target capture` (was `Export_Capture_Resampler.py`):
  Streaming microphone capture frontend (`--mode unified` -> `capture_48000_to_16000.onnx`,
  drop-in replacement for `ai_voice_models/common/capture_48000_to_16000.onnx`; plus
  all 8 branch-free single-operating-point modes `48k_1ch_F32`..`16k_8ch_I16`).
- `--target multirate` (was `Export_MultiRate_Resampler.py`):
  Dynamic multi-rate `(0, 96000] Hz -> 16 kHz` polyphase resampler
  (`multirate_resampler_to_16000.onnx`, drop-in compatible with
  `ai_voice_models/common/tts_reference.onnx`; plus `--legacy-stage0` and `--optimize-input`).
- `--target simple` (was standalone `Export_Resampler.py`):
  Lightweight fixed-ratio mean-pool / linear-interpolate resampler
  (`resampler_48000_to_16000.onnx`, static or `--dynamic`, `F32`/`INT16`).
- `--target all`:
  Export and verify all three default targets sequentially.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import copy
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime
from onnxslim import slim
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Shared Configuration & ONNX Helpers
# ==========================================
_SCRIPT_DIR = Path(__file__).resolve().parent

EXPORT_TARGET = "all"                # Options: "capture", "multirate", "simple", "all".
OPSET = 20
IR_VERSION = 9
OUTPUT_SAMPLE_RATE = 16000
VERIFY_EXPORTED_MODEL = True

# Capture target defaults
CAPTURE_MODE = "unified"             # "unified" or any key in CAPTURE_OPERATING_POINTS.
CAPTURE_OUTPUT_DTYPE = "F32"         # Single-operating-point output/cache dtype ("F32"|"F16").
FIR_INPUT_SAMPLE_RATE = 48000
FIR_TAPS = 191
KAISER_BETA = 8.0
CUTOFF_HZ = 7300.0
DUMMY_OUTPUT_LENGTH = 3200

# Multi-rate target defaults
MULTIRATE_POLYPHASE = True
MAX_INPUT_SAMPLE_RATE = 96000
MULTIRATE_MODEL_NAME = f"multirate_resampler_to_{OUTPUT_SAMPLE_RATE}.onnx"
MAX_HALF = int(math.floor(8.0 * MAX_INPUT_SAMPLE_RATE / OUTPUT_SAMPLE_RATE + 0.5))
B44_NUM_GROUPS = 5

# Simple resampler target defaults
SIMPLE_DYNAMIC = False
SIMPLE_USE_MEAN_POOLING = True
SIMPLE_INPUT_LENGTH = 48000
SIMPLE_RESAMPLE_RATIO = 1.0 / 3.0
SIMPLE_INPUT_DTYPE = "INT16"
SIMPLE_OUTPUT_DTYPE = "INT16"

_DTYPE_TO_TORCH = {"F32": torch.float32, "F16": torch.float16, "INT16": torch.int16}
_DTYPE_TO_NP = {"F32": np.float32, "F16": np.float16, "INT16": np.int16}
_DTYPE_TO_ONNX = {"F32": TensorProto.FLOAT, "F16": TensorProto.FLOAT16, "INT16": TensorProto.INT16}

_GraphPath = tuple[tuple[int, str], ...]
_InitKey = tuple[int, tuple[int, ...], bytes]


def _atomic_save_onnx(model: onnx.ModelProto, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = model.SerializeToString()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, destination)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _make_session_options() -> onnxruntime.SessionOptions:
    so = onnxruntime.SessionOptions()
    so.log_severity_level = 4
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("optimization.enable_cast_chain_elimination", "1")
    return so


def _normalize_default_opset(model: onnx.ModelProto) -> None:
    """Ensure IR_VERSION and default opset domain unset (HasField('domain') == False)."""
    model.ir_version = IR_VERSION
    for op in model.opset_import:
        if op.domain == "":
            op.ClearField("domain")


def _assert_structural_hygiene(
    model: onnx.ModelProto,
    label: str,
    *,
    required_props: tuple[tuple[str, str], ...] = (),
) -> None:
    """Verify IR version, unset default opset domain, zero DOUBLE, zero duplicate initializers, and metadata."""
    onnx.checker.check_model(model)
    if model.ir_version != IR_VERSION:
        raise RuntimeError(f"{label}: expected ir_version={IR_VERSION}, got {model.ir_version}")
    if len(model.opset_import) != 1 or model.opset_import[0].version != OPSET or model.opset_import[0].HasField("domain"):
        raise RuntimeError(f"{label}: unexpected opset_import={list(model.opset_import)}")

    if required_props:
        actual_props = {p.key: p.value for p in model.metadata_props}
        for k, v in required_props:
            if actual_props.get(k) != v:
                raise RuntimeError(f"{label}: metadata mismatch for {k!r}: {actual_props}")
        keys = list(actual_props.keys())
        positions = [keys.index(k) for k, _ in required_props]
        if positions != sorted(positions):
            raise RuntimeError(f"{label}: metadata key order mismatch: {keys}")

    seen: dict[_InitKey, str] = {}

    def _walk(graph: onnx.GraphProto, prefix: str = "") -> None:
        for vi in (*graph.input, *graph.output, *graph.value_info):
            if vi.type.tensor_type.elem_type == TensorProto.DOUBLE:
                raise RuntimeError(f"{label}: forbidden DOUBLE tensor {prefix}{vi.name}")
        for init in graph.initializer:
            if init.data_type == TensorProto.DOUBLE:
                raise RuntimeError(f"{label}: forbidden DOUBLE initializer {prefix}{init.name}")
            key: _InitKey = (init.data_type, tuple(init.dims), numpy_helper.to_array(init).tobytes())
            full_name = f"{prefix}{init.name}"
            if key in seen:
                raise RuntimeError(f"{label}: duplicate initializer {full_name} matches {seen[key]}")
            seen[key] = full_name
        for node in graph.node:
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == TensorProto.DOUBLE:
                        raise RuntimeError(f"{label}: forbidden DOUBLE Cast in {prefix}{node.name or node.output[0]}")
            for attr in node.attribute:
                if attr.HasField("g"):
                    _walk(attr.g, f"{prefix}{attr.g.name}/")

    _walk(model.graph)


def _strip_value_info(model: onnx.ModelProto) -> None:
    """Clear redundant intermediate value_info entries across all subgraphs."""
    def _clear(graph: onnx.GraphProto) -> None:
        del graph.value_info[:]
        for node in graph.node:
            for attr in node.attribute:
                if attr.HasField("g"):
                    _clear(attr.g)

    _clear(model.graph)


def _downcast_slice_indices_to_int32(model: onnx.ModelProto) -> None:
    """Downcast constant INT64 Slice index/step initializers to INT32 across all scopes and drop unit steps."""
    i32_min = np.iinfo(np.int32).min
    i32_max = np.iinfo(np.int32).max
    graphs: list[tuple[_GraphPath, onnx.GraphProto]] = []

    def _collect(graph: onnx.GraphProto, path: _GraphPath) -> None:
        graphs.append((path, graph))
        for idx, node in enumerate(graph.node):
            for attr in node.attribute:
                if attr.HasField("g"):
                    _collect(attr.g, (*path, (idx, attr.name)))

    _collect(model.graph, ())
    inits_by_path: dict[_GraphPath, dict[str, onnx.TensorProto]] = {
        path: {init.name: init for init in graph.initializer} for path, graph in graphs
    }

    def _resolve_init(path: _GraphPath, name: str) -> tuple[_GraphPath, onnx.TensorProto] | None:
        cur = path
        while True:
            if name in inits_by_path[cur]:
                return cur, inits_by_path[cur][name]
            if not cur:
                return None
            cur = cur[:-1]

    slice_candidates: set[tuple[_GraphPath, str]] = set()
    blocked: set[tuple[_GraphPath, str]] = set()

    for path, graph in graphs:
        locally_produced = {vi.name for vi in graph.input}
        for node in graph.node:
            if node.op_type == "Slice":
                if node.input and node.input[0] not in locally_produced:
                    res = _resolve_init(path, node.input[0])
                    if res is not None:
                        blocked.add((res[0], node.input[0]))
                if len(node.input) == 5 and node.input[4] and node.input[4] not in locally_produced:
                    step_res = _resolve_init(path, node.input[4])
                    if step_res is not None:
                        step_arr = numpy_helper.to_array(step_res[1])
                        if step_arr.size == 1 and int(step_arr.reshape(-1)[0]) == 1:
                            del node.input[4]
                idx_refs: list[tuple[_GraphPath, str] | None] = []
                all_const_i64_or_i32 = True
                for idx_inp in node.input[1:]:
                    if not idx_inp:
                        continue
                    if idx_inp in locally_produced:
                        all_const_i64_or_i32 = False
                        continue
                    res = _resolve_init(path, idx_inp)
                    if res is None or res[1].data_type not in (TensorProto.INT64, TensorProto.INT32):
                        all_const_i64_or_i32 = False
                        continue
                    idx_refs.append((res[0], idx_inp))
                for ref in idx_refs:
                    if ref is not None:
                        if all_const_i64_or_i32:
                            slice_candidates.add(ref)
                        else:
                            blocked.add(ref)
            else:
                for inp in node.input:
                    if not inp or inp in locally_produced:
                        continue
                    res = _resolve_init(path, inp)
                    if res is not None:
                        blocked.add((res[0], inp))
            locally_produced.update(o for o in node.output if o)
        for out in graph.output:
            if out.name not in locally_produced:
                res = _resolve_init(path, out.name)
                if res is not None:
                    blocked.add((res[0], out.name))

    for gpath, name in sorted(slice_candidates - blocked):
        init = inits_by_path[gpath][name]
        if init.data_type != TensorProto.INT64:
            continue
        arr = numpy_helper.to_array(init).astype(np.int64)
        init.CopyFrom(numpy_helper.from_array(np.clip(arr, i32_min, i32_max).astype(np.int32), name=name))


def _deduplicate_and_hoist_initializers(model: onnx.ModelProto) -> None:
    """Re-hoist shared subgraph initializers cloned by onnxslim back to their LCA graph."""
    graphs: list[tuple[_GraphPath, onnx.GraphProto]] = []

    def _collect(graph: onnx.GraphProto, path: _GraphPath) -> None:
        graphs.append((path, graph))
        for node_idx, node in enumerate(graph.node):
            for attr in node.attribute:
                if attr.HasField("g"):
                    _collect(attr.g, (*path, (node_idx, attr.name)))

    _collect(model.graph, ())

    local_inits: dict[_GraphPath, dict[str, tuple[_InitKey, onnx.TensorProto]]] = {}
    key_defs: dict[_InitKey, list[tuple[_GraphPath, str, onnx.TensorProto]]] = {}
    all_produced: set[str] = set()
    for path, graph in graphs:
        all_produced.update(vi.name for vi in graph.input)
        for node in graph.node:
            all_produced.update(o for o in node.output if o)
        lmap: dict[str, tuple[_InitKey, onnx.TensorProto]] = {}
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            ckey: _InitKey = (init.data_type, tuple(arr.shape), arr.tobytes())
            lmap[init.name] = (ckey, init)
            key_defs.setdefault(ckey, []).append((path, init.name, init))
        local_inits[path] = lmap

    def _resolve_init_key(path: _GraphPath, name: str) -> _InitKey | None:
        cur = path
        while True:
            lmap = local_inits[cur]
            if name in lmap:
                return lmap[name][0]
            if not cur:
                return None
            cur = cur[:-1]

    key_uses: dict[_InitKey, list[_GraphPath]] = {k: [] for k in key_defs}
    for path, graph in graphs:
        locally_produced = {vi.name for vi in graph.input}
        for node in graph.node:
            for inp in node.input:
                if not inp or inp in locally_produced:
                    continue
                ckey = _resolve_init_key(path, inp)
                if ckey is not None:
                    key_uses[ckey].append(path)
            locally_produced.update(o for o in node.output if o)
        for out in graph.output:
            if out.name not in locally_produced:
                ckey = _resolve_init_key(path, out.name)
                if ckey is not None:
                    key_uses[ckey].append(path)

    def _lca(paths: list[_GraphPath]) -> _GraphPath:
        prefix = paths[0]
        for other in paths[1:]:
            k = 0
            lim = min(len(prefix), len(other))
            while k < lim and prefix[k] == other[k]:
                k += 1
            prefix = prefix[:k]
        return prefix

    canonical_for_key: dict[_InitKey, str] = {}
    used_canon: set[str] = set(all_produced)
    new_inits: dict[_GraphPath, list[onnx.TensorProto]] = {path: [] for path, _ in graphs}
    for ckey, defs in key_defs.items():
        uses = key_uses[ckey]
        if not uses:
            continue
        names = [name for _, name, _ in defs]
        safe_names = [n for n in names if n not in used_canon] or names
        preferred = (
            [n for n in safe_names if n == "weight"]
            or [n for n in safe_names if n.startswith("top_")]
            or [n for n in safe_names if n.startswith("capture_")]
        )
        canon_name = sorted(preferred, key=lambda s: (len(s), s))[0] if preferred else sorted(safe_names, key=lambda s: (len(s), s))[0]
        if canon_name in used_canon:
            suffix = 0
            while f"{canon_name}_h{suffix}" in used_canon:
                suffix += 1
            canon_name = f"{canon_name}_h{suffix}"
        used_canon.add(canon_name)
        proto = copy.deepcopy(defs[0][2])
        proto.name = canon_name
        canonical_for_key[ckey] = canon_name
        new_inits[_lca(uses)].append(proto)

    for path, graph in graphs:
        locally_produced = {vi.name for vi in graph.input}
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if not inp or inp in locally_produced:
                    continue
                ckey = _resolve_init_key(path, inp)
                if ckey is not None and ckey in canonical_for_key:
                    node.input[i] = canonical_for_key[ckey]
            locally_produced.update(o for o in node.output if o)
        for out in graph.output:
            if out.name not in locally_produced:
                ckey = _resolve_init_key(path, out.name)
                if ckey is not None and ckey in canonical_for_key:
                    out.name = canonical_for_key[ckey]

    for path, graph in graphs:
        del graph.initializer[:]
        graph.initializer.extend(new_inits[path])


def _slim_final_model(model: onnx.ModelProto, *, downcast_slice_i32: bool = True) -> onnx.ModelProto:
    """Run onnxslim, optionally downcast Slice indices to INT32, re-hoist shared constants, strip value_info, normalize opset."""
    slimmed = slim(
        model=model,
        no_shape_infer=False,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        verbose=False,
    )
    if not isinstance(slimmed, onnx.ModelProto):
        raise RuntimeError("In-memory onnxslim did not return a ModelProto")
    if downcast_slice_i32:
        _downcast_slice_indices_to_int32(slimmed)
    _deduplicate_and_hoist_initializers(slimmed)
    _strip_value_info(slimmed)
    _normalize_default_opset(slimmed)
    return slimmed


def _torch_export_to_onnx(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    downcast_slice_i32: bool = True,
) -> onnx.ModelProto:
    """Trace a PyTorch module in memory with TorchScript ONNX export and finalize with onnxslim."""
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*legacy TorchScript-based ONNX export.*",
        )
        torch.onnx.export(
            module.eval(),
            args,
            buf,
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    raw = onnx.load_from_string(buf.getvalue())
    return _slim_final_model(raw, downcast_slice_i32=downcast_slice_i32)


# ==========================================
# 2. Target: Simple Fixed-Ratio Resampler
# ==========================================
class SimpleResamplerModel(nn.Module):
    def __init__(
        self,
        scale_factor: float,
        *,
        use_mean_pooling: bool = True,
        input_dtype: torch.dtype = torch.int16,
        output_dtype: torch.dtype = torch.int16,
    ):
        super().__init__()
        self.scale_factor = float(scale_factor)
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        inverse_scale = 1.0 / self.scale_factor
        is_downsample = self.scale_factor < 1.0
        is_integer_factor = abs(inverse_scale - round(inverse_scale)) < 1e-6
        self.use_mean_pooling = is_downsample and is_integer_factor and use_mean_pooling
        if self.use_mean_pooling:
            self.stride = int(round(inverse_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mean_pooling:
            x = x.reshape(1, 1, -1, self.stride)
            x = x.to(torch.float32)
            out = torch.sum(x, dim=-1, keepdim=False) * self.scale_factor
        else:
            if self.input_dtype == torch.int16:
                x = x.float()
            out = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        out = out.clamp(min=-32768.0, max=32767.0)
        if self.output_dtype == torch.int16:
            out = out.to(torch.int16)
        return out


def export_simple_resampler_to_onnx(
    *,
    input_length: int = SIMPLE_INPUT_LENGTH,
    resample_ratio: float = SIMPLE_RESAMPLE_RATIO,
    dynamic: bool = SIMPLE_DYNAMIC,
    use_mean_pooling: bool = SIMPLE_USE_MEAN_POOLING,
    input_dtype: str = SIMPLE_INPUT_DTYPE,
    output_dtype: str = SIMPLE_OUTPUT_DTYPE,
    output_path: Path | None = None,
) -> Path:
    in_torch = _DTYPE_TO_TORCH[input_dtype]
    out_torch = _DTYPE_TO_TORCH[output_dtype]
    model = SimpleResamplerModel(
        resample_ratio,
        use_mean_pooling=use_mean_pooling,
        input_dtype=in_torch,
        output_dtype=out_torch,
    ).eval()
    dummy = torch.ones(1, 1, input_length, dtype=in_torch)
    dynamic_axes = {"input_audio": {2: "audio_len"}, "output_audio": {2: "resampled_len"}} if dynamic else None
    onnx_model = _torch_export_to_onnx(
        model,
        (dummy,),
        input_names=["input_audio"],
        output_names=["output_audio"],
        dynamic_axes=dynamic_axes,
    )
    _assert_structural_hygiene(onnx_model, "simple")
    default_name = f"resampler_{input_length}_to_{int(round(input_length * resample_ratio))}.onnx"
    save_path = output_path if output_path is not None else (_SCRIPT_DIR / default_name)
    _atomic_save_onnx(onnx_model, save_path)
    print(f"Simple resampler exported to: {save_path} ({save_path.stat().st_size} bytes, {len(onnx_model.graph.node)} nodes)")
    return save_path


def verify_simple_resampler_onnx(
    onnx_path: Path,
    *,
    input_length: int = SIMPLE_INPUT_LENGTH,
    resample_ratio: float = SIMPLE_RESAMPLE_RATIO,
    input_dtype: str = SIMPLE_INPUT_DTYPE,
    output_dtype: str = SIMPLE_OUTPUT_DTYPE,
) -> None:
    model = onnx.load(str(onnx_path), load_external_data=False)
    _assert_structural_hygiene(model, onnx_path.name)
    sess = onnxruntime.InferenceSession(str(onnx_path), _make_session_options(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(20261127)
    if input_dtype == "F32":
        sample = rng.standard_normal((1, 1, input_length)).astype(np.float32)
    else:
        sample = rng.integers(-32768, 32768, (1, 1, input_length), dtype=np.int16)
    out = sess.run(None, {sess.get_inputs()[0].name: sample})[0]
    expected_len = int(round(input_length * resample_ratio))
    if out.dtype != _DTYPE_TO_NP[output_dtype] or abs(out.shape[2] - expected_len) > 1 or not np.isfinite(out).all():
        raise RuntimeError(f"Simple resampler verification failed: shape={out.shape}, dtype={out.dtype}, expected_len={expected_len}")
    print(f"SUCCESS: {onnx_path.name} passed simple resampler verification (in={sample.shape}, out={out.shape}, dtype={out.dtype})")


# ==========================================
# 3. Target: Streaming Capture Frontend
# ==========================================
@dataclass(frozen=True)
class CaptureOperatingPointSpec:
    name: str
    input_rate: int
    channels: int
    mix_first_two: bool
    input_dtype: str  # "F32" or "INT16"


CAPTURE_OPERATING_POINTS: dict[str, CaptureOperatingPointSpec] = {
    "48k_1ch_F32": CaptureOperatingPointSpec("48k_1ch_F32", FIR_INPUT_SAMPLE_RATE, 1, False, "F32"),
    "48k_1ch_I16": CaptureOperatingPointSpec("48k_1ch_I16", FIR_INPUT_SAMPLE_RATE, 1, False, "INT16"),
    "48k_8ch_F32": CaptureOperatingPointSpec("48k_8ch_F32", FIR_INPUT_SAMPLE_RATE, 8, True, "F32"),
    "48k_8ch_I16": CaptureOperatingPointSpec("48k_8ch_I16", FIR_INPUT_SAMPLE_RATE, 8, True, "INT16"),
    "16k_1ch_F32": CaptureOperatingPointSpec("16k_1ch_F32", OUTPUT_SAMPLE_RATE, 1, False, "F32"),
    "16k_1ch_I16": CaptureOperatingPointSpec("16k_1ch_I16", OUTPUT_SAMPLE_RATE, 1, False, "INT16"),
    "16k_8ch_F32": CaptureOperatingPointSpec("16k_8ch_F32", OUTPUT_SAMPLE_RATE, 8, True, "F32"),
    "16k_8ch_I16": CaptureOperatingPointSpec("16k_8ch_I16", OUTPUT_SAMPLE_RATE, 8, True, "INT16"),
}
VALID_CAPTURE_MODES = (*CAPTURE_OPERATING_POINTS.keys(), "unified")


def kaiser_lowpass(input_rate: int) -> np.ndarray:
    """Design the unit-DC-gain symmetric Kaiser-windowed sinc FIR in float64, stored as float32."""
    if input_rate <= 0:
        raise ValueError(f"input_rate must be > 0, got {input_rate}")
    if FIR_TAPS < 3 or FIR_TAPS % 2 == 0:
        raise ValueError(f"FIR_TAPS must be an odd integer >= 3, got {FIR_TAPS}")
    fc = CUTOFF_HZ / float(input_rate)
    mid = (FIR_TAPS - 1) * 0.5
    n = np.arange(FIR_TAPS, dtype=np.float64) - mid
    r = n * (1.0 / mid)
    h = 2.0 * fc * np.sinc(2.0 * fc * n) * np.i0(KAISER_BETA * np.sqrt(np.maximum(0.0, 1.0 - r * r)))
    h *= 1.0 / np.i0(KAISER_BETA)
    h *= 1.0 / h.sum()
    return h.astype(np.float32)


def _default_capture_output_path(mode: str) -> Path:
    if mode in ("unified", "48k_1ch_F32"):
        return _SCRIPT_DIR / f"capture_{FIR_INPUT_SAMPLE_RATE}_to_{OUTPUT_SAMPLE_RATE}.onnx"
    if mode == "16k_1ch_F32":
        return _SCRIPT_DIR / f"capture_{OUTPUT_SAMPLE_RATE}_to_{OUTPUT_SAMPLE_RATE}.onnx"
    spec = CAPTURE_OPERATING_POINTS[mode]
    dtype_tag = "i16" if spec.input_dtype == "INT16" else "f32"
    return _SCRIPT_DIR / f"capture_{spec.input_rate}_to_{OUTPUT_SAMPLE_RATE}_{spec.channels}ch_{dtype_tag}.onnx"


class CaptureResampler(nn.Module):
    def __init__(
        self,
        decimation: int,
        taps: np.ndarray,
        *,
        channels: int = 1,
        mix_first_two: bool = False,
        input_dtype: torch.dtype = torch.float32,
        io_dtype: torch.dtype = torch.float32,
        sanitize_input: bool = True,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if mix_first_two and channels < 2:
            raise ValueError(f"mix_first_two=True requires channels >= 2, got {channels}")
        self.decimation = decimation
        self.channels = channels
        self.mix_first_two = mix_first_two
        self.input_dtype = input_dtype
        self.io_dtype = io_dtype
        self.sanitize_input = sanitize_input
        if input_dtype == torch.int16:
            if mix_first_two:
                self.register_buffer("pcm16_mix_scale", torch.tensor(1.0 / 65536.0, dtype=torch.float32))
            else:
                self.register_buffer("pcm16_scale", torch.tensor(1.0 / 32768.0, dtype=torch.float32))
        elif sanitize_input:
            self.register_buffer("zero", torch.tensor(0.0, dtype=torch.float32))
            self.register_buffer("pos_inf", torch.tensor(float("inf"), dtype=torch.float32))
            if mix_first_two:
                self.register_buffer("mix_scale", torch.tensor(0.5, dtype=torch.float32))
        if decimation > 1:
            self.history = int(taps.size - 1)
            k_poly = (int(taps.size) + decimation - 1) // decimation
            pad_lead = decimation * k_poly - int(taps.size)
            self.cache_start = (decimation - 1) - pad_lead
            w_pad = np.concatenate([np.zeros(pad_lead, dtype=np.float32), taps])
            w_poly = np.ascontiguousarray(w_pad.reshape(k_poly, decimation).T.reshape(1, decimation, k_poly), dtype=np.float32)
            self.register_buffer("weight", torch.from_numpy(w_poly))

    def forward(self, pcm: torch.Tensor, cache: torch.Tensor | None = None):
        if self.input_dtype == torch.int16:
            if self.channels == 1:
                x = pcm.to(torch.float32) * self.pcm16_scale
            elif self.mix_first_two:
                ch0 = pcm[..., 0::self.channels].to(torch.float32)
                ch1 = pcm[..., 1::self.channels].to(torch.float32)
                x = (ch0 + ch1) * self.pcm16_mix_scale
            else:
                x = pcm[..., 0::self.channels].to(torch.float32) * self.pcm16_scale
        elif not self.sanitize_input:
            x = pcm
        elif self.channels == 1:
            x = torch.where(pcm.abs() < self.pos_inf, pcm, self.zero).clamp(min=-1.0, max=1.0)
        elif self.mix_first_two:
            ch0 = pcm[..., 0::self.channels]
            ch1 = pcm[..., 1::self.channels]
            fin0 = torch.where(ch0.abs() < self.pos_inf, ch0, self.zero)
            fin1 = torch.where(ch1.abs() < self.pos_inf, ch1, self.zero)
            x = ((fin0 + fin1) * self.mix_scale).clamp(min=-1.0, max=1.0)
        else:
            ch0 = pcm[..., 0::self.channels]
            x = torch.where(ch0.abs() < self.pos_inf, ch0, self.zero).clamp(min=-1.0, max=1.0)

        if self.decimation == 1:
            return x.to(self.io_dtype) if self.io_dtype != torch.float32 else x

        cache_tail = cache[..., self.cache_start:] if self.cache_start > 0 else cache
        cache_f32 = cache_tail.to(torch.float32) if self.io_dtype != torch.float32 else cache_tail
        seq = torch.cat([cache_f32, x], dim=-1)
        seq_poly = seq.reshape(1, -1, self.decimation).transpose(1, 2)
        y = F.conv1d(seq_poly, self.weight, stride=1).clamp(min=-1.0, max=1.0)
        cache_out = seq[..., -self.history:]
        if self.io_dtype != torch.float32:
            return y.to(self.io_dtype), cache_out.to(self.io_dtype)
        return y, cache_out


def _export_capture_core_in_memory(
    input_rate: int,
    *,
    channels: int = 1,
    mix_first_two: bool = False,
    input_dtype: torch.dtype = torch.float32,
    io_dtype: torch.dtype = torch.float32,
    sanitize_input: bool = True,
) -> tuple[onnx.ModelProto, int, np.ndarray]:
    if OUTPUT_SAMPLE_RATE <= 0:
        raise ValueError(f"OUTPUT_SAMPLE_RATE must be > 0, got {OUTPUT_SAMPLE_RATE}")
    decimation = input_rate // OUTPUT_SAMPLE_RATE
    if decimation <= 0 or decimation * OUTPUT_SAMPLE_RATE != input_rate:
        raise ValueError(f"{input_rate} Hz is not an integer multiple of {OUTPUT_SAMPLE_RATE} Hz")
    taps = kaiser_lowpass(input_rate)
    model = CaptureResampler(
        decimation,
        taps,
        channels=channels,
        mix_first_two=mix_first_two,
        input_dtype=input_dtype,
        io_dtype=io_dtype,
        sanitize_input=sanitize_input,
    ).eval()
    in_name = "pcm16" if input_dtype == torch.int16 else "pcm"
    in_len_dim = "pcm16_len" if input_dtype == torch.int16 else "pcm_len"
    pcm = torch.zeros(1, 1, DUMMY_OUTPUT_LENGTH * decimation * channels, dtype=input_dtype)
    if decimation == 1:
        args: tuple[torch.Tensor, ...] = (pcm,)
        input_names, output_names = [in_name], ["audio"]
        dynamic_axes = {in_name: {2: in_len_dim}, "audio": {2: "audio_len"}}
    else:
        cache = torch.zeros(1, 1, model.history, dtype=io_dtype)
        args = (pcm, cache)
        input_names, output_names = [in_name, "cache"], ["audio", "cache_out"]
        dynamic_axes = {in_name: {2: in_len_dim}, "audio": {2: "audio_len"}}
    slimmed = _torch_export_to_onnx(
        model, args, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes
    )
    return slimmed, decimation, taps


def _build_f32_frontend_branch() -> onnx.GraphProto:
    mix_g = helper.make_graph(
        [
            helper.make_node("Slice", ["pcm", "capture_slice_zero", "capture_slice_max", "capture_slice_two", "capture_stride_i32"], ["capture_mix_ch0"]),
            helper.make_node("Slice", ["pcm", "capture_one_i32", "capture_slice_max", "capture_slice_two", "capture_stride_i32"], ["capture_mix_ch1"]),
            helper.make_node("Abs", ["capture_mix_ch0"], ["capture_mix_abs0"]),
            helper.make_node("Less", ["capture_mix_abs0", "capture_pos_inf"], ["capture_mix_ok0"]),
            helper.make_node("Where", ["capture_mix_ok0", "capture_mix_ch0", "capture_zero"], ["capture_mix_fin0"]),
            helper.make_node("Abs", ["capture_mix_ch1"], ["capture_mix_abs1"]),
            helper.make_node("Less", ["capture_mix_abs1", "capture_pos_inf"], ["capture_mix_ok1"]),
            helper.make_node("Where", ["capture_mix_ok1", "capture_mix_ch1", "capture_zero"], ["capture_mix_fin1"]),
            helper.make_node("Add", ["capture_mix_fin0", "capture_mix_fin1"], ["capture_mix_sum"]),
            helper.make_node("Mul", ["capture_mix_sum", "capture_half"], ["capture_mix_output"]),
        ],
        "capture_mix_branch", [],
        [helper.make_tensor_value_info("capture_mix_output", TensorProto.FLOAT, [1, 1, "frames"])],
    )
    mono_g = helper.make_graph(
        [
            helper.make_node("Slice", ["pcm", "capture_slice_zero", "capture_slice_max", "capture_slice_two", "capture_stride_i32"], ["capture_mono_selected"]),
            helper.make_node("Abs", ["capture_mono_selected"], ["capture_mono_abs"]),
            helper.make_node("Less", ["capture_mono_abs", "capture_pos_inf"], ["capture_mono_is_finite"]),
            helper.make_node("Where", ["capture_mono_is_finite", "capture_mono_selected", "capture_zero"], ["capture_mono_output"]),
        ],
        "capture_mono_branch", [],
        [helper.make_tensor_value_info("capture_mono_output", TensorProto.FLOAT, [1, 1, "frames"])],
    )
    return helper.make_graph(
        [
            helper.make_node("If", ["mix_first_two"], ["capture_selected"], then_branch=mix_g, else_branch=mono_g),
            helper.make_node("Clip", ["capture_selected", "capture_min", "capture_max"], ["capture_f32_mono"]),
        ],
        "capture_pcm_float_input", [],
        [helper.make_tensor_value_info("capture_f32_mono", TensorProto.FLOAT, [1, 1, "frames"])],
    )


def _build_pcm16_frontend_branch() -> onnx.GraphProto:
    mix_g = helper.make_graph(
        [
            helper.make_node("Slice", ["pcm16", "capture_slice_zero", "capture_slice_max", "capture_slice_two", "capture_stride_i32"], ["capture_pcm16_ch0"]),
            helper.make_node("Slice", ["pcm16", "capture_one_i32", "capture_slice_max", "capture_slice_two", "capture_stride_i32"], ["capture_pcm16_ch1"]),
            helper.make_node("Cast", ["capture_pcm16_ch0"], ["capture_pcm16_f0"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["capture_pcm16_ch1"], ["capture_pcm16_f1"], to=TensorProto.FLOAT),
            helper.make_node("Add", ["capture_pcm16_f0", "capture_pcm16_f1"], ["capture_pcm16_sum"]),
            helper.make_node("Mul", ["capture_pcm16_sum", "capture_pcm16_mix_scale"], ["capture_pcm16_mix_output"]),
        ],
        "capture_pcm16_mix_branch", [],
        [helper.make_tensor_value_info("capture_pcm16_mix_output", TensorProto.FLOAT, [1, 1, "frames"])],
    )
    mono_g = helper.make_graph(
        [
            helper.make_node("Slice", ["pcm16", "capture_slice_zero", "capture_slice_max", "capture_slice_two", "capture_stride_i32"], ["capture_pcm16_mono_i16"]),
            helper.make_node("Cast", ["capture_pcm16_mono_i16"], ["capture_pcm16_mono_f32"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["capture_pcm16_mono_f32", "capture_pcm16_scale"], ["capture_pcm16_mono_output"]),
        ],
        "capture_pcm16_mono_branch", [],
        [helper.make_tensor_value_info("capture_pcm16_mono_output", TensorProto.FLOAT, [1, 1, "frames"])],
    )
    return helper.make_graph(
        [helper.make_node("If", ["mix_first_two"], ["capture_pcm16_mono"], then_branch=mix_g, else_branch=mono_g)],
        "capture_pcm16_normalize", [],
        [helper.make_tensor_value_info("capture_pcm16_mono", TensorProto.FLOAT, [1, 1, "frames"])],
        [
            helper.make_tensor("capture_pcm16_scale", TensorProto.FLOAT, [], [1.0 / 32768.0]),
            helper.make_tensor("capture_pcm16_mix_scale", TensorProto.FLOAT, [], [1.0 / 65536.0]),
        ],
    )


def build_unified_capture_frontend(fir_core: onnx.ModelProto, input_rate: int) -> onnx.ModelProto:
    """Wrap the PyTorch-exported polyphase FIR core in android_capture_frontend and finalize with onnxslim."""
    fir = copy.deepcopy(fir_core.graph)
    in_map = {v.name: v for v in fir.input}
    out_map = {v.name: v for v in fir.output}
    if set(in_map) != {"pcm", "cache"} or set(out_map) != {"audio", "cache_out"}:
        raise ValueError(f"Unexpected FIR core I/O: in={list(in_map)}, out={list(out_map)}")
    if not any(init.name == "weight" for init in fir.initializer):
        raise ValueError("FIR core is missing 'weight' initializer")

    cache_in = copy.deepcopy(in_map["cache"])
    audio_out = copy.deepcopy(out_map["audio"])
    cache_out = copy.deepcopy(out_map["cache_out"])

    initializers = [
        helper.make_tensor("capture_slice_zero", TensorProto.INT32, [1], [0]),
        helper.make_tensor("capture_one_i32", TensorProto.INT32, [1], [1]),
        helper.make_tensor("capture_slice_two", TensorProto.INT32, [1], [2]),
        helper.make_tensor("capture_slice_max", TensorProto.INT32, [1], [2**31 - 1]),
        helper.make_tensor("capture_scalar_shape", TensorProto.INT64, [0], []),
        helper.make_tensor("capture_output_rate", TensorProto.INT32, [1], [OUTPUT_SAMPLE_RATE]),
        helper.make_tensor("capture_zero", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("capture_half", TensorProto.FLOAT, [], [0.5]),
        helper.make_tensor("capture_min", TensorProto.FLOAT, [], [-1.0]),
        helper.make_tensor("capture_max", TensorProto.FLOAT, [], [1.0]),
        numpy_helper.from_array(np.array(np.inf, dtype=np.float32), name="capture_pos_inf"),
    ]
    del fir.input[:]
    del fir.value_info[:]
    in_remap = {"pcm": "capture_mono", "audio": "capture_fir_audio", "cache_out": "capture_fir_cache"}
    out_remap = {"audio": "capture_fir_audio", "cache_out": "capture_fir_cache"}
    for node in fir.node:
        node.input[:] = [in_remap.get(v, v) for v in node.input]
        node.output[:] = [out_remap.get(v, v) for v in node.output]
    fir.output[0].name = "capture_fir_audio"
    fir.output[1].name = "capture_fir_cache"
    fir.name = "capture_antialias_fir"

    direct_cache_vi = copy.deepcopy(cache_out)
    direct_cache_vi.name = "capture_direct_cache"
    direct = helper.make_graph(
        [
            helper.make_node("Cast", ["capture_mono"], ["capture_direct_audio"], to=TensorProto.FLOAT16),
            helper.make_node("Identity", ["cache"], ["capture_direct_cache"]),
        ],
        "capture_same_rate", [],
        [
            helper.make_tensor_value_info("capture_direct_audio", TensorProto.FLOAT16, [1, 1, "audio_len"]),
            direct_cache_vi,
        ],
    )
    nodes = [
        helper.make_node("Cast", ["input_rate"], ["capture_input_rate_i32"], to=TensorProto.INT32),
        helper.make_node("Cast", ["channel_stride"], ["capture_stride_i32"], to=TensorProto.INT32),
        helper.make_node(
            "If", ["use_pcm16"], ["capture_mono"],
            then_branch=_build_pcm16_frontend_branch(), else_branch=_build_f32_frontend_branch(),
            name="capture_choose_pcm_input",
        ),
        helper.make_node("Equal", ["capture_input_rate_i32", "capture_output_rate"], ["capture_same_rate_vector"]),
        helper.make_node("Reshape", ["capture_same_rate_vector", "capture_scalar_shape"], ["capture_same_rate"]),
        helper.make_node("If", ["capture_same_rate"], ["audio", "cache_out"], then_branch=direct, else_branch=fir),
    ]
    graph_inputs = [
        helper.make_tensor_value_info("pcm", TensorProto.FLOAT, [1, 1, "pcm_len"]),
        cache_in,
        helper.make_tensor_value_info("input_rate", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("channel_stride", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("mix_first_two", TensorProto.BOOL, []),
        helper.make_tensor_value_info("pcm16", TensorProto.INT16, [1, 1, "pcm16_len"]),
        helper.make_tensor_value_info("use_pcm16", TensorProto.BOOL, []),
    ]
    model = onnx.ModelProto()
    model.ir_version = IR_VERSION
    model.producer_name = fir_core.producer_name or "pytorch"
    model.producer_version = fir_core.producer_version or torch.__version__.split("+")[0]
    op = model.opset_import.add()
    op.version = OPSET
    model.graph.CopyFrom(helper.make_graph(nodes, "android_capture_frontend", graph_inputs, [audio_out, cache_out], initializers))
    helper.set_model_props(
        model,
        {
            "android_pcm_dtype": "F32",
            "state_dtype": "F16",
            "out_audio_dtype": "F16",
            "fir_compute_dtype": "F32",
            "capture_output_rate": str(OUTPUT_SAMPLE_RATE),
            "capture_fir_input_rate": str(input_rate),
            "capture_input_layout": "interleaved_f32",
            "capture_frontend_v2": "1",
            "pcm16_conversion_in_graph": "1",
        },
    )
    model = _slim_final_model(model)
    _assert_structural_hygiene(model, "unified")
    return model


def export_capture_unified_to_onnx(
    input_rate: int = FIR_INPUT_SAMPLE_RATE,
    output_path: Path | None = None,
) -> tuple[Path, int, np.ndarray]:
    fir_core, decimation, taps = _export_capture_core_in_memory(input_rate, sanitize_input=False, io_dtype=torch.float16)
    unified = build_unified_capture_frontend(fir_core, input_rate)
    save_path = output_path if output_path is not None else _default_capture_output_path("unified")
    _atomic_save_onnx(unified, save_path)
    print(f"Unified capture frontend exported to: {save_path} ({save_path.stat().st_size} bytes)")
    return save_path, decimation, taps


def export_capture_operating_point_to_onnx(
    mode: str,
    *,
    output_dtype: str = CAPTURE_OUTPUT_DTYPE,
    output_path: Path | None = None,
) -> tuple[Path, CaptureOperatingPointSpec, int, np.ndarray]:
    if mode not in CAPTURE_OPERATING_POINTS:
        raise ValueError(f"Unsupported capture mode {mode!r}; expected one of {tuple(CAPTURE_OPERATING_POINTS)}")
    if output_dtype not in ("F32", "F16"):
        raise ValueError(f"Unsupported output_dtype {output_dtype!r}; expected 'F32' or 'F16'")
    spec = CAPTURE_OPERATING_POINTS[mode]
    model, decimation, taps = _export_capture_core_in_memory(
        spec.input_rate,
        channels=spec.channels,
        mix_first_two=spec.mix_first_two,
        input_dtype=_DTYPE_TO_TORCH[spec.input_dtype],
        io_dtype=_DTYPE_TO_TORCH[output_dtype],
        sanitize_input=True,
    )
    props = {
        "operating_point": spec.name,
        "input_sample_rate": str(spec.input_rate),
        "output_sample_rate": str(OUTPUT_SAMPLE_RATE),
        "input_channels": str(spec.channels),
        "mix_first_two": "1" if spec.mix_first_two else "0",
        "input_audio_dtype": spec.input_dtype,
        "output_audio_dtype": output_dtype,
        **({"state_dtype": output_dtype, "fir_compute_dtype": "F32", "fir_taps": str(FIR_TAPS)} if decimation > 1 else {}),
    }
    helper.set_model_props(model, props)
    _assert_structural_hygiene(model, spec.name)
    save_path = output_path if output_path is not None else _default_capture_output_path(mode)
    _atomic_save_onnx(model, save_path)
    print(
        f"Capture operating-point model [{mode}, out={output_dtype}] exported to: "
        f"{save_path} ({save_path.stat().st_size} bytes, {len(model.graph.node)} nodes)"
    )
    return save_path, spec, decimation, taps


def reference_capture_stream(chunks: list[np.ndarray], decimation: int, taps: np.ndarray) -> np.ndarray:
    """Sample-by-sample float32 streaming reference matching the former C++ PcmResampler."""
    t = taps.size
    line = np.zeros(2 * t, dtype=np.float32)
    pos = phase = 0
    out: list[np.float32] = []
    for chunk in chunks:
        for s in chunk:
            s = np.float32(0.0) if not np.isfinite(s) else np.float32(min(1.0, max(-1.0, float(s))))
            if decimation > 1:
                line[pos] = line[pos + t] = s
                pos = (pos + 1) % t
                if phase != decimation - 1:
                    phase += 1
                    continue
                phase = 0
                s = np.float32(min(1.0, max(-1.0, float(np.dot(taps, line[pos:pos + t])))))
            out.append(s)
    return np.asarray(out, dtype=np.float32)


def _reference_operating_point_mono(packed: np.ndarray, spec: CaptureOperatingPointSpec) -> np.ndarray:
    frames = packed.reshape(-1, spec.channels)
    if spec.input_dtype == "INT16":
        if spec.mix_first_two:
            ch = frames[:, :2].astype(np.float32) * np.float32(1.0 / 32768.0)
            return ((ch[:, 0] + ch[:, 1]) * np.float32(0.5)).astype(np.float32)
        return (frames[:, 0].astype(np.float32) * np.float32(1.0 / 32768.0)).astype(np.float32)
    selected = frames[:, :2] if spec.mix_first_two else frames[:, :1]
    finite = np.where(np.isfinite(selected), selected, np.float32(0.0))
    mono = finite.mean(axis=1) if spec.mix_first_two else finite[:, 0]
    return np.clip(mono, np.float32(-1.0), np.float32(1.0)).astype(np.float32)


def _verify_frequency_response(
    run_block: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    input_rate: int,
    decimation: int,
    hist_len: int,
    cache_dtype: type[np.floating],
    input_dtype: str = "F32",
    channels: int = 1,
    mix_first_two: bool = False,
    label: str = "capture",
) -> dict[str, float]:
    frames_fir = DUMMY_OUTPUT_LENGTH * decimation
    t_tone = np.arange(frames_fir * 2, dtype=np.float64) / float(input_rate)
    stopband_db: dict[str, float] = {}
    for freq_hz in (1000.0, 6800.0, 7200.0, 8000.0, 12000.0, 18000.0, 23000.0):
        tone_1ch = 0.8 * np.sin(2.0 * np.pi * freq_hz * t_tone)
        tone_mono = (
            np.clip(np.round(tone_1ch * 32768.0), -32768, 32767).astype(np.int16)
            if input_dtype == "INT16"
            else tone_1ch.astype(np.float32)
        )
        if channels == 1:
            tone_packed = tone_mono
        else:
            tone_packed = np.zeros((tone_mono.size, channels), dtype=tone_mono.dtype)
            tone_packed[:, 0] = tone_mono
            if mix_first_two:
                tone_packed[:, 1] = tone_mono
        c_tone = np.zeros((1, 1, hist_len), dtype=cache_dtype)
        outs_tone: list[np.ndarray] = []
        for b_idx in range(2):
            chunk = tone_packed[b_idx * frames_fir:(b_idx + 1) * frames_fir].reshape(1, 1, -1)
            out_t, c_tone = run_block(chunk, c_tone)
            outs_tone.append(out_t.reshape(-1).astype(np.float64))
        steady = outs_tone[1][200:]
        gain_db = float(20.0 * np.log10((np.sqrt(np.mean(steady * steady)) + 1e-15) / (0.8 / np.sqrt(2.0))))
        if freq_hz <= 6800.0:
            if abs(gain_db) > 0.15:
                raise RuntimeError(f"{label} passband {freq_hz} Hz gain {gain_db:.2f} dB exceeds +-0.15 dB")
        elif freq_hz == 7200.0:
            if not (-4.0 <= gain_db <= -3.0):
                raise RuntimeError(f"{label} transition {freq_hz} Hz gain {gain_db:.2f} dB outside [-4.0, -3.0] dB")
        else:
            stopband_db[f"{int(freq_hz)}Hz"] = round(-gain_db, 2)
            if -gain_db < 75.0:
                raise RuntimeError(f"{label} stopband {freq_hz} Hz rejection {-gain_db:.2f} dB < 75 dB")
    return stopband_db


def _assert_f16_local_ulp(actual: np.ndarray, ref: np.ndarray, msg: str) -> None:
    local_ulp = np.maximum(np.abs(np.spacing(ref)).astype(np.float32), np.float32(2.0 ** -14))
    max_ulp = float((np.abs(actual.astype(np.float32) - ref.astype(np.float32)) / local_ulp).max())
    if max_ulp > 1.0:
        raise RuntimeError(f"{msg} ({max_ulp:.2f} ULP)")


def verify_capture_operating_point_onnx(
    onnx_path: Path,
    spec: CaptureOperatingPointSpec,
    output_dtype: str,
    decimation: int,
    taps: np.ndarray,
    *,
    reference_onnx_path: Path | None = None,
) -> None:
    model = onnx.load(str(onnx_path), load_external_data=False)
    _assert_structural_hygiene(model, spec.name)
    if any(node.op_type == "If" for node in model.graph.node):
        raise RuntimeError(f"Operating-point model {spec.name} unexpectedly contains If nodes")
    in_name = "pcm16" if spec.input_dtype == "INT16" else "pcm"
    in_dt, out_dt = _DTYPE_TO_ONNX[spec.input_dtype], _DTYPE_TO_ONNX[output_dtype]
    exp_in = [(in_name, in_dt)] + ([("cache", out_dt)] if decimation > 1 else [])
    exp_out = [("audio", out_dt)] + ([("cache_out", out_dt)] if decimation > 1 else [])
    if [(v.name, v.type.tensor_type.elem_type) for v in model.graph.input] != exp_in:
        raise RuntimeError(f"Input contract mismatch for {spec.name}")
    if [(v.name, v.type.tensor_type.elem_type) for v in model.graph.output] != exp_out:
        raise RuntimeError(f"Output contract mismatch for {spec.name}")
    if spec.input_dtype == "F32" and output_dtype == "F32" and any(n.op_type == "Cast" for n in model.graph.node):
        raise RuntimeError(f"Pure F32 operating point {spec.name} should not contain Cast nodes")

    session = onnxruntime.InferenceSession(str(onnx_path), _make_session_options(), providers=["CPUExecutionProvider"])
    ref_session = (
        onnxruntime.InferenceSession(str(reference_onnx_path), _make_session_options(), providers=["CPUExecutionProvider"])
        if reference_onnx_path is not None
        else None
    )
    out_np_dtype = _DTYPE_TO_NP[output_dtype]
    hist_len = int(taps.size - 1)
    rng = np.random.default_rng(20261116)
    frames_per_block = 800 * decimation
    if spec.input_dtype == "INT16":
        blocks = [rng.integers(-32768, 32768, (frames_per_block, spec.channels), dtype=np.int16) for _ in range(4)]
        blocks[0][0, :min(2, spec.channels)] = -32768
        blocks[0][1, :min(2, spec.channels)] = 32767
    else:
        blocks = [rng.uniform(-1.5, 1.5, (frames_per_block, spec.channels)).astype(np.float32) for _ in range(4)]
        blocks[1].reshape(-1)[:6] = [np.nan, np.inf, -np.inf, 1.0, -1.0, -0.0]

    if spec.name == "16k_1ch_I16":
        all_i16 = np.arange(-32768, 32768, dtype=np.int16)
        (got_all,) = session.run(None, {in_name: all_i16.reshape(1, 1, -1)})
        exp_all = (all_i16.astype(np.float32) * np.float32(1.0 / 32768.0)).astype(out_np_dtype).reshape(1, 1, -1)
        if not np.array_equal(got_all, exp_all):
            raise RuntimeError("16k_1ch_I16 exhaustive 65,536-value check failed")

    expected_f32 = reference_capture_stream([_reference_operating_point_mono(b, spec) for b in blocks], decimation, taps)
    got_parts: list[np.ndarray] = []
    cache = np.zeros((1, 1, hist_len), dtype=out_np_dtype)
    ref_cache = np.zeros((1, 1, hist_len), dtype=np.float16)
    empty_f32, empty_i16 = np.zeros((1, 1, 0), dtype=np.float32), np.zeros((1, 1, 0), dtype=np.int16)
    ref_ctrl = {
        "input_rate": np.array([spec.input_rate], dtype=np.int64),
        "channel_stride": np.array([spec.channels], dtype=np.int64),
        "mix_first_two": np.array(spec.mix_first_two, dtype=bool),
    }
    for blk in blocks:
        flat = blk.reshape(1, 1, -1)
        if decimation > 1:
            audio, cache = session.run(None, {in_name: flat, "cache": cache})
        else:
            (audio,) = session.run(None, {in_name: flat})
        if audio.dtype != out_np_dtype or not np.isfinite(audio).all():
            raise RuntimeError(f"{spec.name}: invalid audio output")
        got_parts.append(audio.reshape(-1))
        if ref_session is not None:
            ref_audio, ref_cache = ref_session.run(None, {
                "pcm": empty_f32 if spec.input_dtype == "INT16" else flat,
                "pcm16": flat if spec.input_dtype == "INT16" else empty_i16,
                "use_pcm16": np.array(spec.input_dtype == "INT16", dtype=bool),
                "cache": ref_cache,
                **ref_ctrl,
            })
            if output_dtype == "F16":
                if decimation == 1:
                    if not np.array_equal(audio, ref_audio):
                        raise RuntimeError(f"{spec.name} [F16]: mismatch vs {reference_onnx_path}")
                else:
                    _assert_f16_local_ulp(audio, ref_audio, f"{spec.name} [F16] mismatch vs {reference_onnx_path}")
                if decimation > 1 and not np.array_equal(cache, ref_cache):
                    raise RuntimeError(f"{spec.name} [F16]: cache mismatch vs {reference_onnx_path}")
            else:
                if float(np.max(np.abs(audio.astype(np.float32) - ref_audio.astype(np.float32)))) > float(np.spacing(np.float16(1.0))):
                    raise RuntimeError(f"{spec.name} [F32]: diff exceeds 1 full-scale F16 ULP vs {reference_onnx_path}")

    got_arr = np.concatenate(got_parts).astype(np.float32)
    max_diff = float(np.max(np.abs(got_arr - expected_f32)))
    tol = 1e-6 if output_dtype == "F32" else float(np.spacing(np.float16(1.0)))
    if got_arr.shape != expected_f32.shape or max_diff > tol:
        raise RuntimeError(f"{onnx_path.name} [{spec.name}, out={output_dtype}]: max |diff| {max_diff:.3e} > {tol:.3e}")
    stopband_msg = ""
    if decimation > 1:
        sb = _verify_frequency_response(
            lambda chunk, c: session.run(None, {in_name: chunk, "cache": c}),
            input_rate=spec.input_rate, decimation=decimation, hist_len=hist_len, cache_dtype=out_np_dtype,
            input_dtype=spec.input_dtype, channels=spec.channels, mix_first_two=spec.mix_first_two, label=spec.name,
        )
        stopband_msg = f", stopband_dB={sb}"
    print(f"SUCCESS: {onnx_path.name} [{spec.name}, out={output_dtype}] passed verification (max |diff| = {max_diff:.3e}{stopband_msg})")


def verify_capture_unified_onnx(
    onnx_path: Path,
    decimation: int,
    taps: np.ndarray,
    *,
    reference_onnx_path: Path | None = None,
) -> None:
    expected_props = (
        ("android_pcm_dtype", "F32"),
        ("state_dtype", "F16"),
        ("out_audio_dtype", "F16"),
        ("fir_compute_dtype", "F32"),
        ("capture_output_rate", str(OUTPUT_SAMPLE_RATE)),
        ("capture_fir_input_rate", str(OUTPUT_SAMPLE_RATE * decimation)),
        ("capture_input_layout", "interleaved_f32"),
        ("capture_frontend_v2", "1"),
        ("pcm16_conversion_in_graph", "1"),
    )
    model = onnx.load(str(onnx_path), load_external_data=False)
    _assert_structural_hygiene(model, "unified", required_props=expected_props)
    expected_inputs = [
        ("pcm", TensorProto.FLOAT), ("cache", TensorProto.FLOAT16), ("input_rate", TensorProto.INT64),
        ("channel_stride", TensorProto.INT64), ("mix_first_two", TensorProto.BOOL),
        ("pcm16", TensorProto.INT16), ("use_pcm16", TensorProto.BOOL),
    ]
    expected_outputs = [("audio", TensorProto.FLOAT16), ("cache_out", TensorProto.FLOAT16)]
    if [(v.name, v.type.tensor_type.elem_type) for v in model.graph.input] != expected_inputs:
        raise RuntimeError("Unified capture input contract mismatch")
    if [(v.name, v.type.tensor_type.elem_type) for v in model.graph.output] != expected_outputs:
        raise RuntimeError("Unified capture output contract mismatch")

    hist_len = int(taps.size - 1)
    session = onnxruntime.InferenceSession(str(onnx_path), _make_session_options(), providers=["CPUExecutionProvider"])
    ref_session = (
        onnxruntime.InferenceSession(str(reference_onnx_path), _make_session_options(), providers=["CPUExecutionProvider"])
        if reference_onnx_path is not None
        else None
    )
    empty_f32, empty_i16 = np.zeros((1, 1, 0), dtype=np.float32), np.zeros((1, 1, 0), dtype=np.int16)
    zero_cache = np.zeros((1, 1, hist_len), dtype=np.float16)
    rng = np.random.default_rng(20261116)
    fir_rate = OUTPUT_SAMPLE_RATE * decimation

    # Exhaustive 65,536-value INT16 mono check
    all_i16 = np.arange(-32768, 32768, dtype=np.int16)
    a16, c16 = session.run(None, {
        "pcm": empty_f32, "cache": zero_cache, "input_rate": np.array([OUTPUT_SAMPLE_RATE], np.int64),
        "channel_stride": np.array([1], np.int64), "mix_first_two": np.array(False),
        "pcm16": all_i16.reshape(1, 1, -1), "use_pcm16": np.array(True),
    })
    exp16 = (all_i16.astype(np.float32) * np.float32(1.0 / 32768.0)).astype(np.float16).reshape(1, 1, -1)
    if not np.array_equal(a16, exp16) or not np.array_equal(c16, zero_cache):
        raise RuntimeError("Exhaustive 65,536-value INT16 mono check failed")

    # Multi-rate / multi-channel / non-finite / PCM16 equivalence + parity vs reference
    for rate in (OUTPUT_SAMPLE_RATE, fir_rate):
        frames = DUMMY_OUTPUT_LENGTH * (rate // OUTPUT_SAMPLE_RATE)
        for stride, mix in ((1, False), (2, False), (2, True), (8, False), (8, True)):
            cache = np.zeros((1, 1, hist_len), dtype=np.float16)
            ref_cache = cache.copy()
            f32_blocks = [
                rng.uniform(-1.0, 1.0, (frames, stride)).astype(np.float32),
                np.zeros((frames, stride), dtype=np.float32),
                rng.uniform(-2.0, 2.0, (frames, stride)).astype(np.float32),
            ]
            f32_blocks[-1].reshape(-1)[:6] = [np.nan, np.inf, -np.inf, 1.0, -1.0, -0.0]
            ctrl = {"input_rate": np.array([rate], np.int64), "channel_stride": np.array([stride], np.int64), "mix_first_two": np.array(mix)}
            for packed in f32_blocks:
                feeds = {"pcm": packed.reshape(1, 1, -1), "cache": cache, "pcm16": empty_i16, "use_pcm16": np.array(False), **ctrl}
                audio, next_cache = session.run(None, feeds)
                if audio.dtype != np.float16 or next_cache.dtype != np.float16 or not np.isfinite(audio).all() or not np.isfinite(next_cache).all():
                    raise RuntimeError(f"Invalid output at rate={rate}, stride={stride}, mix={mix}")
                if rate == OUTPUT_SAMPLE_RATE and not np.array_equal(next_cache, cache):
                    raise RuntimeError("16 kHz fast path must preserve cache untouched")
                if ref_session is not None:
                    ref_audio, ref_next_cache = ref_session.run(None, {**feeds, "cache": ref_cache})
                    if not np.array_equal(next_cache, ref_next_cache):
                        raise RuntimeError(f"F32 cache mismatch vs {reference_onnx_path}")
                    if rate == OUTPUT_SAMPLE_RATE:
                        if not np.array_equal(audio, ref_audio):
                            raise RuntimeError(f"16k F32 parity mismatch vs {reference_onnx_path}")
                    else:
                        _assert_f16_local_ulp(audio, ref_audio, f"48k F32 parity mismatch vs {reference_onnx_path}")
                    ref_cache = ref_next_cache
                cache = next_cache

            pcm16 = rng.integers(-32768, 32768, (frames, stride), dtype=np.int16)
            pcm16.reshape(-1)[:3] = [-32768, 0, 32767]
            af32, cf32 = session.run(None, {"pcm": pcm16.astype(np.float32).reshape(1, 1, -1) * np.float32(1.0 / 32768.0), "cache": cache, "pcm16": empty_i16, "use_pcm16": np.array(False), **ctrl})
            ai16, ci16 = session.run(None, {"pcm": empty_f32, "cache": cache, "pcm16": pcm16.reshape(1, 1, -1), "use_pcm16": np.array(True), **ctrl})
            if not np.array_equal(ai16, af32) or not np.array_equal(ci16, cf32):
                raise RuntimeError(f"PCM16 vs F32 mismatch at rate={rate}, stride={stride}, mix={mix}")
            if ref_session is not None:
                ri16, rc16 = ref_session.run(None, {"pcm": empty_f32, "cache": cache, "pcm16": pcm16.reshape(1, 1, -1), "use_pcm16": np.array(True), **ctrl})
                if not np.array_equal(ci16, rc16):
                    raise RuntimeError(f"PCM16 cache mismatch vs {reference_onnx_path}")
                if rate == OUTPUT_SAMPLE_RATE:
                    if not np.array_equal(ai16, ri16):
                        raise RuntimeError(f"16k PCM16 parity mismatch vs {reference_onnx_path}")
                else:
                    _assert_f16_local_ulp(ai16, ri16, f"48k PCM16 parity mismatch vs {reference_onnx_path}")

    # Continuous unquantized F64 FIR audit + exact impulse group delay + frequency response
    frames_fir = DUMMY_OUTPUT_LENGTH * decimation
    t_total = np.arange(frames_fir * 4, dtype=np.float64) / float(fir_rate)
    chirp = (0.8 * np.sin(2.0 * np.pi * (100.0 * t_total + (7400.0 / (2.0 * t_total[-1])) * t_total * t_total))).astype(np.float32)
    stream_blocks = [*(chirp[i * frames_fir:(i + 1) * frames_fir] for i in range(4)), *(rng.uniform(-1.0, 1.0, frames_fir).astype(np.float32) for _ in range(4))]
    cont_hist_f32 = np.zeros(hist_len, dtype=np.float32)
    graph_cache_f16 = np.zeros((1, 1, hist_len), dtype=np.float16)
    fs_f16_ulp = float(np.spacing(np.float16(0.9995)))
    boundary_outs = (hist_len + decimation - 1) // decimation
    taps_f64 = taps.astype(np.float64)
    ctrl_fir = {"input_rate": np.array([fir_rate], np.int64), "channel_stride": np.array([1], np.int64), "mix_first_two": np.array(False)}
    max_abs_err = max_interior_ulp = max_boundary_ulp = 0.0
    for blk in stream_blocks:
        clean = np.clip(np.where(np.isfinite(blk), blk, np.float32(0.0)), np.float32(-1.0), np.float32(1.0))
        joined = np.concatenate([cont_hist_f32, clean])
        s0 = int(joined.size - ((DUMMY_OUTPUT_LENGTH - 1) * decimation + taps.size))
        idx_mat = (s0 + np.arange(DUMMY_OUTPUT_LENGTH)[:, None] * decimation) + np.arange(taps.size)[None, :]
        ref_f32 = np.clip(joined[idx_mat].astype(np.float64) @ taps_f64, -1.0, 1.0).astype(np.float32)
        cont_hist_f32 = joined[-hist_len:]
        got_audio, graph_cache_f16 = session.run(None, {"pcm": blk.reshape(1, 1, -1), "cache": graph_cache_f16, "pcm16": empty_i16, "use_pcm16": np.array(False), **ctrl_fir})
        got_f16 = got_audio.reshape(-1)
        ref_f16 = ref_f32.astype(np.float16)
        abs_diff = np.abs(got_f16.astype(np.float32) - ref_f32)
        local_ulp = np.maximum(np.abs(np.spacing(ref_f16)).astype(np.float32), np.float32(2.0 ** -14))
        interior_ulp = float((np.abs(got_f16[boundary_outs:].astype(np.float32) - ref_f16[boundary_outs:].astype(np.float32)) / local_ulp[boundary_outs:]).max())
        boundary_ulp = float(abs_diff[:boundary_outs].max()) / fs_f16_ulp
        max_abs_err, max_interior_ulp, max_boundary_ulp = max(max_abs_err, float(abs_diff.max())), max(max_interior_ulp, interior_ulp), max(max_boundary_ulp, boundary_ulp)
        if interior_ulp > 1.0 or boundary_ulp > 1.0:
            raise RuntimeError(f"FIR F16 precision drift: interior_ulp={interior_ulp:.2f}, boundary_ulp={boundary_ulp:.3f}")

    imp = np.zeros((1, 1, frames_fir), dtype=np.float32)
    imp[0, 0, frames_fir // 2] = np.float32(0.5)
    imp_out, _ = session.run(None, {"pcm": imp, "cache": zero_cache, "pcm16": empty_i16, "use_pcm16": np.array(False), **ctrl_fir})
    expected_delay = ((taps.size - 1) // 2) // decimation
    if int(np.argmax(np.abs(imp_out.reshape(-1).astype(np.float32)))) != (DUMMY_OUTPUT_LENGTH // 2) + expected_delay:
        raise RuntimeError("FIR group delay mismatch")

    sb = _verify_frequency_response(
        lambda chunk, c: session.run(None, {"pcm": chunk, "cache": c, "pcm16": empty_i16, "use_pcm16": np.array(False), **ctrl_fir}),
        input_rate=fir_rate, decimation=decimation, hist_len=hist_len, cache_dtype=np.float16, label="unified",
    )
    parity_msg = f", verified vs {reference_onnx_path.name}" if reference_onnx_path is not None else ""
    print(f"SUCCESS: {onnx_path.name} passed unified capture verification (max_err={max_abs_err:.3e}, interior_ulp={max_interior_ulp:.2f}, boundary_ulp={max_boundary_ulp:.4f}, delay={expected_delay}, stopband_dB={sb}{parity_msg})")


# ==========================================
# 4. Target: Multi-Rate Polyphase Resampler
# ==========================================
class MultiRateStage0(nn.Module):
    def __init__(self):
        super().__init__()
        f32 = torch.float32
        self.register_buffer("taps", torch.arange(-MAX_HALF, MAX_HALF + 1, dtype=torch.int64).view(1, -1))
        self.register_buffer("taps_f", torch.arange(-MAX_HALF, MAX_HALF + 1, dtype=f32).view(1, -1))
        self.register_buffer("out_rate", torch.tensor([OUTPUT_SAMPLE_RATE], dtype=torch.int64))
        self.register_buffer("inv_out_rate", torch.tensor([1.0 / OUTPUT_SAMPLE_RATE], dtype=f32))
        self.register_buffer("pi", torch.tensor([math.pi], dtype=f32))
        self.register_buffer("one", torch.tensor([1.0], dtype=f32))
        self.register_buffer("half_one", torch.tensor([0.5], dtype=f32))
        self.register_buffer("eight", torch.tensor([8.0], dtype=f32))
        self.register_buffer("zero", torch.tensor([0.0], dtype=f32))
        self.register_buffer("tiny", torch.tensor([1e-12], dtype=f32))
        self.register_buffer("zero_i", torch.tensor([0], dtype=torch.int64))

    def forward(self, audio, sample_rate, out_start, out_count):
        x = audio.reshape(-1).clamp(min=-1.0, max=1.0)
        n = x.shape[0]
        ratio = sample_rate.to(torch.float32) * self.inv_out_rate
        decimate = ratio > self.one
        cutoff = torch.reciprocal(ratio)
        half = torch.where(decimate, torch.round(ratio * self.eight), self.one)
        inv_half1 = torch.reciprocal(half + self.one)

        i = out_start + torch.arange(out_count[0], dtype=torch.int64)
        num = (i * sample_rate).view(-1, 1)
        base = num // self.out_rate
        frac = (num - base * self.out_rate).to(torch.float32) * self.inv_out_rate
        k = base + self.taps
        d = self.taps_f - frac
        valid = (k >= self.zero_i) & (k < n) & (torch.abs(self.taps_f) <= half)

        z = self.pi * d * cutoff
        sinc = torch.where(torch.abs(z) < self.tiny, self.one, torch.sin(z) * torch.reciprocal(z))
        hann = self.half_one + self.half_one * torch.cos(self.pi * d * inv_half1)
        tri = torch.clamp(self.one - torch.abs(d), min=0.0)
        w = torch.where(valid, torch.where(decimate, sinc * hann, tri), self.zero)

        samples = x[torch.clamp(k, min=0, max=n - 1)]
        acc = torch.sum(samples * w, dim=-1)
        norm = torch.sum(w, dim=-1)
        out = torch.where(norm == self.zero, self.zero, acc * torch.reciprocal(norm))
        return out.view(1, 1, -1)


def _sinc_hann_kernel(taps: np.ndarray, frac: np.float32, inv_ratio: np.float32, radius: np.float32) -> np.ndarray:
    inv_rp1 = np.float32(1.0) / (radius + np.float32(1.0))
    pi, tiny = np.float32(np.pi), np.float32(1e-12)
    dt = taps.astype(np.float32) - frac
    x = pi * dt * inv_ratio
    safe_x = np.where(np.abs(x) < tiny, np.float32(1.0), x)
    sinc = np.where(np.abs(x) < tiny, np.float32(1.0), np.sin(x) / safe_x)
    win = np.float32(0.5) + np.float32(0.5) * np.cos(pi * dt * inv_rp1)
    return np.where(np.abs(taps) <= radius, sinc * win, np.float32(0.0)).astype(np.float32)


def _make_cs_table(w_active: np.ndarray) -> np.ndarray:
    """Build [P, K_active + 1, 2] pure float32 (hi, lo) compensated prefix-sum table (zero DOUBLE)."""
    phases, k_act = w_active.shape
    cs64 = np.zeros((phases, k_act + 1), dtype=np.float64)
    cs64[:, 1:] = np.cumsum(w_active.astype(np.float64), axis=1, dtype=np.float64)
    hi = cs64.astype(np.float32)
    lo = (cs64 - hi.astype(np.float64)).astype(np.float32)
    return np.ascontiguousarray(np.stack([hi, lo], axis=-1), dtype=np.float32)


def _make_norm_weights_24k() -> tuple[np.ndarray, np.ndarray]:
    inv_out = np.float32(1.0 / OUTPUT_SAMPLE_RATE)
    ratio = np.float32(24000.0) * inv_out
    inv_ratio = np.float32(1.0) / ratio
    rad = np.float32(np.round(ratio * np.float32(8.0)))
    taps = np.arange(-12, 13, dtype=np.float32)
    w0 = _sinc_hann_kernel(taps, np.float32(0.0), inv_ratio, rad)
    w1 = _sinc_hann_kernel(taps, np.float32(8000.0) * inv_out, inv_ratio, rad)
    w0_n, w1_n = (w0 / np.sum(w0)).astype(np.float32), (w1 / np.sum(w1)).astype(np.float32)
    w_conv = np.zeros((2, 1, 26), dtype=np.float32)
    w_conv[0, 0, :25] = w0_n
    w_conv[1, 0, 1:] = w1_n
    return w_conv, np.stack([w0_n, w1_n], axis=0)


def _make_norm_weights_int_ratio(sample_rate: int) -> tuple[np.ndarray, int, int]:
    stride = sample_rate // OUTPUT_SAMPLE_RATE
    if stride <= 1 or stride * OUTPUT_SAMPLE_RATE != sample_rate:
        raise ValueError(f"{sample_rate} Hz is not an integer decimation ratio of {OUTPUT_SAMPLE_RATE} Hz")
    inv_out = np.float32(1.0 / OUTPUT_SAMPLE_RATE)
    inv_ratio = np.float32(1.0) / (np.float32(sample_rate) * inv_out)
    rad_i = int(round(8.0 * stride))
    w = _sinc_hann_kernel(np.arange(-rad_i, rad_i + 1, dtype=np.float32), np.float32(0.0), inv_ratio, np.float32(rad_i))
    return (w / np.sum(w)).astype(np.float32).reshape(1, 1, 2 * rad_i + 1), stride, rad_i


def _make_norm_weights_44k1() -> np.ndarray:
    inv_out = np.float32(1.0 / OUTPUT_SAMPLE_RATE)
    g = helper.make_graph(
        [
            helper.make_node("Mul", ["rem_f", "inv_out"], ["frac"]),
            helper.make_node("Sub", ["taps_f", "frac"], ["dt"]),
            helper.make_node("Mul", ["pi", "dt"], ["pi_dt"]),
            helper.make_node("Mul", ["pi_dt", "inv_ratio"], ["xs"]),
            helper.make_node("Mul", ["pi_dt", "inv_rp1"], ["xw"]),
            helper.make_node("Abs", ["xs"], ["abs_xs"]),
            helper.make_node("Less", ["abs_xs", "tiny"], ["is_tiny"]),
            helper.make_node("Sin", ["xs"], ["sin_xs"]),
            helper.make_node("Reciprocal", ["xs"], ["inv_xs"]),
            helper.make_node("Mul", ["sin_xs", "inv_xs"], ["sinc_raw"]),
            helper.make_node("Where", ["is_tiny", "one", "sinc_raw"], ["sinc"]),
            helper.make_node("Cos", ["xw"], ["cos_xw"]),
            helper.make_node("Mul", ["half", "cos_xw"], ["hcos"]),
            helper.make_node("Add", ["half", "hcos"], ["win"]),
            helper.make_node("Mul", ["sinc", "win"], ["w_raw45"]),
        ],
        "w44_eval", [],
        [helper.make_tensor_value_info("w_raw45", TensorProto.FLOAT, [160, 45])],
        [
            numpy_helper.from_array(((np.arange(160, dtype=np.int64) * 44100) % OUTPUT_SAMPLE_RATE).astype(np.float32).reshape(160, 1), name="rem_f"),
            numpy_helper.from_array(np.arange(-22, 23, dtype=np.float32).reshape(1, 45), name="taps_f"),
            numpy_helper.from_array(np.array(inv_out, dtype=np.float32), name="inv_out"),
            numpy_helper.from_array(np.array(1.0 / (np.float32(44100.0) * inv_out), dtype=np.float32), name="inv_ratio"),
            numpy_helper.from_array(np.array(1.0 / np.float32(23.0), dtype=np.float32), name="inv_rp1"),
            numpy_helper.from_array(np.array(np.pi, dtype=np.float32), name="pi"),
            numpy_helper.from_array(np.array(1e-12, dtype=np.float32), name="tiny"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one"),
            numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="half"),
        ],
    )
    m = helper.make_model(g, opset_imports=[onnx.OperatorSetIdProto(version=OPSET)])
    m.ir_version = IR_VERSION
    sess = onnxruntime.InferenceSession(m.SerializeToString(), _make_session_options(), providers=["CPUExecutionProvider"])
    w45 = sess.run(None, {})[0].astype(np.float32)
    return (w45 * (np.float32(1.0) / np.sum(w45, axis=1, keepdims=True))).astype(np.float32)


def _emit_grouped_mm_44k1_nodes(p: str, x_in: str, y_out: str, n_groups: int, tag: str) -> list[onnx.NodeProto]:
    pg = 160 // n_groups
    t = p + tag + "_"
    nodes = [
        helper.make_node("Slice", [x_in, "top_i0", p + "cn42", "top_i2"], [t + "xm"]),
        helper.make_node("Reshape", [t + "xm", p + "sh_blk"], [t + "Bm"]),
        helper.make_node("Slice", [x_in, p + "c42", "top_imax", "top_i2"], [t + "xh"]),
        helper.make_node("Reshape", [t + "xh", p + "sh_blk"], [t + "Bh"]),
    ]
    g_outs: list[str] = []
    for g in range(n_groups):
        ph1 = (g + 1) * pg
        c1 = ((ph1 - 1) * 441) // 160 + 45
        src = (t + "Bm") if c1 <= 441 else (t + "Bh")
        s0_name = "top_i0" if g == 0 else (p + f"s0_{g}")
        s1_val = c1 if c1 <= 441 else (c1 - 42)
        s1_name = (p + "c441") if s1_val == 441 else (p + f"s1_{g}")
        nodes.append(helper.make_node("Slice", [src, s0_name, s1_name, "top_i1"], [t + f"X_{g}"]))
        nodes.append(helper.make_node("MatMul", [t + f"X_{g}", p + f"W_{g}"], [t + f"Y_{g}"]))
        g_outs.append(t + f"Y_{g}")
    nodes.append(helper.make_node("Concat", g_outs, [y_out], axis=1))
    return nodes


def _emit_cs_norm(
    p: str,
    y_in: str,
    blk_indices: str,
    pad_l: str,
    val_end: str,
    y_out: str,
    tag: str,
    *,
    idx_shape_name: str,
    stride_name: str,
    k_act_name: str,
    phases: int,
) -> list[onnx.NodeProto]:
    """Unified pure-float32 (hi, lo) CS boundary normalization for both 44.1 kHz MatMul and Conv branches."""
    t = p + tag + "_"
    nodes = [
        helper.make_node("Reshape", [blk_indices, idx_shape_name], [t + "bidx"]),
        helper.make_node("Mul", [t + "bidx", stride_name], [t + "bbase"]),
    ]
    if phases > 1:
        nodes.append(helper.make_node("Add", [t + "bbase", p + "ph_off"], [t + "start"]))
        start_name = t + "start"
    else:
        start_name = t + "bbase"
    nodes.extend([
        helper.make_node("Sub", [pad_l, start_name], [t + "klo_raw"]),
        helper.make_node("Clip", [t + "klo_raw", "top_i0", k_act_name], [t + "klo"]),
        helper.make_node("Sub", [val_end, start_name], [t + "khi_raw"]),
        helper.make_node("Clip", [t + "khi_raw", "top_i0", k_act_name], [t + "khi"]),
    ])
    if phases > 1:
        nodes.append(helper.make_node("Add", [p + "cs_row_base", t + "klo"], [t + "glo_idx"]))
        nodes.append(helper.make_node("Add", [p + "cs_row_base", t + "khi"], [t + "ghi_idx"]))
        glo_name, ghi_name = t + "glo_idx", t + "ghi_idx"
    else:
        glo_name, ghi_name = t + "klo", t + "khi"
    nodes.extend([
        helper.make_node("Gather", [p + "cs_flat", glo_name], [t + "cs_lo"], axis=0),
        helper.make_node("Gather", [p + "cs_flat", ghi_name], [t + "cs_hi"], axis=0),
        helper.make_node("Sub", [t + "cs_hi", t + "cs_lo"], [t + "den2"]),
        helper.make_node("ReduceSum", [t + "den2", "top_neg1"], [t + "den"], keepdims=0),
        helper.make_node("Sub", [t + "khi", t + "klo"], [t + "span"]),
        helper.make_node("Equal", [t + "span", k_act_name], [t + "is_full"]),
        helper.make_node("Equal", [t + "den", "top_f0"], [t + "isz"]),
        helper.make_node("Or", [t + "isz", t + "is_full"], [t + "bypass"]),
        helper.make_node("Where", [t + "bypass", "top_f1", t + "den"], [t + "sden"]),
        helper.make_node("Div", [y_in, t + "sden"], [t + "div"]),
        helper.make_node("Where", [t + "isz", "top_f0", t + "div"], [y_out]),
    ])
    return nodes


def _emit_flat_edge_cs_norm(
    p: str,
    y_raw: str,
    n_blk: str,
    pad_l: str,
    pad_r: str,
    in_lo: str,
    in_hi: str,
    y_out: str,
    *,
    c_s: str,
    c_sm1: str,
    c_ka: str,
    phases: int,
    axis: int,
    idx_shape_name: str,
) -> list[onnx.NodeProto]:
    """Branch-free edge-split CS boundary normalization shared by 44.1 kHz and Conv branches."""
    ax_const = "top_i0" if axis == 0 else "top_i2"
    return [
        helper.make_node("Sub", [in_hi, in_lo], [p + "L_val"]),
        helper.make_node("Add", [pad_l, p + "L_val"], [p + "val_end"]),
        helper.make_node("Add", [pad_l, c_sm1], [p + "pl_u"]),
        helper.make_node("Div", [p + "pl_u", c_s], [p + "nl_raw"]),
        helper.make_node("Clip", [p + "nl_raw", "top_i0", n_blk], [p + "n_left"]),
        helper.make_node("Add", [pad_r, c_sm1], [p + "pr_u"]),
        helper.make_node("Div", [p + "pr_u", c_s], [p + "nr_raw"]),
        helper.make_node("Clip", [p + "nr_raw", "top_i0", n_blk], [p + "n_right"]),
        helper.make_node("Sub", [n_blk, p + "n_right"], [p + "me_raw"]),
        helper.make_node("Max", [p + "n_left", p + "me_raw"], [p + "mid_end"]),
        helper.make_node("Squeeze", [p + "n_left", "top_ax0"], [p + "nl_s"]),
        helper.make_node("Squeeze", [p + "mid_end", "top_ax0"], [p + "me_s"]),
        helper.make_node("Squeeze", [n_blk, "top_ax0"], [p + "nb_s"]),
        helper.make_node("Range", ["top_s0", p + "nl_s", "top_s1"], [p + "l_idx"]),
        helper.make_node("Range", [p + "me_s", p + "nb_s", "top_s1"], [p + "r_idx"]),
        helper.make_node("Concat", [p + "l_idx", p + "r_idx"], [p + "edge_idx"], axis=0),
        helper.make_node("Slice", [y_raw, "top_i0", p + "n_left", ax_const], [p + "y_l_in"]),
        helper.make_node("Slice", [y_raw, p + "n_left", p + "mid_end", ax_const], [p + "y_mid"]),
        helper.make_node("Slice", [y_raw, p + "mid_end", n_blk, ax_const], [p + "y_r_in"]),
        helper.make_node("Concat", [p + "y_l_in", p + "y_r_in"], [p + "y_edge_in"], axis=axis),
        *_emit_cs_norm(
            p, p + "y_edge_in", p + "edge_idx", pad_l, p + "val_end", p + "y_edge_out", "cse",
            idx_shape_name=idx_shape_name, stride_name=c_s, k_act_name=c_ka, phases=phases,
        ),
        helper.make_node("Slice", [p + "y_edge_out", "top_i0", p + "n_left", ax_const], [p + "y_l_out"]),
        helper.make_node("Slice", [p + "y_edge_out", p + "n_left", "top_imax", ax_const], [p + "y_r_out"]),
        helper.make_node("Concat", [p + "y_l_out", p + "y_mid", p + "y_r_out"], [y_out], axis=axis),
    ]


def _build_b44_polyphase_branch(n_groups: int = B44_NUM_GROUPS) -> onnx.GraphProto:
    if 160 % n_groups != 0:
        raise ValueError(f"160 phases must be divisible by n_groups={n_groups}")
    pg = 160 // n_groups
    w_norm45 = _make_norm_weights_44k1()
    cs_flat = _make_cs_table(w_norm45).reshape(-1, 2)
    phase_off = np.array([(ph * 441) // 160 for ph in range(160)], dtype=np.int32).reshape(1, 160)
    cs_row_base = (np.arange(160, dtype=np.int32) * 46).reshape(1, 160)
    p = "b44_"
    inits = [
        numpy_helper.from_array(np.array([22], dtype=np.int32), name=p + "rad"),
        numpy_helper.from_array(np.array([42], dtype=np.int32), name=p + "c42"),
        numpy_helper.from_array(np.array([-42], dtype=np.int32), name=p + "cn42"),
        numpy_helper.from_array(np.array([45], dtype=np.int32), name=p + "c45"),
        numpy_helper.from_array(np.array([159], dtype=np.int32), name=p + "c159"),
        numpy_helper.from_array(np.array([160], dtype=np.int32), name=p + "c160"),
        numpy_helper.from_array(np.array([440], dtype=np.int32), name=p + "c440"),
        numpy_helper.from_array(np.array([441], dtype=np.int32), name=p + "c441"),
        numpy_helper.from_array(np.array([461], dtype=np.int32), name=p + "c461"),
        numpy_helper.from_array(np.array([-1, 441], dtype=np.int64), name=p + "sh_blk"),
        numpy_helper.from_array(phase_off, name=p + "ph_off"),
        numpy_helper.from_array(cs_row_base, name=p + "cs_row_base"),
        numpy_helper.from_array(cs_flat, name=p + "cs_flat"),
    ]
    for g in range(n_groups):
        ph0, ph1 = g * pg, (g + 1) * pg
        c0, c1 = (ph0 * 441) // 160, ((ph1 - 1) * 441) // 160 + 45
        w_g = np.zeros((c1 - c0, pg), dtype=np.float32)
        for ph in range(ph0, ph1):
            off = (ph * 441) // 160 - c0
            w_g[off:off + 45, ph - ph0] = w_norm45[ph]
        s0, s1 = (c0, c1) if c1 <= 441 else (c0 - 42, c1 - 42)
        if g > 0:
            inits.append(numpy_helper.from_array(np.array([s0], dtype=np.int32), name=p + f"s0_{g}"))
        if s1 != 441:
            inits.append(numpy_helper.from_array(np.array([s1], dtype=np.int32), name=p + f"s1_{g}"))
        inits.append(numpy_helper.from_array(w_g, name=p + f"W_{g}"))

    nodes = [
        helper.make_node("Mod", ["top_st", p + "c160"], [p + "par_start"], fmod=0),
        helper.make_node("Sub", ["top_st", p + "par_start"], [p + "q_base"]),
        helper.make_node("Div", [p + "q_base", p + "c160"], [p + "q_start"]),
        helper.make_node("Add", [p + "par_start", "top_cnt"], [p + "par_end"]),
        helper.make_node("Add", [p + "par_end", p + "c159"], [p + "pe_u"]),
        helper.make_node("Div", [p + "pe_u", p + "c160"], [p + "n_blk"]),
        helper.make_node("Sub", [p + "n_blk", "top_i1"], [p + "qb_m1"]),
        helper.make_node("Add", [p + "q_start", p + "qb_m1"], [p + "q_end"]),
        helper.make_node("Mul", [p + "q_start", p + "c441"], [p + "q_lo"]),
        helper.make_node("Sub", [p + "q_lo", p + "rad"], [p + "conv_lo"]),
        helper.make_node("Mul", [p + "q_end", p + "c441"], [p + "q_hi"]),
        helper.make_node("Add", [p + "q_hi", p + "c461"], [p + "conv_hi"]),
        helper.make_node("Clip", [p + "conv_lo", "top_i0", "top_N"], [p + "c_in_lo"]),
        helper.make_node("Clip", [p + "conv_hi", p + "c_in_lo", "top_N"], [p + "c_in_hi"]),
        helper.make_node("Sub", [p + "c_in_lo", p + "conv_lo"], [p + "c_pad_l"]),
        helper.make_node("Sub", [p + "conv_hi", p + "c_in_hi"], [p + "c_pad_r"]),
        helper.make_node("Slice", ["audio", p + "c_in_lo", p + "c_in_hi", "top_i2"], [p + "c_sl"]),
        helper.make_node("Clip", [p + "c_sl", "top_c_min", "top_f1"], [p + "clipped"]),
        helper.make_node("Add", [p + "c_pad_l", p + "c_pad_r"], [p + "pad_sum"]),
        helper.make_node("Equal", [p + "pad_sum", "top_i0"], [p + "no_pad_v"]),
        helper.make_node("Squeeze", [p + "no_pad_v", "top_ax0"], [p + "no_pad"]),
    ]
    fast_g = helper.make_graph(
        _emit_grouped_mm_44k1_nodes(p, p + "clipped", p + "yf", n_groups, "f"),
        p + "fast", [], [helper.make_tensor_value_info(p + "yf", TensorProto.FLOAT, ["n_blk", 160])],
    )
    pad_g = helper.make_graph(
        [
            helper.make_node("Concat", ["top_z2", p + "c_pad_l", "top_z2", p + "c_pad_r"], [p + "pads_i32"], axis=0),
            helper.make_node("Cast", [p + "pads_i32"], [p + "pads_vec"], to=TensorProto.INT64),
            helper.make_node("Pad", [p + "clipped", p + "pads_vec", "top_f0"], [p + "px"]),
            *_emit_grouped_mm_44k1_nodes(p, p + "px", p + "yp_raw", n_groups, "p"),
            *_emit_flat_edge_cs_norm(
                p, p + "yp_raw", p + "n_blk", p + "c_pad_l", p + "c_pad_r", p + "c_in_lo", p + "c_in_hi", p + "yp",
                c_s=p + "c441", c_sm1=p + "c440", c_ka=p + "c45", phases=160, axis=0, idx_shape_name="top_col_sh",
            ),
        ],
        p + "pad", [], [helper.make_tensor_value_info(p + "yp", TensorProto.FLOAT, ["n_blk", 160])],
    )
    nodes.extend([
        helper.make_node("If", [p + "no_pad"], [p + "y_blk"], then_branch=fast_g, else_branch=pad_g),
        helper.make_node("Reshape", [p + "y_blk", "top_out_sh"], [p + "flat"]),
        helper.make_node("Slice", [p + "flat", p + "par_start", p + "par_end", "top_i2"], [p + "ref"]),
    ])
    return helper.make_graph(nodes, p + "branch", [], [helper.make_tensor_value_info(p + "ref", TensorProto.FLOAT, [1, 1, "out_count"])], inits)


def _build_conv_polyphase_branch(
    p: str,
    stride: int,
    taps_half: int,
    k_len: int,
    w_norm: np.ndarray,
    w_active: np.ndarray,
    phase_offsets: list[int],
) -> onnx.GraphProto:
    phases, k_act = w_active.shape
    cs_flat = _make_cs_table(w_active).reshape(-1, 2)
    top_small = {0: "top_i0", 1: "top_i1", 2: "top_i2", 3: "top_i3", 25: "top_i25"}
    inits: list[onnx.TensorProto] = []

    def _i32_const(val: int, name: str) -> str:
        if val in top_small:
            return top_small[val]
        inits.append(numpy_helper.from_array(np.array([val], dtype=np.int32), name=name))
        return name

    c_s = _i32_const(stride, p + "c_s")
    c_sm1 = _i32_const(stride - 1, p + "c_sm1")
    c_ka = _i32_const(k_act, p + "c_ka")
    c_re = _i32_const(k_len - taps_half, p + "right_extra")
    inits.extend([
        numpy_helper.from_array(np.array([taps_half], dtype=np.int32), name=p + "rad"),
        numpy_helper.from_array(cs_flat, name=p + "cs_flat"),
        numpy_helper.from_array(w_norm, name=p + "w_norm"),
    ])
    nodes: list[onnx.NodeProto] = []
    if phases > 1:
        inits.append(numpy_helper.from_array(np.array(phase_offsets, dtype=np.int32).reshape(1, phases, 1), name=p + "ph_off"))
        inits.append(numpy_helper.from_array((np.arange(phases, dtype=np.int32) * (k_act + 1)).reshape(1, phases, 1), name=p + "cs_row_base"))
        c_p = _i32_const(phases, p + "c_p")
        c_pm1 = _i32_const(phases - 1, p + "c_pm1")
        n_blk_name = p + "n_blk"
        nodes.extend([
            helper.make_node("Mod", ["top_st", c_p], [p + "par_start"], fmod=0),
            helper.make_node("Sub", ["top_st", p + "par_start"], [p + "q_base"]),
            helper.make_node("Div", [p + "q_base", c_p], [p + "q_start"]),
            helper.make_node("Add", [p + "par_start", "top_cnt"], [p + "par_end"]),
            helper.make_node("Add", [p + "par_end", c_pm1], [p + "pe_u"]),
            helper.make_node("Div", [p + "pe_u", c_p], [n_blk_name]),
            helper.make_node("Sub", [n_blk_name, "top_i1"], [p + "qb_m1"]),
            helper.make_node("Add", [p + "q_start", p + "qb_m1"], [p + "q_end"]),
            helper.make_node("Mul", [p + "q_start", c_s], [p + "c_lo"]),
            helper.make_node("Mul", [p + "q_end", c_s], [p + "c_hi"]),
        ])
    else:
        n_blk_name = "top_cnt"
        nodes.extend([
            helper.make_node("Sub", ["top_cnt", "top_i1"], [p + "cnt_m1"]),
            helper.make_node("Add", ["top_st", p + "cnt_m1"], [p + "end_m1"]),
            helper.make_node("Mul", ["top_st", c_s], [p + "c_lo"]),
            helper.make_node("Mul", [p + "end_m1", c_s], [p + "c_hi"]),
        ])

    nodes.extend([
        helper.make_node("Sub", [p + "c_lo", p + "rad"], [p + "req_lo"]),
        helper.make_node("Add", [p + "c_hi", c_re], [p + "req_hi"]),
        helper.make_node("Clip", [p + "req_lo", "top_i0", "top_N"], [p + "in_lo"]),
        helper.make_node("Clip", [p + "req_hi", p + "in_lo", "top_N"], [p + "in_hi"]),
        helper.make_node("Sub", [p + "in_lo", p + "req_lo"], [p + "pad_l"]),
        helper.make_node("Sub", [p + "req_hi", p + "in_hi"], [p + "pad_r"]),
        helper.make_node("Slice", ["audio", p + "in_lo", p + "in_hi", "top_i2"], [p + "sliced"]),
        helper.make_node("Clip", [p + "sliced", "top_c_min", "top_f1"], [p + "clipped"]),
        helper.make_node("Add", [p + "pad_l", p + "pad_r"], [p + "pad_sum"]),
        helper.make_node("Equal", [p + "pad_sum", "top_i0"], [p + "no_pad_v"]),
        helper.make_node("Squeeze", [p + "no_pad_v", "top_ax0"], [p + "no_pad"]),
    ])
    fast_g = helper.make_graph(
        [helper.make_node("Conv", [p + "clipped", p + "w_norm"], [p + "conv_f"], strides=[stride])],
        p + "fast", [], [helper.make_tensor_value_info(p + "conv_f", TensorProto.FLOAT, [1, phases, "n_blk"])],
    )
    pad_g = helper.make_graph(
        [
            helper.make_node("Concat", ["top_z2", p + "pad_l", "top_z2", p + "pad_r"], [p + "pads_i32"], axis=0),
            helper.make_node("Cast", [p + "pads_i32"], [p + "pads_vec"], to=TensorProto.INT64),
            helper.make_node("Pad", [p + "clipped", p + "pads_vec", "top_f0"], [p + "px"]),
            helper.make_node("Conv", [p + "px", p + "w_norm"], [p + "conv_p"], strides=[stride]),
            *_emit_flat_edge_cs_norm(
                p, p + "conv_p", n_blk_name, p + "pad_l", p + "pad_r", p + "in_lo", p + "in_hi", p + "yp",
                c_s=c_s, c_sm1=c_sm1, c_ka=c_ka, phases=phases, axis=2, idx_shape_name="top_out_sh",
            ),
        ],
        p + "padded", [], [helper.make_tensor_value_info(p + "yp", TensorProto.FLOAT, [1, phases, "n_blk"])],
    )
    if phases > 1:
        nodes.extend([
            helper.make_node("If", [p + "no_pad"], [p + "y_conv"], then_branch=fast_g, else_branch=pad_g),
            helper.make_node("Transpose", [p + "y_conv"], [p + "tr"], perm=[0, 2, 1]),
            helper.make_node("Reshape", [p + "tr", "top_out_sh"], [p + "flat"]),
            helper.make_node("Slice", [p + "flat", p + "par_start", p + "par_end", "top_i2"], [p + "ref"]),
        ])
    else:
        nodes.append(helper.make_node("If", [p + "no_pad"], [p + "ref"], then_branch=fast_g, else_branch=pad_g))
    return helper.make_graph(nodes, p + "branch", [], [helper.make_tensor_value_info(p + "ref", TensorProto.FLOAT, [1, 1, "out_count"])], inits)


def _build_b16_branch() -> onnx.GraphProto:
    p = "b16_"
    nodes = [
        helper.make_node("Add", ["top_st", "top_cnt"], [p + "end"]),
        helper.make_node("Clip", ["top_st", "top_i0", "top_N"], [p + "lo"]),
        helper.make_node("Clip", [p + "end", p + "lo", "top_N"], [p + "hi"]),
        helper.make_node("Sub", [p + "lo", "top_st"], [p + "pad_l"]),
        helper.make_node("Sub", [p + "end", p + "hi"], [p + "pad_r"]),
        helper.make_node("Slice", ["audio", p + "lo", p + "hi", "top_i2"], [p + "sl"]),
        helper.make_node("Clip", [p + "sl", "top_c_min", "top_f1"], [p + "cl"]),
        helper.make_node("Concat", ["top_z2", p + "pad_l", "top_z2", p + "pad_r"], [p + "pads_i32"], axis=0),
        helper.make_node("Cast", [p + "pads_i32"], [p + "pads"], to=TensorProto.INT64),
        helper.make_node("Pad", [p + "cl", p + "pads", "top_f0"], [p + "ref"]),
    ]
    return helper.make_graph(nodes, p + "branch", [], [helper.make_tensor_value_info(p + "ref", TensorProto.FLOAT, [1, 1, "out_count"])])


def _build_windowed_fallback_branch() -> onnx.GraphProto:
    p = "fbg_"
    inits = [
        numpy_helper.from_array(np.array([1, -1], dtype=np.int64), name=p + "row_sh"),
        numpy_helper.from_array(np.array(1.0 / OUTPUT_SAMPLE_RATE, dtype=np.float32), name=p + "inv_out_rate"),
        numpy_helper.from_array(np.array(8.0, dtype=np.float32), name=p + "f8"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), name=p + "fhalf"),
        numpy_helper.from_array(np.array(np.pi, dtype=np.float32), name=p + "pi"),
        numpy_helper.from_array(np.array(1e-12, dtype=np.float32), name=p + "tiny"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), name=p + "ozero1"),
    ]
    down_g = helper.make_graph(
        [
            helper.make_node("Reciprocal", [p + "ratio"], [p + "inv_ratio"]),
            helper.make_node("Add", [p + "rad_f", "top_f1"], [p + "rad_p1_f"]),
            helper.make_node("Reciprocal", [p + "rad_p1_f"], [p + "inv_rad_p1"]),
            helper.make_node("Mul", [p + "pi", p + "dt"], [p + "pi_dt"]),
            helper.make_node("Mul", [p + "pi_dt", p + "inv_ratio"], [p + "x_sinc"]),
            helper.make_node("Mul", [p + "pi_dt", p + "inv_rad_p1"], [p + "x_win"]),
            helper.make_node("Abs", [p + "x_sinc"], [p + "abs_xs"]),
            helper.make_node("Less", [p + "abs_xs", p + "tiny"], [p + "is_tiny"]),
            helper.make_node("Sin", [p + "x_sinc"], [p + "sin_xs"]),
            helper.make_node("Reciprocal", [p + "x_sinc"], [p + "inv_xs"]),
            helper.make_node("Mul", [p + "sin_xs", p + "inv_xs"], [p + "sinc_raw"]),
            helper.make_node("Where", [p + "is_tiny", "top_f1", p + "sinc_raw"], [p + "sinc"]),
            helper.make_node("Cos", [p + "x_win"], [p + "cos_xw"]),
            helper.make_node("Mul", [p + "fhalf", p + "cos_xw"], [p + "hcos"]),
            helper.make_node("Add", [p + "fhalf", p + "hcos"], [p + "win"]),
            helper.make_node("Mul", [p + "sinc", p + "win"], [p + "k_down"]),
        ],
        p + "down", [], [helper.make_tensor_value_info(p + "k_down", TensorProto.FLOAT, ["out_count", "K"])],
    )
    up_g = helper.make_graph(
        [
            helper.make_node("Abs", [p + "dt"], [p + "abs_dt"]),
            helper.make_node("Sub", ["top_f1", p + "abs_dt"], [p + "lin_raw"]),
            helper.make_node("Clip", [p + "lin_raw", "top_f0"], [p + "k_up"]),
        ],
        p + "up", [], [helper.make_tensor_value_info(p + "k_up", TensorProto.FLOAT, ["out_count", "K"])],
    )
    nodes = [
        helper.make_node("Cast", ["top_sr"], [p + "sr_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", [p + "sr_f", p + "inv_out_rate"], [p + "ratio"]),
        helper.make_node("Greater", [p + "ratio", "top_f1"], [p + "gt1"]),
        helper.make_node("Squeeze", [p + "gt1", "top_ax0"], [p + "gt1_s"]),
        helper.make_node("Mul", [p + "ratio", p + "f8"], [p + "r8"]),
        helper.make_node("Round", [p + "r8"], [p + "r_round"]),
        helper.make_node("Where", [p + "gt1", p + "r_round", "top_f1"], [p + "rad_f"]),
        helper.make_node("Cast", [p + "rad_f"], [p + "rad_i"], to=TensorProto.INT32),
        helper.make_node("Neg", [p + "rad_i"], [p + "neg_rad_i"]),
        helper.make_node("Add", [p + "rad_i", "top_i1"], [p + "rad_p1_i"]),
        helper.make_node("Squeeze", [p + "neg_rad_i", "top_ax0"], [p + "t_lo_s"]),
        helper.make_node("Squeeze", [p + "rad_p1_i", "top_ax0"], [p + "t_hi_s"]),
        helper.make_node("Range", [p + "t_lo_s", p + "t_hi_s", "top_s1"], [p + "taps_1d"]),
        helper.make_node("Reshape", [p + "taps_1d", p + "row_sh"], [p + "taps"]),
        helper.make_node("Cast", [p + "taps"], [p + "taps_f"], to=TensorProto.FLOAT),
        helper.make_node("Squeeze", ["top_st", "top_ax0"], [p + "st_s"]),
        helper.make_node("Add", ["top_st", "top_cnt"], [p + "end_v"]),
        helper.make_node("Squeeze", [p + "end_v", "top_ax0"], [p + "end_s"]),
        helper.make_node("Range", [p + "st_s", p + "end_s", "top_s1"], [p + "n_vec"]),
        helper.make_node("Div", [p + "n_vec", "top_16k"], [p + "nq"]),
        helper.make_node("Mul", [p + "nq", "top_16k"], [p + "nq16"]),
        helper.make_node("Sub", [p + "n_vec", p + "nq16"], [p + "nr"]),
        helper.make_node("Mul", [p + "nr", "top_sr"], [p + "nr_sr"]),
        helper.make_node("Div", [p + "nr_sr", "top_16k"], [p + "nq_rem"]),
        helper.make_node("Mul", [p + "nq", "top_sr"], [p + "nq_sr"]),
        helper.make_node("Add", [p + "nq_sr", p + "nq_rem"], [p + "centers_1d"]),
        helper.make_node("Reshape", [p + "centers_1d", "top_col_sh"], [p + "centers"]),
        helper.make_node("Mul", [p + "nq_rem", "top_16k"], [p + "nq_rem16"]),
        helper.make_node("Sub", [p + "nr_sr", p + "nq_rem16"], [p + "rem_1d"]),
        helper.make_node("Reshape", [p + "rem_1d", "top_col_sh"], [p + "rem_i"]),
        helper.make_node("Cast", [p + "rem_i"], [p + "rem_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", [p + "rem_f", p + "inv_out_rate"], [p + "frac"]),
        helper.make_node("Slice", [p + "centers_1d", "top_i0", "top_i1", "top_i0"], [p + "c_lo"]),
        helper.make_node("Sub", ["top_cnt", "top_i1"], [p + "cnt_m1"]),
        helper.make_node("Slice", [p + "centers_1d", p + "cnt_m1", "top_cnt", "top_i0"], [p + "c_hi"]),
        helper.make_node("Sub", [p + "c_lo", p + "rad_i"], [p + "req_lo"]),
        helper.make_node("Add", [p + "c_hi", p + "rad_p1_i"], [p + "req_hi"]),
        helper.make_node("Clip", [p + "req_lo", "top_i0", "top_N"], [p + "in_lo"]),
        helper.make_node("Clip", [p + "req_hi", p + "in_lo", "top_N"], [p + "in_hi"]),
        helper.make_node("Slice", ["audio", p + "in_lo", p + "in_hi", "top_i2"], [p + "sliced3"]),
        helper.make_node("Reshape", [p + "sliced3", "top_neg1"], [p + "sliced1"]),
        helper.make_node("Clip", [p + "sliced1", "top_c_min", "top_f1"], [p + "clipped"]),
        helper.make_node("Concat", [p + "clipped", p + "ozero1"], [p + "safe_clip"], axis=0),
        helper.make_node("Sub", [p + "in_hi", p + "in_lo"], [p + "clen_v"]),
        helper.make_node("Sub", [p + "clen_v", "top_i1"], [p + "clen_m1"]),
        helper.make_node("Max", [p + "clen_m1", "top_i0"], [p + "clen_hi"]),
        helper.make_node("Add", [p + "centers", p + "taps"], [p + "abs_idx"]),
        helper.make_node("GreaterOrEqual", [p + "abs_idx", "top_i0"], [p + "ge0"]),
        helper.make_node("Less", [p + "abs_idx", "top_N"], [p + "ltN"]),
        helper.make_node("And", [p + "ge0", p + "ltN"], [p + "valid_m"]),
        helper.make_node("Sub", [p + "abs_idx", p + "in_lo"], [p + "rel_idx"]),
        helper.make_node("Clip", [p + "rel_idx", "top_i0", p + "clen_hi"], [p + "safe_idx"]),
        helper.make_node("Gather", [p + "safe_clip", p + "safe_idx"], [p + "xg"], axis=0),
        helper.make_node("Sub", [p + "taps_f", p + "frac"], [p + "dt"]),
        helper.make_node("If", [p + "gt1_s"], [p + "k_kern"], then_branch=down_g, else_branch=up_g),
        helper.make_node("Where", [p + "valid_m", p + "k_kern", "top_f0"], [p + "wm"]),
        helper.make_node("Mul", [p + "xg", p + "wm"], [p + "prod_w"]),
        helper.make_node("ReduceSum", [p + "prod_w", "top_ax1"], [p + "num"], keepdims=0),
        helper.make_node("ReduceSum", [p + "wm", "top_ax1"], [p + "den"], keepdims=0),
        helper.make_node("Equal", [p + "den", "top_f0"], [p + "isz"]),
        helper.make_node("Reciprocal", [p + "den"], [p + "inv_den"]),
        helper.make_node("Mul", [p + "num", p + "inv_den"], [p + "div_out"]),
        helper.make_node("Where", [p + "isz", "top_f0", p + "div_out"], [p + "clean"]),
        helper.make_node("Reshape", [p + "clean", "top_out_sh"], ["fb_ref"]),
    ]
    return helper.make_graph(nodes, "fallback_general", [], [helper.make_tensor_value_info("fb_ref", TensorProto.FLOAT, [1, 1, "out_count"])], inits)


def export_multirate_stage0_in_memory() -> onnx.ModelProto:
    args = (
        torch.zeros(1, 1, 24000, dtype=torch.float32),
        torch.tensor([24000], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([4096], dtype=torch.int64),
    )
    model = _torch_export_to_onnx(
        MultiRateStage0(),
        args,
        input_names=["audio", "sample_rate", "out_start", "out_count"],
        output_names=["reference"],
        dynamic_axes={"audio": {2: "audio_len"}, "reference": {2: "out_len"}},
        downcast_slice_i32=False,
    )
    onnx.checker.check_model(model)
    return model


def _default_multirate_io_value_infos() -> tuple[list[onnx.ValueInfoProto], list[onnx.ValueInfoProto]]:
    inputs = [
        helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "audio_len"]),
        helper.make_tensor_value_info("sample_rate", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("out_start", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("out_count", TensorProto.INT64, [1]),
    ]
    return inputs, [helper.make_tensor_value_info("reference", TensorProto.FLOAT, [1, 1, "out_len"])]


def _validate_multirate_io(model: onnx.ModelProto) -> None:
    expected_in = [("audio", TensorProto.FLOAT), ("sample_rate", TensorProto.INT64), ("out_start", TensorProto.INT64), ("out_count", TensorProto.INT64)]
    expected_out = [("reference", TensorProto.FLOAT)]
    if [(v.name, v.type.tensor_type.elem_type) for v in model.graph.input] != expected_in:
        raise ValueError("Unexpected multirate input contract")
    if [(v.name, v.type.tensor_type.elem_type) for v in model.graph.output] != expected_out:
        raise ValueError("Unexpected multirate output contract")


def _collect_subgraphs(graph: onnx.GraphProto) -> dict[str, onnx.GraphProto]:
    found: dict[str, onnx.GraphProto] = {graph.name: graph}
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                found.update(_collect_subgraphs(attr.g))
    return found


def _is_current_multirate_v2(model: onnx.ModelProto) -> bool:
    props = {e.key: e.value for e in model.metadata_props}
    if props.get("tts_reference_polyphase_conv") != "1" or props.get("tts_reference_polyphase_v2") != "1":
        return False
    if not any(n.op_type == "Cast" and "top_N" in n.output for n in model.graph.node):
        return False
    subs = _collect_subgraphs(model.graph)
    if not {"b24_branch", "b48_branch", "b16_branch", "b44_branch", "b32_branch", "fallback_general"}.issubset(subs):
        return False
    if {"b24_full_cs", "b44_full_cs"}.intersection(subs) or any(g.value_info for g in subs.values()):
        return False
    b44_inits = {i.name: (i.data_type, list(i.dims)) for i in subs["b44_branch"].initializer}
    return b44_inits.get("b44_cs_flat") == (TensorProto.FLOAT, [160 * 46, 2]) and b44_inits.get("b44_ph_off") == (TensorProto.INT32, [1, 160])


def _validate_multirate_source(source: onnx.ModelProto) -> None:
    onnx.checker.check_model(source)
    _validate_multirate_io(source)
    props = {e.key: e.value for e in source.metadata_props}
    top_ops = {n.op_type for n in source.graph.node}
    if props.get("tts_reference_polyphase_conv") == "1":
        subs = _collect_subgraphs(source.graph)
        if "If" in top_ops and {"b24_branch", "b48_branch", "b16_branch", "b44_branch", "fallback_general"}.issubset(subs):
            return
        raise ValueError("Source claims tts_reference_polyphase_conv=1 but is missing required branch subgraphs")
    req_ops = {"Sin", "Cos", "Gather", "ReduceSum", "Where", "Clip"}
    if req_ops.issubset(top_ops):
        return
    raise ValueError(f"Unrecognized source multirate graph (missing Stage-0 ops {sorted(req_ops - top_ops)})")


def _build_multirate_polyphase_model(
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    extra_props: dict[str, str] | None = None,
) -> onnx.ModelProto:
    w24_conv, w24_act = _make_norm_weights_24k()
    w48_conv, s48, r48 = _make_norm_weights_int_ratio(48000)
    w32_conv, s32, r32 = _make_norm_weights_int_ratio(32000)
    b24_g = _build_conv_polyphase_branch("b24_", 3, 12, 26, w24_conv, w24_act, [0, 1])
    b48_g = _build_conv_polyphase_branch("b48_", s48, r48, 2 * r48 + 1, w48_conv, w48_conv.reshape(1, -1), [0])
    b32_g = _build_conv_polyphase_branch("b32_", s32, r32, 2 * r32 + 1, w32_conv, w32_conv.reshape(1, -1), [0])
    b16_g, b44_g, fb_g = _build_b16_branch(), _build_b44_polyphase_branch(), _build_windowed_fallback_branch()

    top_inits = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="top_ax0"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="top_ax1"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="top_neg1"),
        numpy_helper.from_array(np.array([-1, 1], dtype=np.int64), name="top_col_sh"),
        numpy_helper.from_array(np.array([1, 1, -1], dtype=np.int64), name="top_out_sh"),
        numpy_helper.from_array(np.array([0], dtype=np.int32), name="top_i0"),
        numpy_helper.from_array(np.array([1], dtype=np.int32), name="top_i1"),
        numpy_helper.from_array(np.array([2], dtype=np.int32), name="top_i2"),
        numpy_helper.from_array(np.array([3], dtype=np.int32), name="top_i3"),
        numpy_helper.from_array(np.array([25], dtype=np.int32), name="top_i25"),
        numpy_helper.from_array(np.array([np.iinfo(np.int32).max], dtype=np.int32), name="top_imax"),
        numpy_helper.from_array(np.array(0, dtype=np.int32), name="top_s0"),
        numpy_helper.from_array(np.array(1, dtype=np.int32), name="top_s1"),
        numpy_helper.from_array(np.array([0, 0], dtype=np.int32), name="top_z2"),
        numpy_helper.from_array(np.array([16000], dtype=np.int32), name="top_16k"),
        numpy_helper.from_array(np.array([24000], dtype=np.int32), name="top_24k"),
        numpy_helper.from_array(np.array([32000], dtype=np.int32), name="top_32k"),
        numpy_helper.from_array(np.array([44100], dtype=np.int32), name="top_44k1"),
        numpy_helper.from_array(np.array([48000], dtype=np.int32), name="top_48k"),
        numpy_helper.from_array(np.array(-1.0, dtype=np.float32), name="top_c_min"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="top_f0"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="top_f1"),
    ]
    g_else_32 = helper.make_graph([helper.make_node("Equal", ["top_sr", "top_32k"], ["v_32k"]), helper.make_node("Squeeze", ["v_32k", "top_ax0"], ["is_32k"]), helper.make_node("If", ["is_32k"], ["r_32_fb"], then_branch=b32_g, else_branch=fb_g)], "else_32_fb", [], [helper.make_tensor_value_info("r_32_fb", TensorProto.FLOAT, [1, 1, "out_count"])])
    g_else_44 = helper.make_graph([helper.make_node("Equal", ["top_sr", "top_44k1"], ["v_44k1"]), helper.make_node("Squeeze", ["v_44k1", "top_ax0"], ["is_44k1"]), helper.make_node("If", ["is_44k1"], ["r_44_fb"], then_branch=b44_g, else_branch=g_else_32)], "else_44_fb", [], [helper.make_tensor_value_info("r_44_fb", TensorProto.FLOAT, [1, 1, "out_count"])])
    g_else_16 = helper.make_graph([helper.make_node("Equal", ["top_sr", "top_16k"], ["v_16k"]), helper.make_node("Squeeze", ["v_16k", "top_ax0"], ["is_16k"]), helper.make_node("If", ["is_16k"], ["r_16_fb"], then_branch=b16_g, else_branch=g_else_44)], "else_16_fb", [], [helper.make_tensor_value_info("r_16_fb", TensorProto.FLOAT, [1, 1, "out_count"])])
    g_else_48 = helper.make_graph([helper.make_node("Equal", ["top_sr", "top_48k"], ["v_48k"]), helper.make_node("Squeeze", ["v_48k", "top_ax0"], ["is_48k"]), helper.make_node("If", ["is_48k"], ["r_48_other"], then_branch=b48_g, else_branch=g_else_16)], "else_48_other", [], [helper.make_tensor_value_info("r_48_other", TensorProto.FLOAT, [1, 1, "out_count"])])
    top_nodes = [
        helper.make_node("Shape", ["audio"], ["top_ashape"]),
        helper.make_node("Slice", ["top_ashape", "top_i2", "top_i3", "top_i0"], ["top_N64"]),
        helper.make_node("Cast", ["top_N64"], ["top_N"], to=TensorProto.INT32),
        helper.make_node("Cast", ["sample_rate"], ["top_sr"], to=TensorProto.INT32),
        helper.make_node("Cast", ["out_start"], ["top_st"], to=TensorProto.INT32),
        helper.make_node("Cast", ["out_count"], ["top_cnt"], to=TensorProto.INT32),
        helper.make_node("Equal", ["top_sr", "top_24k"], ["v_24k"]),
        helper.make_node("Squeeze", ["v_24k", "top_ax0"], ["is_24k"]),
        helper.make_node("If", ["is_24k"], ["reference"], then_branch=b24_g, else_branch=g_else_48),
    ]
    out_m = onnx.ModelProto()
    out_m.ir_version = IR_VERSION
    op = out_m.opset_import.add()
    op.version = OPSET
    merged_props = {k: v for k, v in (extra_props or {}).items() if k not in ("tts_reference_polyphase_conv", "tts_reference_polyphase_v2")}
    merged_props["tts_reference_polyphase_conv"] = "1"
    merged_props["tts_reference_polyphase_v2"] = "1"
    helper.set_model_props(out_m, merged_props)
    out_m.graph.CopyFrom(helper.make_graph(top_nodes, "tts_reference_polyphase_conv", inputs, outputs, top_inits))
    out_m = _slim_final_model(out_m, downcast_slice_i32=False)
    _assert_structural_hygiene(
        out_m, "multirate_resampler",
        required_props=(("tts_reference_polyphase_conv", "1"), ("tts_reference_polyphase_v2", "1")),
    )
    return out_m


def optimize_multirate_model(source: onnx.ModelProto) -> onnx.ModelProto:
    _validate_multirate_source(source)
    if _is_current_multirate_v2(source):
        _assert_structural_hygiene(
            source, "multirate_resampler",
            required_props=(("tts_reference_polyphase_conv", "1"), ("tts_reference_polyphase_v2", "1")),
        )
        return copy.deepcopy(source)
    props = {e.key: e.value for e in source.metadata_props}
    return _build_multirate_polyphase_model(list(source.graph.input), list(source.graph.output), props)


def export_multirate_to_onnx(
    output_path: Path | None = None,
    *,
    polyphase: bool = MULTIRATE_POLYPHASE,
    source_onnx_path: Path | None = None,
    need_baseline: bool = True,
) -> tuple[Path, onnx.ModelProto | None]:
    save_path = output_path if output_path is not None else (_SCRIPT_DIR / MULTIRATE_MODEL_NAME)
    if not polyphase:
        stage0 = export_multirate_stage0_in_memory()
        _atomic_save_onnx(stage0, save_path)
        print(f"Legacy Stage-0 multi-rate resampler exported to: {save_path} ({save_path.stat().st_size} bytes)")
        return save_path, None
    baseline: onnx.ModelProto | None
    if source_onnx_path is not None:
        baseline = onnx.load(str(source_onnx_path), load_external_data=False)
        optimized = optimize_multirate_model(baseline)
    elif need_baseline:
        baseline = export_multirate_stage0_in_memory()
        optimized = optimize_multirate_model(baseline)
    else:
        baseline = None
        inputs, outputs = _default_multirate_io_value_infos()
        optimized = _build_multirate_polyphase_model(inputs, outputs)
    _atomic_save_onnx(optimized, save_path)
    print(f"Optimized multi-rate polyphase resampler exported to: {save_path} ({save_path.stat().st_size} bytes)")
    return save_path, baseline


def reference_multirate_resample(x: np.ndarray, rate: int) -> np.ndarray:
    """Float64 analytical reference matching the former Kotlin TtsPlayer arithmetic."""
    if rate == OUTPUT_SAMPLE_RATE:
        return x.copy()
    n = max(1, x.size * OUTPUT_SAMPLE_RATE // rate)
    ratio = rate / OUTPUT_SAMPLE_RATE
    out = np.zeros(n, dtype=np.float32)
    if rate > OUTPUT_SAMPLE_RATE:
        half = int(math.floor(8 * ratio + 0.5))
        cutoff = 1.0 / ratio
        for i in range(n):
            center = i * ratio
            c = int(center)
            acc = norm = 0.0
            for k in range(c - half, c + half + 1):
                if k < 0 or k >= x.size:
                    continue
                d = k - center
                sinc = 1.0 if d == 0.0 else math.sin(math.pi * d * cutoff) / (math.pi * d * cutoff)
                w = 0.5 + 0.5 * math.cos(math.pi * d / (half + 1))
                acc += float(x[k]) * sinc * w
                norm += sinc * w
            out[i] = 0.0 if norm == 0.0 else acc / norm
        return out
    for i in range(n):
        p = i * ratio
        k = int(p)
        f = p - k
        a, b = float(x[min(k, x.size - 1)]), float(x[min(k + 1, x.size - 1)])
        out[i] = a + (b - a) * f
    return out


def verify_multirate_stage0_onnx(onnx_path: Path, chunk: int = 4096) -> None:
    sess = onnxruntime.InferenceSession(str(onnx_path), _make_session_options(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    raw = (0.3 * rng.standard_normal(9001)).astype(np.float32)
    raw[:4] = [1.5, -1.5, 0.25, -0.75]
    x = np.clip(raw, -1.0, 1.0).astype(np.float32)
    audio = raw.reshape(1, 1, -1)
    for rate in (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000):
        n = max(1, x.size * OUTPUT_SAMPLE_RATE // rate)
        parts = [
            sess.run(None, {
                "audio": audio, "sample_rate": np.array([rate], np.int64),
                "out_start": np.array([start], np.int64), "out_count": np.array([min(chunk, n - start)], np.int64),
            })[0].reshape(-1)
            for start in range(0, n, chunk)
        ]
        got = np.concatenate(parts)
        diff = float(np.max(np.abs(got - reference_multirate_resample(x, rate))))
        if got.size != n or diff > 2e-5:
            raise RuntimeError(f"{onnx_path} @ {rate} Hz: size {got.size}/{n}, max |diff| {diff}")
        print(f"SUCCESS: {rate} Hz -> 16 kHz, {n} samples, max |diff| = {diff:.3e}")


def verify_multirate_onnx(
    onnx_path: Path,
    *,
    baseline_model: onnx.ModelProto | None = None,
    reference_onnx_path: Path | None = None,
) -> None:
    model = onnx.load(str(onnx_path), load_external_data=False)
    _validate_multirate_io(model)
    _assert_structural_hygiene(
        model, onnx_path.name,
        required_props=(("tts_reference_polyphase_conv", "1"), ("tts_reference_polyphase_v2", "1")),
    )
    verify_multirate_stage0_onnx(onnx_path)
    opt_sess = onnxruntime.InferenceSession(str(onnx_path), _make_session_options(), providers=["CPUExecutionProvider"])
    base_sess = onnxruntime.InferenceSession(baseline_model.SerializeToString(), _make_session_options(), providers=["CPUExecutionProvider"]) if baseline_model is not None else None
    ref_sess = onnxruntime.InferenceSession(str(reference_onnx_path), _make_session_options(), providers=["CPUExecutionProvider"]) if reference_onnx_path is not None else None

    rng = np.random.default_rng(20261127)
    max_vs_base = max_vs_ref = 0.0
    checks = 0
    for sr in (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000):
        for n_in in (1, 2, 3, 17, 45, 100, 441, 483, 1764, 3200, 50000):
            x = rng.uniform(-1.3, 1.3, (1, 1, n_in)).astype(np.float32)
            total = max(1, (n_in * OUTPUT_SAMPLE_RATE) // sr)
            starts = sorted({0, min(1, total - 1), min(11, total - 1), total // 2, max(0, total - 25), total + 50})
            for st in starts:
                counts = sorted({1, min(13, max(1, total - st)), min(648, max(1, total)), min(4096, max(1, total))})
                for cnt in counts:
                    feeds = {"audio": x, "sample_rate": np.array([sr], np.int64), "out_start": np.array([st], np.int64), "out_count": np.array([cnt], np.int64)}
                    y_opt = opt_sess.run(None, feeds)[0]
                    if y_opt.dtype != np.float32 or y_opt.shape != (1, 1, cnt) or not np.isfinite(y_opt).all():
                        raise RuntimeError(f"Invalid output sr={sr} n={n_in} st={st} cnt={cnt}: shape={y_opt.shape}")
                    in_bounds = max(0, min(cnt, total - st))
                    for tag, other_sess in (("stage0", base_sess), ("reference", ref_sess)):
                        if other_sess is None:
                            continue
                        y_other = other_sess.run(None, feeds)[0]
                        rel = float(np.max(np.abs(y_opt - y_other) / np.maximum(np.float32(1.0), np.abs(y_other))))
                        if rel >= 2e-5:
                            raise RuntimeError(f"{tag} rel mismatch sr={sr} n={n_in} st={st} cnt={cnt}: {rel:.3e}")
                        if in_bounds > 0:
                            err_in = float(np.max(np.abs(y_opt[0, 0, :in_bounds] - y_other[0, 0, :in_bounds])))
                            if err_in >= 5e-6:
                                raise RuntimeError(f"{tag} mismatch sr={sr} n={n_in} st={st} cnt={cnt}: {err_in:.3e}")
                            if tag == "stage0":
                                max_vs_base = max(max_vs_base, err_in)
                            else:
                                max_vs_ref = max(max_vs_ref, err_in)
                    checks += 1
        x_neg = rng.uniform(-1.2, 1.2, (1, 1, 500)).astype(np.float32)
        for st_neg, cnt_neg in ((-1, 10), (-10, 5), (-160, 10)):
            y_neg = opt_sess.run(None, {"audio": x_neg, "sample_rate": np.array([sr], np.int64), "out_start": np.array([st_neg], np.int64), "out_count": np.array([cnt_neg], np.int64)})[0]
            if y_neg.dtype != np.float32 or y_neg.shape != (1, 1, cnt_neg) or not np.isfinite(y_neg).all():
                raise RuntimeError(f"Invalid negative out_start output sr={sr} st={st_neg} cnt={cnt_neg}")
            checks += 1

    # Bit-exact 40 ms overlap-stitched streaming parity (matches native_aec_regression.cjs)
    for sr in (11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000):
        block = sr // 25
        g = math.gcd(sr, OUTPUT_SAMPLE_RATE)
        step_in, step_out = sr // g, OUTPUT_SAMPLE_RATE // g
        radius = max(1, int(round(sr / float(OUTPUT_SAMPLE_RATE) * 8.0)))
        overlap_out = 8
        need_in = radius + (overlap_out * sr + OUTPUT_SAMPLE_RATE - 1) // OUTPUT_SAMPLE_RATE
        k_steps = max(1, (need_in + step_in - 1) // step_in)
        left_in, left_out = k_steps * step_in, k_steps * step_out
        sig = (0.5 * np.sin(2.0 * math.pi * 1000.0 * np.arange(block * 4, dtype=np.float64) / sr)).astype(np.float32)

        def _run(arr: np.ndarray, st: int, cnt: int) -> np.ndarray:
            return opt_sess.run(None, {"audio": arr.reshape(1, 1, -1), "sample_rate": np.array([sr], np.int64), "out_start": np.array([st], np.int64), "out_count": np.array([cnt], np.int64)})[0].reshape(-1)

        full = _run(sig, 0, 640 * 4)
        stitched = np.zeros(640 * 4, dtype=np.float32)
        for k in range(4):
            if k == 0:
                stitched[:640] = _run(sig[:block], 0, 640)
            else:
                stitched[k * 640 - overlap_out:(k + 1) * 640] = _run(sig[k * block - left_in:(k + 1) * block], left_out - overlap_out, 640 + overlap_out)
        if not np.array_equal(full, stitched):
            raise RuntimeError(f"40 ms overlap-stitch not bit-exact at {sr} Hz: max_diff={float(np.max(np.abs(full - stitched))):.3e}")

    ref_msg = f", max|opt-ref|={max_vs_ref:.3e}" if ref_sess is not None else ""
    base_msg = f", max|opt-stage0|={max_vs_base:.3e}" if base_sess is not None else ""
    print(f"SUCCESS: {onnx_path.name} passed optimized polyphase verification ({checks} slice checks{base_msg}{ref_msg}, 9/9 rates bit-exact 40 ms overlap-stitch)")


# ==========================================
# 5. Unified CLI
# ==========================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ONNX audio resampler exporter, optimizer, and verifier.")
    parser.add_argument("--target", choices=("capture", "multirate", "simple", "all"), default=EXPORT_TARGET, help=f"Export target (default: {EXPORT_TARGET}).")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path (valid for single-target runs).")
    parser.add_argument("--reference-onnx", type=Path, default=None, help="Optional reference ONNX model for parity verification.")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX Runtime verification after export.")

    # Capture target options
    cap = parser.add_argument_group("capture target options")
    cap.add_argument("--mode", choices=VALID_CAPTURE_MODES, default=None, help=f"Capture mode (default: {CAPTURE_MODE}).")
    cap.add_argument("--unified", action="store_true", help="Shorthand for --target capture --mode unified.")
    cap.add_argument("--output-dtype", choices=("F32", "F16", "INT16"), default=None, help="Output dtype (capture: F32|F16; simple: F32|INT16).")

    # Multi-rate target options
    mr = parser.add_argument_group("multirate target options")
    mr_mode = mr.add_mutually_exclusive_group()
    mr_mode.add_argument("--polyphase", action="store_true", help="Export optimized polyphase multi-rate model (default).")
    mr_mode.add_argument("--legacy-stage0", action="store_true", help="Export legacy Stage-0 PyTorch multi-rate model.")
    mr.add_argument("--optimize-input", type=Path, default=None, help="Optimize an existing Stage-0 or legacy v2 multirate ONNX file.")

    # Simple target options
    smp = parser.add_argument_group("simple target options")
    smp.add_argument("--dynamic", action="store_true", default=SIMPLE_DYNAMIC, help="Enable dynamic length axis for simple resampler.")
    smp.add_argument("--no-mean-pooling", action="store_true", help="Disable Reshape+Mean integer downsampling optimization in simple resampler.")
    smp.add_argument("--input-length", type=int, default=SIMPLE_INPUT_LENGTH, help=f"Input length for simple resampler (default: {SIMPLE_INPUT_LENGTH}).")
    smp.add_argument("--ratio", type=float, default=SIMPLE_RESAMPLE_RATIO, help=f"Resample scale factor for simple resampler (default: {SIMPLE_RESAMPLE_RATIO:.4f}).")
    smp.add_argument("--input-dtype", choices=("F32", "INT16"), default=SIMPLE_INPUT_DTYPE, help=f"Input dtype for simple resampler (default: {SIMPLE_INPUT_DTYPE}).")
    return parser.parse_args()


def _run_capture(args: argparse.Namespace, verify: bool) -> None:
    mode = "unified" if args.unified else (args.mode if args.mode is not None else CAPTURE_MODE)
    if mode == "unified":
        if args.output_dtype is not None and args.output_dtype != "F16":
            raise ValueError("Capture mode 'unified' requires FLOAT16 state/output (--output-dtype F16).")
        path, decimation, taps = export_capture_unified_to_onnx(FIR_INPUT_SAMPLE_RATE, args.output)
        if verify:
            verify_capture_unified_onnx(path, decimation, taps, reference_onnx_path=args.reference_onnx)
        return
    out_dtype = args.output_dtype if args.output_dtype is not None else CAPTURE_OUTPUT_DTYPE
    if out_dtype not in ("F32", "F16"):
        raise ValueError(f"Capture operating-point output dtype must be F32 or F16, got {out_dtype!r}")
    path, spec, decimation, taps = export_capture_operating_point_to_onnx(mode, output_dtype=out_dtype, output_path=args.output)
    if verify:
        verify_capture_operating_point_onnx(path, spec, out_dtype, decimation, taps, reference_onnx_path=args.reference_onnx)


def _run_multirate(args: argparse.Namespace, verify: bool) -> None:
    polyphase = True if args.polyphase else (False if args.legacy_stage0 else MULTIRATE_POLYPHASE)
    path, baseline = export_multirate_to_onnx(args.output, polyphase=polyphase, source_onnx_path=args.optimize_input, need_baseline=verify)
    if verify:
        if polyphase:
            verify_multirate_onnx(path, baseline_model=baseline, reference_onnx_path=args.reference_onnx)
        else:
            verify_multirate_stage0_onnx(path)


def _run_simple(args: argparse.Namespace, verify: bool) -> None:
    out_dtype = args.output_dtype if args.output_dtype is not None else SIMPLE_OUTPUT_DTYPE
    if out_dtype not in ("F32", "INT16"):
        raise ValueError(f"Simple resampler output dtype must be F32 or INT16, got {out_dtype!r}")
    path = export_simple_resampler_to_onnx(
        input_length=args.input_length,
        resample_ratio=args.ratio,
        dynamic=args.dynamic,
        use_mean_pooling=not args.no_mean_pooling,
        input_dtype=args.input_dtype,
        output_dtype=out_dtype,
        output_path=args.output,
    )
    if verify:
        verify_simple_resampler_onnx(
            path, input_length=args.input_length, resample_ratio=args.ratio, input_dtype=args.input_dtype, output_dtype=out_dtype
        )


def main() -> None:
    args = _parse_args()
    verify = VERIFY_EXPORTED_MODEL and not args.no_verify
    target = "capture" if args.unified else args.target
    if target == "all":
        if args.output is not None or args.reference_onnx is not None:
            raise ValueError("--output and --reference-onnx require a single --target (capture, multirate, or simple).")
        _run_capture(args, verify)
        _run_multirate(args, verify)
        _run_simple(args, verify)
    elif target == "capture":
        _run_capture(args, verify)
    elif target == "multirate":
        _run_multirate(args, verify)
    else:
        _run_simple(args, verify)


if __name__ == "__main__":
    main()
