"""Export the loudspeaker-latency probe as ONNX graphs (float32 only).

Probe_Generator_{sr}[_Optimized].onnx : no inputs  -> probe float32[M]
Probe_Delay_{sr}[_Optimized].onnx     : recording float32[N], submitted int64[K]
                            -> offset int64[K], pfr, peak_ratio, echo_dbfs, snr_db float32[K], status int32[K]

status: 0 ok, 1 non-finite, 2 recording shorter than probe, 3 search window not covered,
        4 pfr, 5 peak ratio, 6 echo level, 7 snr (same gate order as estimate_delay).
Graphs are optimized in the PyTorch source, simplified with onnxslim in memory, validated,
and written atomically as *_Optimized.onnx; no raw/intermediate model is persisted.
Usage: python Export_Probe_Delay.py [sample_rate ...] [--output-dir DIR]   (default 16000 48000)
Env OUTPUT_DIR overrides the destination folder (default: current directory).
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path

import numpy as np
import torch

# ============================================================================
# EXPORT CONFIGURATION & PROBE PARAMETERS
# ============================================================================
OPSET = 23                              # ONNX operator set version (broad runtime compatibility)
DEFAULT_SAMPLE_RATES = (16000, 48000)   # Standard audio sample rates supported (Hz)
FORBIDDEN_OPS = {"Loop", "If", "Scan", "DFT", "STFT", "SequenceAt", "SequenceConstruct", "SplitToSequence"}
DELAY_OUTPUTS = ["offset", "pfr", "peak_ratio", "echo_dbfs", "snr_db", "status"]
DB_PER_LN = 20.0 / math.log(10.0)       # Conversion factor: natural log (ln) to decibels (20 * log10(x))

# Latency search window constraints
MIN_LATENCY_MS: int = 0                 # Minimum plausible acoustic round-trip delay (ms)
MAX_LATENCY_MS: int = 600               # Maximum search window bound (ms); delays beyond are rejected

# Signal detection and validation thresholds (gate parameters)
MIN_PFR: float = 8.0                    # Minimum Peak-to-Floor Ratio: peak / mean correlation floor
MAX_PEAK_RATIO: float = 0.7             # Peak uniqueness: max secondary peak outside main lobe / main peak
MIN_ECHO_DBFS: float = -55.0            # Minimum echo energy in dBFS (rejects silent/inaudible captures)
MIN_SNR_DB: float = 6.0                 # Minimum signal-to-noise ratio in dB over background noise floor
NOISE_WINDOW_MS: int = 100              # Duration of recording before probe submission used for noise estimation
PROBE_GUARD_MS: int = 150               # Reverberation decay margin following probe playback (ms)

# Chirp acoustic waveform specifications
PROBE_DURATION_S: float = 0.15          # Probe duration in seconds (150 ms)
PROBE_F_START: float = 400.0            # Start frequency of linear chirp sweep in Hz
PROBE_F_END: float = 3200.0             # End frequency of linear chirp sweep in Hz (< Nyquist for sr >= 8000)
PROBE_AMPLITUDE: float = 0.25           # Peak chirp waveform amplitude (linear scale, avoid clipping)


def _default_output_dir() -> Path:
    """Determine the default model export directory.
    
    Priority order:
    1. OUTPUT_DIR environment variable
    2. AI_VOICE_DIR environment variable + '/models/calibration'
    3. Current script directory (standalone default)
    """
    if "OUTPUT_DIR" in os.environ:
        return Path(os.environ["OUTPUT_DIR"]).resolve()
    if "AI_VOICE_DIR" in os.environ:
        return (Path(os.environ["AI_VOICE_DIR"]) / "models" / "calibration").resolve()
    return Path(__file__).resolve().parent


OUTPUT_DIR = _default_output_dir()


# ============================================================================
# PURE NUMPY ALGORITHM REFERENCE (CALIBRATION & VERIFICATION)
# ============================================================================

def generate_chirp_probe(sample_rate: int = 16000, duration_s: float = PROBE_DURATION_S,
                         f_start: float = PROBE_F_START, f_end: float = PROBE_F_END,
                         amplitude: float = PROBE_AMPLITUDE) -> np.ndarray:
    """Synthesize a linear frequency sweep (chirp) with Hann tapers.
    
    The instantaneous frequency sweeps linearly from f_start to f_end:
        f(t) = f_start + k * t, where k = (f_end - f_start) / duration
        phase(t) = 2 * pi * (f_start * t + 0.5 * k * t^2)
        chirp(t) = amplitude * sin(phase(t))
    
    A 10 ms Hann taper is applied at both ends to eliminate abrupt onset/offset
    clicks and suppress high-frequency spectral leakage.
    
    Returns:
        np.ndarray: float32 array of shape [M], where M = round(sr * duration).
    """
    if f_end >= sample_rate / 2 or f_start >= sample_rate / 2:
        raise ValueError(f"chirp frequencies {f_start}-{f_end} Hz must be below Nyquist ({sample_rate / 2} Hz)")
    total_samples = max(1, int(round(sample_rate * duration_s)))
    t = np.arange(total_samples, dtype=np.float32) / np.float32(sample_rate)
    k = np.float32((f_end - f_start) / float(duration_s))
    phase = np.float32(2.0 * math.pi) * (np.float32(f_start) * t + np.float32(0.5) * k * t * t)
    chirp = np.float32(amplitude) * np.sin(phase)
    taper_len = min(total_samples // 4, int(round(sample_rate * 0.01)))
    if taper_len > 0:
        window = np.float32(0.5) - np.float32(0.5) * np.cos(
            np.float32(np.pi) * np.arange(taper_len, dtype=np.float32) / np.float32(taper_len))
        chirp[:taper_len] *= window
        chirp[-taper_len:] *= window[::-1]
    return np.asarray(chirp, dtype=np.float32)


def probe_spacing_ms() -> float:
    """Calculate the minimum safe time spacing (ms) between consecutive probes.
    
    Spacing equals MAX_LATENCY_MS + chirp duration + PROBE_GUARD_MS to allow room
    reverberation and echo to completely subside before the next probe is emitted.
    """
    return MAX_LATENCY_MS + PROBE_DURATION_S * 1000 + PROBE_GUARD_MS


def _dbfs(x: np.ndarray) -> float:
    """Compute root-mean-square (RMS) level of a signal in dBFS (full-scale dB).
    
    Returns -120.0 dBFS for silence (RMS <= 1e-12) to avoid log of zero.
    """
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float32)))) if x.size else 0.0
    return 20.0 * math.log10(rms) if rms > 1e-12 else -120.0


@dataclass(frozen=True)
class DelayEstimate:
    """Structured result of acoustic delay estimation for a single probe.
    
    Attributes:
        index: Absolute sample index in recording where the echo starts (None if invalid).
        offset_samples: Delay in samples relative to probe submission (index - submitted).
        pfr: Peak-to-Floor Ratio of the matched-filter cross-correlation.
        peak_ratio: Ratio of secondary correlation peak to main peak outside main lobe.
        echo_dbfs: Measured RMS level of the detected echo segment in dBFS.
        snr_db: Estimated SNR (echo RMS vs pre-submit noise RMS) in dB.
        valid: True if all quality gates passed (status == 0).
        reason: Human-readable diagnostic description of pass or rejection reason.
    """
    index: int | None
    offset_samples: int | None
    pfr: float
    peak_ratio: float
    echo_dbfs: float
    snr_db: float
    valid: bool
    reason: str


def estimate_delay(reference: np.ndarray, recording: np.ndarray, sample_rate: int, submitted: int = 0, *,
                   min_latency_ms: int = MIN_LATENCY_MS, max_latency_ms: int = MAX_LATENCY_MS,
                   min_pfr: float = MIN_PFR, max_peak_ratio: float = MAX_PEAK_RATIO,
                   min_echo_dbfs: float = MIN_ECHO_DBFS, min_snr_db: float = MIN_SNR_DB,
                   noise_window_ms: int = NOISE_WINDOW_MS) -> DelayEstimate:
    """Windowed matched-filter cross-correlation with multi-stage quality gates.
    
    Gate Evaluation Hierarchy (same rejection priority as ProbeDelay ONNX graph):
        1. Non-finite values: checks for NaN / Inf in probe or recording (status=1).
        2. Recording length: recording must be at least as long as the probe (status=2).
        3. Search coverage: recording must fully encompass [w + lag_min, w + lag_max + M] (status=3).
        4. Peak-to-Floor Ratio (PFR): peak correlation / mean correlation floor >= min_pfr (status=4).
        5. Peak Uniqueness: second-highest peak outside main lobe (|lag - lag*| > M) must not exceed
           max_peak_ratio * peak (status=5).
        6. Echo Energy: RMS energy of detected echo window >= min_echo_dbfs (status=6).
        7. Signal-to-Noise Ratio: ratio of echo RMS to pre-submit noise RMS >= min_snr_db (status=7).
        Passed: status=0 ("ok").
    """
    ref = np.asarray(reference, dtype=np.float32).ravel()
    rec = np.asarray(recording, dtype=np.float32).ravel()
    m, n_rec = ref.shape[0], rec.shape[0]

    def invalid(reason: str, pfr: float = 0.0, ratio: float = 0.0, echo: float = -120.0,
                snr: float = -math.inf) -> DelayEstimate:
        return DelayEstimate(None, None, pfr, ratio, echo, snr, False, reason)

    # Gate 1: Check for non-finite samples
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(rec)):
        return invalid("non-finite samples in probe or recording")
        
    # Gate 2: Recording must be at least as long as probe
    if m == 0 or n_rec < m:
        return invalid("recording shorter than probe")
        
    # Gate 3: Check search window coverage
    w = int(submitted)
    k_min = w + math.ceil(min_latency_ms * sample_rate / 1000)
    k_max = min(w + math.floor(max_latency_ms * sample_rate / 1000), n_rec - m)
    if k_min < 0 or k_max < k_min:
        return invalid("recording does not cover the search window")
        
    # Time-domain direct float32 cross-correlation (avoids FFT spectral leakage into silent margins)
    window = np.abs(np.correlate(rec[k_min:k_max + m], ref, mode="valid"))
    corr = np.zeros(n_rec, dtype=np.float32)
    corr[k_min:k_max + 1] = window
    k_star = k_min + int(np.argmax(window))
    peak = float(corr[k_star])
    
    # Gate 4: Peak-to-Floor Ratio (PFR)
    pfr = peak / (float(np.mean(window)) + 1e-12)
    
    # Gate 5: Peak Uniqueness Ratio (secondary peak outside main lobe of width +/- m)
    mask = np.ones(window.shape[0], dtype=bool)
    mask[max(0, k_star - m - k_min):max(0, k_star + m + 1 - k_min)] = False
    ratio = float(np.max(window[mask])) / (peak + 1e-12) if mask.any() else 0.0
    
    # Gate 6: Echo Energy (RMS in dBFS)
    echo = rec[k_star:k_star + m]
    echo_dbfs = _dbfs(echo)
    
    # Gate 7: Signal-to-Noise Ratio (SNR in dB comparing echo to pre-submit noise floor)
    noise_len = int(noise_window_ms * sample_rate // 1000)
    if w >= noise_len and noise_len > 0:
        rms_e = float(np.sqrt(np.mean(np.square(echo))))
        rms_n = float(np.sqrt(np.mean(np.square(rec[w - noise_len:w]))))
        snr_db = 20.0 * math.log10(max(rms_e, 1e-30) / (rms_n + 1e-12))
    else:
        snr_db = math.inf
        
    if pfr < min_pfr:
        return invalid(f"pfr {pfr:.1f} < {min_pfr}", pfr, ratio, echo_dbfs, snr_db)
    if ratio > max_peak_ratio:
        return invalid(f"peak ratio {ratio:.2f} > {max_peak_ratio}", pfr, ratio, echo_dbfs, snr_db)
    if echo_dbfs < min_echo_dbfs:
        return invalid(f"echo {echo_dbfs:.1f} dBFS < {min_echo_dbfs}", pfr, ratio, echo_dbfs, snr_db)
    if snr_db < min_snr_db:
        return invalid(f"snr {snr_db:.1f} dB < {min_snr_db}", pfr, ratio, echo_dbfs, snr_db)
    return DelayEstimate(k_star, k_star - w, pfr, ratio, echo_dbfs, snr_db, True, "ok")


# ============================================================================
# PYTORCH ONNX EXPORT MODULES
# ============================================================================

class ProbeGenerator(torch.nn.Module):
    """Generates the constant acoustic chirp probe tensor.
    
    Inputs: None (parameterless graph).
    Outputs:
        probe: float32[M] chirp waveform buffer ready for loudspeaker playback.
    """
    def __init__(self, sample_rate: int):
        super().__init__()
        self.register_buffer("probe", torch.from_numpy(generate_chirp_probe(sample_rate=sample_rate)))

    def forward(self) -> torch.Tensor:
        return self.probe


class ProbeDelay(torch.nn.Module):
    """Batch matched filter (1D Conv) and validation gates for K probe submissions.
    
    Inputs:
        recording: float32[N] - continuous audio capture from microphone.
        submitted: int64[K]   - sample indices at which probe playback was triggered.
    
    Outputs:
        offset:     int64[K]   - detected latency in samples (argmax lag + lag_min).
        pfr:        float32[K] - Peak-to-Floor Ratio (peak correlation / mean floor).
        peak_ratio: float32[K] - secondary peak ratio outside the chirp main lobe.
        echo_dbfs:  float32[K] - echo segment energy in dBFS.
        snr_db:     float32[K] - SNR comparing echo RMS to pre-submit noise RMS.
        status:     int32[K]   - diagnostic status code (0: ok, 1-7: gate failures).
    """

    def __init__(self, sample_rate: int):
        super().__init__()
        probe = generate_chirp_probe(sample_rate=sample_rate)
        self.m = int(probe.shape[0])                                    # Probe length (samples)
        self.lag_min = math.ceil(MIN_LATENCY_MS * sample_rate / 1000)   # Minimum search lag (samples)
        lag_max = math.floor(MAX_LATENCY_MS * sample_rate / 1000)       # Maximum search lag (samples)
        n_lags = lag_max - self.lag_min + 1                             # Number of candidate lag offsets
        self.seg = n_lags + self.m - 1                                  # Segment size required for valid convolution
        self.noise = int(NOISE_WINDOW_MS * sample_rate // 1000)         # Noise floor analysis window (samples)
        self.pad_right = self.seg + self.m                              # Right-side padding margin
        self.n_lags = n_lags

        # Static convolution kernel and indexing buffers (offsets pre-added so forward does one Add each)
        self.register_buffer("kernel", torch.from_numpy(probe).reshape(1, 1, -1))
        self.register_buffer("lag_ar", torch.arange(n_lags, dtype=torch.int64)[None, :])
        self.register_buffer("seg_ar", torch.arange(self.noise + self.lag_min,
                                                    self.noise + self.lag_min + self.seg, dtype=torch.int64)[None, :])
        self.register_buffer("echo_ar", torch.arange(self.m, dtype=torch.int64)[None, :])
        self.register_buffer("noise_ar", torch.arange(max(self.noise, 1), dtype=torch.int64)[None, :])
        self.register_buffer("codes", torch.arange(8, dtype=torch.int32))

    def forward(self, recording: torch.Tensor, submitted: torch.Tensor):
        # 1. x - x is 0 only for finite x: one N-pass instead of IsInf/IsNaN/Or/Not
        finite = (recording - recording) == 0
        bad = ~finite.all()
        rec = torch.where(finite, recording, 0.0)
        n_rec = recording.shape[0]
        rec_p = torch.nn.functional.pad(rec, (self.noise, self.pad_right))
        last = n_rec + (self.noise + self.pad_right - 1)

        def take(idx: torch.Tensor) -> torch.Tensor:
            return rec_p[torch.minimum(idx.clamp(min=0), last)]

        # 2. Matched filter over each submission's search segment
        w = submitted[:, None]
        start = w + self.lag_min
        seg = take(w + self.seg_ar)
        corr = torch.nn.functional.conv1d(seg.unsqueeze(1), self.kernel).squeeze(1).abs()

        # 3. Valid lags are the contiguous range [lo, hi] of lag_ar; invalid lags are zeroed (corr >= 0,
        #    and argmax ties resolve to the first index, which is valid whenever coverage passes)
        lo = -start
        hi = (n_rec - self.m) - start
        valid = (self.lag_ar >= lo) & (self.lag_ar <= hi)
        masked = torch.where(valid, corr, 0.0)

        # 4. Peak and Peak-to-Floor Ratio; the valid count is analytic, not a ReduceSum over [K, L]
        k_idx = masked.argmax(dim=-1, keepdim=True)
        peak = masked.gather(1, k_idx)
        count = (hi.clamp(max=self.n_lags - 1) - lo.clamp(min=0) + 1).clamp(min=1)
        mean = masked.sum(-1, keepdim=True) / count.to(corr.dtype)
        pfr = peak / (mean + 1e-12)

        # 5. Secondary peak outside the main lobe; an empty set gives 0 / (peak + eps) = 0 directly
        far = (self.lag_ar - k_idx).abs() > self.m
        ratio = torch.where(far, masked, 0.0).amax(dim=-1, keepdim=True) / (peak + 1e-12)

        # 6. Echo window is a sub-range of seg, so GatherElements avoids re-clamping into rec_p
        echo = seg.gather(1, k_idx + self.echo_ar)
        rms_e = echo.square().mean(-1, keepdim=True).sqrt()
        echo_dbfs = torch.where(rms_e > 1e-12, DB_PER_LN * rms_e.clamp(min=1e-12).log(), -120.0)

        # 7. SNR against the noise window just before the submit watermark
        if self.noise > 0:
            rms_n = take(w + self.noise_ar).square().mean(-1, keepdim=True).sqrt()
            snr = DB_PER_LN * (rms_e.clamp(min=1e-30) / (rms_n + 1e-12)).log()
            snr = torch.where(w >= self.noise, snr, float("inf"))
        else:
            snr = torch.full_like(rms_e, float("inf"))

        # Gate priority 1 > 2 > ... > 7 > 0; with start >= 0, any valid lag exists iff hi >= 0
        coverage = (start < 0) | (hi < 0)
        codes = self.codes
        status = torch.where(snr < MIN_SNR_DB, codes[7], codes[0])
        status = torch.where(echo_dbfs < MIN_ECHO_DBFS, codes[6], status)
        status = torch.where(ratio > MAX_PEAK_RATIO, codes[5], status)
        status = torch.where(pfr < MIN_PFR, codes[4], status)
        status = torch.where(coverage, codes[3], status)
        status = torch.where(n_rec < self.m, codes[2], status)
        status = torch.where(bad, codes[1], status)
        return ((k_idx + self.lag_min).squeeze(1), pfr.squeeze(1), ratio.squeeze(1), echo_dbfs.squeeze(1),
                snr.squeeze(1), status.squeeze(1))


# ============================================================================
# GRAPH VERIFICATION & NUMERICAL PARITY TESTING
# ============================================================================

def _check_graph(name: str, data: bytes) -> None:
    """Checker, forbidden ops, no float64 tensors; prints node/initializer metrics."""
    import onnx

    model = onnx.load_from_string(data)
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    hist = collections.Counter(node.op_type for node in model.graph.node)
    bad = set(hist) & FORBIDDEN_OPS
    if bad:
        raise RuntimeError(f"{name}: forbidden ops {sorted(bad)}")
    if any(n.domain not in ("", "ai.onnx") for n in model.graph.node):
        raise RuntimeError(f"{name}: non-standard domain nodes")
    doubles = [t.name for t in model.graph.initializer if t.data_type == onnx.TensorProto.DOUBLE]
    doubles += [v.name for v in list(model.graph.input) + list(model.graph.output)
                if v.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE]
    if doubles:
        raise RuntimeError(f"{name}: float64 tensors {doubles}")
    init_bytes = sum(onnx.numpy_helper.to_array(t).nbytes for t in model.graph.initializer)
    print(f"  {name}: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers "
          f"({init_bytes} B), ops {dict(sorted(hist.items()))}")


def _parity(sample_rate: int, gen_data: bytes, delay_data: bytes) -> None:
    """Verify numerical parity between ONNX Runtime inference and NumPy reference.
    
    Synthesizes a realistic test recording containing probe chirps injected at known
    delays into Gaussian background noise, then verifies:
    1. ProbeGenerator output is bit-identical to generate_chirp_probe().
    2. ProbeDelay outputs match estimate_delay() within +/- 1 sample tolerance.
    """
    import onnxruntime as ort

    gen = ort.InferenceSession(gen_data, providers=["CPUExecutionProvider"])
    probe = gen.run(None, {})[0]
    ref = generate_chirp_probe(sample_rate=sample_rate)
    if not np.array_equal(probe, ref):
        raise RuntimeError("probe mismatch")
    sess = ort.InferenceSession(delay_data, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    w = int(0.2 * sample_rate)
    spacing = int(round(probe_spacing_ms() * sample_rate / 1000))
    submits = [w + i * spacing for i in range(3)]
    rec = rng.normal(0.0, 0.02, submits[-1] + spacing).astype(np.float32)
    for i, s in enumerate(submits):
        d = int(round((40 + 60 * i) * sample_rate / 1000))
        rec[s + d:s + d + ref.shape[0]] += ref
    out = sess.run(None, {"recording": rec, "submitted": np.asarray(submits, dtype=np.int64)})
    for i, s in enumerate(submits):
        est = estimate_delay(ref, rec, sample_rate, s)
        if int(out[5][i]) != 0 or not est.valid or abs(int(out[0][i]) - est.offset_samples) > 1:
            raise RuntimeError(f"parity failed for probe {i}: onnx {out[0][i]}/{out[5][i]} numpy {est}")
    print(f"  parity ok: offsets {out[0].tolist()}")


# ============================================================================
# EXPORT & OPTIMIZATION PIPELINE
# ============================================================================

def _export_bytes(model: torch.nn.Module, args: tuple, **kwargs) -> bytes:
    import time

    import onnx
    from onnxslim import slim
    from onnxslim.utils import print_model_info_as_table, summarize_model

    buf = io.BytesIO()
    with torch.inference_mode():
        torch.onnx.export(model.eval(), args, buf, opset_version=OPSET, dynamo=False, **kwargs)
    name = type(model).__name__
    raw = onnx.load_from_string(buf.getvalue())
    raw_info = summarize_model(raw, f"{name} (raw)")
    start = time.time()
    slimmed = slim(raw)
    if not isinstance(slimmed, onnx.ModelProto):
        raise RuntimeError("onnxslim did not return a model")
    # slim() only prints its summary table when given an output path, so print it here.
    print_model_info_as_table([raw_info, summarize_model(slimmed, f"{name} (slim)")], time.time() - start)
    return slimmed.SerializeToString()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def export(sample_rate: int, out_dir: Path | str = OUTPUT_DIR) -> tuple[Path, Path]:
    """Export both graphs in memory, validate them, then atomically write only the final models."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / f"Probe_Generator_{sample_rate}_Optimized.onnx"
    delay_path = out_dir / f"Probe_Delay_{sample_rate}_Optimized.onnx"
    gen_data = _export_bytes(ProbeGenerator(sample_rate), (), input_names=[], output_names=["probe"])
    model = ProbeDelay(sample_rate)
    n = model.noise + model.seg + model.m
    dummy = (torch.zeros(n, dtype=torch.float32), torch.tensor([model.noise, model.noise + 1], dtype=torch.int64))
    delay_data = _export_bytes(model, dummy, input_names=["recording", "submitted"], output_names=DELAY_OUTPUTS,
                               dynamic_axes={"recording": {0: "N"}, "submitted": {0: "K"},
                                             **{name: {0: "K"} for name in DELAY_OUTPUTS}})
    _check_graph(gen_path.name, gen_data)
    _check_graph(delay_path.name, delay_data)
    _parity(sample_rate, gen_data, delay_data)
    _atomic_write(gen_path, gen_data)
    _atomic_write(delay_path, delay_data)
    return gen_path, delay_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export loudspeaker-latency probe ONNX models.")
    parser.add_argument("sample_rates", nargs="*", type=int, default=list(DEFAULT_SAMPLE_RATES),
                        help=f"Sample rates to export (default: {' '.join(str(r) for r in DEFAULT_SAMPLE_RATES)})")
    parser.add_argument("--output-dir", "-o", type=Path, default=OUTPUT_DIR,
                        help=f"Directory to save exported models (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    for sr in args.sample_rates:
        print(f"Exporting probe models for {sr} Hz to {args.output_dir}")
        export(sr, out_dir=args.output_dir)
