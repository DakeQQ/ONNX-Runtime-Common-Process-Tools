import torch
import torch.nn as nn
from onnxslim import slim

# Export to ONNX
DTYPE = torch.float32                     # [torch.float32, torch.int16]
DUMMY_CACHE_LENGTH = 64000                # Only for tracing; all lengths are dynamic.
DUMMY_WINDOW_LENGTH = 24000
update_save_path = "cache_update.onnx"
slice_save_path = "cache_slice.onnx"
OPSET = 23


class CacheUpdate(nn.Module):
    """Sliding audio cache: drop the oldest `advance` samples and replace the newest len(window) samples.

    cache covers [F - C, F); the output covers [F + advance - C, F + advance) and ends with `window`.
    Requires advance <= len(window) <= C.
    """

    def forward(self, cache, window, advance):
        cache_length = cache.shape[2]
        keep_end = cache_length - window.shape[2] + advance
        kept = cache[:, :, advance:keep_end]
        return torch.cat([kept, window], dim=2)


class CacheSlice(nn.Module):
    """segment = cache[start:end] (indices relative to the cache head)."""

    def forward(self, cache, start, end):
        return cache[:, :, start:end]


def _export(model, args, path, input_names, output_names, dynamic_axes):
    torch.onnx.export(
        model.eval(),
        args,
        path,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False
    )
    slim(
        model=path,
        output_model=path,
        no_shape_infer=False,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=False,
        verbose=False
    )
    print(f"Model exported successfully to {path}")


def export_to_onnx():
    cache = torch.zeros((1, 1, DUMMY_CACHE_LENGTH), dtype=DTYPE)
    window = torch.ones((1, 1, DUMMY_WINDOW_LENGTH), dtype=DTYPE)
    _export(CacheUpdate(), (cache, window, torch.tensor([1600], dtype=torch.int64)), update_save_path,
            ['cache', 'window', 'advance'], ['cache_out'],
            {'cache': {2: 'cache_length'}, 'window': {2: 'window_length'}, 'cache_out': {2: 'cache_length'}})
    _export(CacheSlice(), (cache, torch.tensor([10], dtype=torch.int64), torch.tensor([5000], dtype=torch.int64)),
            slice_save_path, ['cache', 'start', 'end'], ['segment'],
            {'cache': {2: 'cache_length'}, 'segment': {2: 'segment_length'}})


def verify():
    import numpy as np
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    cache = rng.standard_normal((1, 1, 1000)).astype(np.float32)
    window = rng.standard_normal((1, 1, 300)).astype(np.float32)
    update = ort.InferenceSession(update_save_path, providers=["CPUExecutionProvider"])
    out = update.run(None, {"cache": cache, "window": window, "advance": np.array([40], np.int64)})[0]
    expected = np.concatenate([cache[:, :, 40:1000 - 300 + 40], window], axis=2)
    assert out.shape == cache.shape and np.array_equal(out, expected)
    seg = ort.InferenceSession(slice_save_path, providers=["CPUExecutionProvider"]).run(
        None, {"cache": cache, "start": np.array([7], np.int64), "end": np.array([123], np.int64)})[0]
    assert np.array_equal(seg, cache[:, :, 7:123])
    print("Verification passed.")


if __name__ == "__main__":
    export_to_onnx()
    verify()
