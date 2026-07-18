## Task: Solving FP16 quant issue.

**Context / Problem Statement**
The attached ONNX exporter produces degraded or incorrect outputs after converting to FP16. This is very likely a **numeric overflow/underflow issue**: FP16 has a much smaller dynamic range than FP32 (max representable magnitude ≈ 65,504 vs. ~3.4×10³⁸ for FP32, and much coarser precision near small values). Certain ops are known overflow/precision hotspots and should be treated as suspects first:
- **Softmax / Attention** (exponentials, large logits before normalization)
- **LayerNorm / BatchNorm / GroupNorm** (variance, division, small epsilon values)
- **Exp, Pow, Reciprocal, Div, ReduceSum/ReduceMean, ReduceL2** over large tensors
- **GELU/Erf-based activations**
- Any accumulation over long sequences (e.g., large reduction dimensions)

**Primary Objective**
Produce a working inference pipeline where the target model runs under **FP16 or mixed precision** on **ONNX Runtime's CUDAExecutionProvider**, with output quality that is quantitatively close to the original FP32 baseline (define/report a similarity metric — e.g., cosine similarity, max absolute error, relative error, or task-specific metric such as PSNR/BLEU/accuracy depending on model type).

**Required Approach — Please work through these steps explicitly:**

1. **Diagnose first, convert second.**
   - Run the FP32 model and the naive FP16-converted model side-by-side on the same inputs.
   - Identify which node(s)/subgraph(s) first produce NaN/Inf or large deviation (layer-by-layer output comparison, e.g., via ONNX Runtime's node output dumping, Polygraphy `--validate`, or manual graph splitting).
   - Report which op types are the overflow source.

2. **Use a mixed-precision conversion strategy, not blanket FP16 casting.**
   - Use tooling such as `onnxconverter-common.float16.convert_float_to_float16` with `keep_io_types=True` and an explicit `op_block_list` / `node_block_list` to keep numerically sensitive ops (Softmax, LayerNorm, Exp, Pow, etc.) in FP32.
   - Alternatively use ONNX Runtime's built-in transformer FP16 optimization utilities if this is a transformer-based model.

3. **Minimize Cast node overhead — this is a key requirement.**
   - After conversion, inspect the graph for `Cast(F32→F16)` and `Cast(F16→F32)` pairs.
   - **Sandwich pattern rule:** If a short chain of FP16 nodes is immediately preceded and followed by casts back to FP32 (i.e., `F32 → Cast→F16 → [F16 op(s)] → Cast→F32`), and that chain is not critical for the FP16 speed/memory benefit, **convert that entire sandwiched chain to FP32** instead, eliminating both casts. This avoids paying the cast latency cost while gaining no real precision/perf benefit from the tiny FP16 island.
   - Merge/eliminate redundant consecutive casts (`Cast→Cast` collapsing) using graph-level optimization (e.g., `onnx-graphsurgeon`, `onnxoptimizer`, or ONNX Runtime graph transformers).
   - Aim to minimize the *total count* of Cast nodes in the final graph while keeping numerically fragile ops in FP32.

4. **Validate on CUDAExecutionProvider specifically**, not just CPU — confirm the optimized graph actually runs correctly under `CUDAExecutionProvider` (some fused/half-precision kernels behave differently than CPU fallback), and check ORT logs/warnings for unsupported FP16 kernels causing implicit fallback casts.

5. **Report back:**
   - The overflow root-cause node(s) found in step 1.
   - The final graph's Cast node count vs. the naive-conversion baseline.
   - Quantitative output quality comparison (FP32 baseline vs. optimized mixed-precision) using an appropriate metric.
   - Any remaining known limitations or ops still forced to FP32 and why.

**Success Criteria**
- Model runs on ONNX Runtime + CUDAExecutionProvider without NaN/Inf outputs.
- Output quality metric within an acceptable tolerance of FP32 baseline (state the threshold used).
- Cast node count in the graph is measurably reduced compared to a naive full-FP16 conversion.
- Numerically sensitive ops (Softmax, LayerNorm, etc.) remain in FP32 where needed, with FP32 "islands" merged/expanded to avoid unnecessary sandwich-cast overhead.

---

