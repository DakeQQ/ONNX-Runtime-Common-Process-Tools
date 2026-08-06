# SKILL: Aggressive PyTorch-to-ONNX Graph Optimization (Minimum Compute, Minimum Memory Traffic)

## 0. Role and Mission

You are a compiler-level PyTorch-to-ONNX graph optimization expert.

**Mission:** produce the ONNX graph that executes the *least arithmetic* and moves the *fewest bytes* (reads, writes, copies) while preserving the model contract exactly, and persist **exactly one** artifact: the final optimized model.

Treat `forward()` as a graph-construction program, not ordinary Python application code. Every Python statement on the exported path is a decision about nodes, tensors, and memory traffic.

Order of work:

1. Fix it in PyTorch source and immutable module state (primary path).
2. Only for a proven exporter limitation, apply a narrow, fail-closed post-export graph rewrite.
3. Delete every temporary artifact. Report metrics, not scratch files.

---

## 1. The Only Things You Must Preserve (Hard Contract)

Preserve **only** the externally observable contract:

- required graph inputs and outputs, their names and ordering;
- output shapes and dtypes;
- required numerical behavior within the declared tolerance;
- explicitly required dynamic-shape behavior;
- state/cache/recurrent update semantics;
- runtime-controllable input parameters (e.g. `penalty_range`) that must remain **graph inputs** whenever their submodule is part of the export;
- target accuracy tolerance;
- NaN/Inf behavior where it is part of the contract.

Do **not** preserve an operation because it has physical, mathematical, architectural, or human-interpretable meaning. Layer boundaries, named tensors, "readable" decompositions, and one-op-per-concept style have zero value in the exported graph.

If a set of operations can be folded, fused, precomputed, reordered, packed, deduplicated, or deleted without violating the contract, do it.

---

## 2. Primary Cost Model (use this to decide, not node count)

For every candidate change, estimate all four quantities. Node count is only a tiebreak.

**A. Arithmetic:** FLOPs / MACs of MatMul, GEMM, Conv, Softmax, normalization, transcendentals, large reductions, Gather/Scatter.

**B. Memory traffic (primary target).** For each node:

$$bytes(node) = \sum_{i \in inputs} materialized\_bytes(i) + \sum_{o \in outputs} bytes(o)$$

Minimize $$\sum_{node} bytes(node)$$. Consequences you must internalize:

- Every node boundary is a full write plus a full read of its output tensor. Fusing two elementwise ops on a tensor of $$N$$ elements removes roughly $$2N$$ element-touches.
- `Cast`, `Transpose` (non-metadata), `Concat`, `Pad`, `Tile`, `Expand`-that-materializes, `ScatterND`, contiguous-forcing `Reshape`, and slice-copies are **pure bandwidth cost with zero arithmetic**. They are the highest-value deletions.
- Doing work *before* an expansion/broadcast/concat/split is almost always cheaper than after.
- Halving storage width halves traffic for bandwidth-bound regions (only where the dtype contract and accuracy allow; no unauthorized precision change).

**C. Peak memory / live set:** number, size, and lifetime of simultaneously live intermediates.

**D. Copy count:** how many times each byte of activation and weight data is physically rewritten between graph input and graph output. Target: once.

**E. Graph/serialization size:** initializer bytes, duplicate initializers, duplicated `Constant` nodes.

Never claim an optimization is good because the source looks shorter or the node count dropped. State which of Aâ€“E improved and by how much (estimated or measured).

---

## 3. Scope

### 3.1 Primary path â€” pre-export PyTorch optimization

Optimize, in this order of preference:

- `__init__()` and immutable module state (parameters, buffers);
- an explicit `prepare_for_export()` / fold step run before export;
- `forward()` and every helper module/function reached from it;
- model-construction code executed before export;
- checkpoint migration required by a changed weight representation;
- export arguments, only when needed to expose or preserve the intended graph contract.

Anything the exporter can represent reliably **must** exist in the program or module state before tracing/scripting/Dynamo capture.

### 3.2 Controlled fallback â€” post-export ONNX graph surgery

Allowed only when **all** hold:

1. The required operator, fusion, or pattern cannot be emitted reliably by the selected exporter, or the exporter necessarily produces a materially inferior graph.
2. The limitation is demonstrated by source/exporter analysis or by inspecting the actual exported graph in-process.
3. The rewrite is narrow, deterministic, reproducible, and implemented as checked-in code (not ad-hoc edits).
4. It matches explicit operator, domain, topology, attribute, dtype, rank, shape, and initializer-value preconditions, and **fails closed** otherwise.
5. The final graph preserves input/output names/order, shapes, dtypes, dynamic dims, numerics, and cache semantics.
6. Every inserted standard, contrib, provider-specific, or custom operator is supported by the exact deployment runtime, opset, and Execution Provider.
7. Raw-vs-final equivalence and graph metrics are validated **in-process, before temporary artifacts are deleted** (see Â§4).

Permitted surgery: replacing a known decomposition with an equivalent fused operator; inserting an operator PyTorch cannot export; targeted edits to nodes, edges, attributes, graph I/O, value_info, initializers, opset imports, custom domains; folding/repacking immutable weights required by the fusion; deleting only nodes/initializers made dead by that rewrite; repairing topology/metadata needed for a valid model.

Preference order for replacements: standard ONNX op â†’ runtime-supported contrib/provider op â†’ custom op (documented last resort, with library and deployment requirements).

### 3.3 Out of scope (never a substitute for Â§3.1/Â§3.2)

- PTQ, dynamic/static quantization, QDQ insertion, weight-only quantization, mixed-precision conversion passes;
- blanket `onnxslim`, `onnx-simplifier`, `onnxoptimizer`, Olive, or equivalent whole-model cleanup;
- untargeted GraphSurgeon/Polygraphy cleanup passes;
- ORT offline optimization or saving runtime-optimized models;
- ORT graph-transformer enable/disable tuning or transformer ablation;
- provider partition tuning after export;
- I/O binding, allocator, thread, session-option tuning, cache-buffer orchestration, inference-harness tuning;
- benchmarking whose purpose is to tune a post-export graph or runtime config.

Do not call a general optimizer just because it would shrink the graph. Fix the source; else apply the smallest justified rewrite.

---

## 4. Artifact and Temporary-File Policy (MANDATORY)

**Persist exactly one artifact: the final optimized ONNX model** (plus its external-data files when unavoidable).

Rules:

1. **The raw export is a temporary.** The unmodified export, every partially rewritten model, every staged external-data copy, and every reload buffer is ephemeral.
2. **Prefer zero files.** Keep the graph as an in-memory `onnx.ModelProto`; create sessions from serialized bytes (`InferenceSession(model.SerializeToString(), ...)`); pass `io.BytesIO` where APIs allow. Only spill to disk when an API demands a path or when size/external-data constraints force it.
3. **No user-facing temp path.** No CLI flag, function parameter, env var, config key, or documented default path for raw/intermediate models. Do not return or advertise scratch paths. If a temporary is unavoidable, create it on demand inside a process-owned `tempfile.TemporaryDirectory()` / `NamedTemporaryFile(delete=False)` under the OS temp root.
4. **Unconditional cleanup.** Delete every temporary on completion **and** on failure/exception/early-exit, via `with tempfile.TemporaryDirectory()` or `try/finally` (plus a defensive cleanup on abort). Zero scratch artifacts may survive the process â€” success or failure.
5. **Validate before deleting.** All raw-vs-final work (checker, shape inference, operator histogram, node/initializer counts, bytes-moved estimate, numerical comparison, custom-node scan) must run while the raw graph is still live in memory or in its temporary. Auditability is delivered as **recorded metrics + a deterministic re-runnable script**, not as retained files.
6. **Atomic final write.** Serialize to a temporary name in the destination directory, validate, then `os.replace()` into the final path. On any failure: remove the partial file and leave any pre-existing final model untouched.
7. **No stale outputs.** Never leave `model_raw.onnx`, `*_tmp.onnx`, `*.onnx.bak`, orphaned `*.data`/external-data shards, or debug dumps behind. If external data is used, the final model's external-data files are part of the single final artifact and must be self-consistent.
8. **Reproducibility replaces retention.** The export + rewrite pipeline must be deterministic: same code, weights, versions, and seeds regenerate the identical raw graph on demand, so the deleted raw model is never needed for audit.
9. **Idempotence.** Re-running on an already-optimized model either performs a clean no-op or exits with a clear message; it must never double-apply a rewrite or emit a corrupt artifact.
10. **Report, don't store.** Emit counts, histograms, diffs, and error metrics in the run report/log.

---

## 5. Required Inputs (determine or assume explicitly)

PyTorch version; exporter path (TorchScript vs Dynamo/`dynamo=True`); onnx package + IR version; target runtime version/revision; target opset; target hardware; intended Execution Provider (as a *source-graph compatibility constraint*); model dtype; input/output dtypes; static vs dynamic dims; representative shapes; production shape distribution; batch-size distribution; numerical tolerance; whether approximate math is allowed; whether weights/constants are immutable; whether caches/recurrent states are graph inputs/outputs; whether checkpoint representation may change; whether contrib/provider/custom ops are permitted (+ domain, opset, library, deployment path); relative priority of compute vs memory vs export stability vs graph size.

If information is missing, **do not stall**. State assumptions and deliver:

1. a provider-neutral pre-export optimization;
2. source-level target-specific alternatives where they materially differ;
3. a controlled graph-surgery alternative for each exporter limitation;
4. an explicit list of choices that require later deployment measurement â€” without performing runtime tuning here.

---

## 6. No-Skill-Lost Rule

Every candidate optimization in Â§9 and the checklist in Â§13 must be evaluated and labeled:

`APPLIED` Â· `ALREADY OPTIMAL` Â· `NOT APPLICABLE` Â· `REJECTED â€” POST-EXPORT JUSTIFICATION NOT MET` Â· `REJECTED â€” EXPORTER LIMITATION` Â· `REJECTED â€” PROVIDER/RUNTIME LIMITATION` Â· `REJECTED â€” NUMERICAL DIFFERENCE` Â· `REJECTED â€” DYNAMIC-SHAPE REQUIREMENT` Â· `REJECTED â€” CHECKPOINT COMPATIBILITY` Â· `REJECTED â€” INCREASED MEMORY TRAFFIC` Â· `REJECTED â€” INCREASED FINAL-GRAPH COMPLEXITY`

Silent skipping is a failure. Each label needs a one-line concrete reason.

---

## 7. Priority Order

1. Successful, stable export.
2. Required numerical correctness and contract preservation.
3. Elimination of avoidable computation in `forward()`.
4. Elimination of avoidable memory traffic and copies (Cast/Transpose/Concat/Pad/Expand/Scatter/slice-copies).
5. Folding and precomputation of all invariant work into immutable state.
6. Fewer/smaller live intermediates.
7. Fewer dynamic-shape reads and index/mask constructions.
8. Exporter- and provider-friendly operator patterns (fusable, standard, well-supported).
9. Targeted post-export replacement of unavoidable decompositions with supported fused operators.
10. Fewer final nodes, initializers, provider boundaries, and graph-level copies.
11. Source and rewrite-script simplicity.

---

## 8. Procedure

### Phase 1 â€” Baseline (source-first)

1. Locate the exported module and the exact `forward()` path used by export; follow only reachable helpers/submodules.
2. Build a table of every tensor on that path: shape (static/symbolic), dtype, layout, materialized?, consumer count, lifetime.
3. Mark all invariant work currently executed per call.
4. Mark repeated shape reads, casts, transposes, mask/index construction, duplicate subexpressions, temporary constants, per-branch duplicated work.
5. Record the required I/O and state/cache contract verbatim.
6. If export is possible, export **into memory/temporary** and collect operator histogram, node/initializer counts, and bytes-moved estimate as the baseline. Do not persist it (Â§4).
7. If assets/deps/hardware are unavailable, static source analysis is sufficient to start â€” state assumptions.

### Phase 2 â€” Design the pre-export rewrite

For each candidate: name the emitted operation; classify it input-dependent / shape-dependent / fully invariant; move invariant work to `__init__` or the export-prep step; fold immutable weights, biases, scales, indices, masks, layout transforms; simplify the input-dependent remainder; preserve required dynamic dims, dtypes, outputs, cache semantics; prefer source patterns with known-stable standard lowering; and record any fusion the exporter cannot produce, together with the exact decomposed subgraph you expect to see.

### Phase 3 â€” Implement and validate the source change

1. Smallest complete rewrite that removes the targeted graph work (no pseudocode).
2. Syntax/import validation.
3. Focused numerical tests vs the original PyTorch implementation when deps/weights exist.
4. Cover min/typical/max shapes, odd/even dims, boundary sequence lengths, multiple batch sizes, cache-empty/populated.
5. Export the source-optimized module (in memory or temporary).
6. Verify export stability, lowering, shape behavior, and numerics **before** any surgery.
7. Report honestly what could not be executed.

### Phase 4 â€” Controlled surgery (only for gated items)

1. Match the exact subgraph via structural + semantic preconditions.
2. Refuse and exit non-zero when pattern, count, attributes, dtypes, ranks, shapes, or initializer values disagree.
3. Replace only that region with the chosen fused operator.
4. Rewire all consumers and graph outputs before deleting replaced nodes.
5. Delete only newly dead nodes/initializers; leave unrelated structure and metadata intact.
6. Update value_info, custom domains, opset imports deliberately.
7. Keep it idempotent; write the final model atomically per Â§4.6.
8. No quantization, simplifier, generic optimizer, or unrelated cleanup before or after.

### Phase 5 â€” Validate final graph, then clean up

1. ONNX checker + shape inference where supported; document expected checker gaps for custom ops.
2. Load the final model with the exact custom-op library and provider config when available.
3. Compare: original PyTorch â†’ source-optimized PyTorch â†’ raw ONNX (in-memory) â†’ final ONNX.
4. Cover the full shape/dtype/cache matrix from Phase 3.
5. Exact equality for integer/bool outputs; declared tolerances for floats.
6. Confirm the decomposition is gone, the fused op appears exactly as intended, and no unrelated region changed (histogram diff + initializer diff).
7. Scan for unexpected ATen/custom/fallback nodes.
8. Record all metrics, then **delete every temporary** (Â§4.4) and confirm in the report that only the final artifact remains.

---

## 9. Aggressive Optimization Skills

### 9.1 Precompute everything invariant

Store as parameters, registered buffers, or export-time constants: transformed/transposed/reshaped/concatenated/packed weights; fused biases and scales; reciprocals; normalization constants; lookup tables; fixed indices, masks, causal masks, position IDs; rotary inverse frequencies and sin/cos tables; fixed permutations, reshape targets, split sizes, ranges, sign patterns, scalar/zero/one tensors; dtype-converted constants.

Never do these per call: `torch.arange` over fixed ranges; repeated `torch.tensor(...)`; repeated `.to()/.float()/.half()`; repeated weight transpose/reshape; repeated mask construction; repeated concatenation of immutable tensors; repeated scale/reciprocal/sqrt/log/exp of constants.

**Exception â€” runtime-controllable parameters:** never fold a value that must stay runtime-controllable. When the export includes a control submodule (e.g. repetition/frequency penalty, sampling), keep its control parameters â€” including `penalty_range` â€” as graph inputs even if they look constant at export time. If that submodule is not exported, the parameter is not required.

### 9.2 Fold weights, biases, scales

Evaluate: Add â†’ bias of Linear/GEMM/Conv; Mul/Div â†’ weights; normalization scale â†’ adjacent weight; constant affine folds; BatchNorm â†’ Conv; input scaling â†’ first weight; output scaling â†’ next weight; composing consecutive linear maps; concatenating projections that share an input (Q/K/V, gate/up); provider-friendly immutable weight packing; deleting identity weights, zero biases, dead/duplicate params; deduplicating identical initializers.

Account for broadcast axis, accumulation dtype, overflow/underflow, exporter lowering, kernel compatibility, tolerance. Prefer one larger GEMM over several small ones when it removes duplicated work **and** does not create a harmful intermediate.

### 9.3 Fuse operators and whole processes

Targets: MatMul+Add; GEMM+activation; Conv+Add/Mul/BN; bias+activation; bias+GELU; residual Add+LayerNorm/RMSNorm; normalization decompositions; attention subgraphs; rotary subgraphs; Q/K/V paths; scale+mask+Softmax; reshape/transpose chains around attention; repeated pointwise chains; adjacent affines; consecutive reductions; repeated broadcasts.

Express fusion in source when the exporter preserves it; otherwise use Â§3.2. Count a fusion as applied only when the redundant work is absent from the final graph, the replacement op and its weight transformations are present, and numerics pass. State whether it was achieved in source or by surgery.

### 9.4 Eliminate operators and processes

Delete/fold: identities; no-op and same-dtype casts; no-op reshapes/transposes; redundant `contiguous`/`clone`/`expand`/broadcast/squeeze/unsqueeze; adjacent inverse transposes/reshapes; repeated shape reads; repeated index/mask construction; duplicate subexpressions; Ã—1, +0, âˆ’0, Ã·1, double negation; redundant comparisons; unused outputs; dead branches/params/cache updates; full-tensor slices; single-input concats; split-then-reconstruct; cancelling view chains; identity permute chains; constant conditional branches.

Do constant propagation and dead-branch removal in module construction after every major rewrite. Never defer either to an ONNX cleanup pass.

### 9.5 Process together, split late

Concatenate/stack compatible inputs or weights, run one larger op, split afterwards; reuse shared intermediates; never repeat normalization/casting/masking/position processing per branch. Typical wins: fused Q/K/V, fused gate/up, single scale application, single position-embedding application, shared cache preprocessing, shared activation preprocessing.

Verify the merge does not break a documented lowering/fusion pattern, blow up peak memory, create unsupported dynamic splits, reduce parallelism, produce a target-incompatible op, or turn a static dimension dynamic.

### 9.6 Minimize elementwise / position-wise ops

Reduce Add, Sub, Mul, Div, Neg, Pow, Where, comparisons, Clamp, repeated activations, position-wise masks/scales, bitwise ops, bool conversions, per-position index math.

Techniques: algebraic simplification; constant folding; reassociation; scale/bias folding; activation fusion; one fused op instead of many pointwise ops; compute shared masks once; apply an op once before a split instead of per branch; prefer formulations that export directly.

Do not reassociate floating point beyond tolerance. Approximate math is forbidden unless explicitly authorized.

### 9.7 Minimize bitwise ops

Avoid unless clearly superior on the target provider. Fold/replace bit masks, shifts, parity, boolâ†”int conversions, packing/unpacking, repeated logical combinations; precompute constant bit patterns. Never adopt a bitwise form just because it is fewer source statements â€” verify exporter and provider support.

### 9.8 Minimize shape queries and dynamic shape math

Treat every dynamic `.shape` / `.size()` / shape-derived expression as a potential `Shape`â†’`Gather`â†’`Unsqueeze`â†’`Concat` subgraph.

Classify each dimension: compile-time constant Â· config constant Â· symbolically dynamic Â· runtime dynamic but reused Â· runtime dynamic used once.

Then: substitute constants; read each unknown dim exactly once and reuse it; reuse whole shape tensors; never rebuild the same shape tensor; no Python branching on tensor-derived shapes; avoid dynamic `arange`/masks/reshape targets when deployment shapes are static; specialize where production allows; minimize symbolic dims; keep shape arithmetic in one integer dtype; delete shape casts; never round-trip shape values between tensors and Python; never use `.item()` on the exported path.

Prefer static shapes when they materially improve the graph and the contract permits. For several shape profiles, prefer separately specialized module instances over one over-dynamic `forward()`.

### 9.9 Minimize dimension and layout changes

Reduce `transpose`, `permute`, `reshape`, `view`, `flatten`, `unflatten`, `squeeze`, `unsqueeze`, `expand`, `repeat`, layout conversions, contiguous materializations.

Rules: pick one canonical internal layout and stay in it; compose adjacent permutations; cancel inverse pairs; push transposes into constant weights; pre-transpose immutable weights; replace runtime data transpose with a weight-layout change when algebraically valid; avoid transposeâ†’contiguousâ†’reshape; collapse reshape chains; avoid flatten/unflatten and squeeze/unsqueeze pairs; avoid broadcasting via physical replication; prefer views only when they lower as metadata-only; verify whether a nominal view becomes a materializing op; respect layouts required by documented op lowering.

Never move a transpose merely to cut node count if it grows the tensor at the transpose point or breaks a documented fusion.

### 9.10 Indexing: Gather / Slice / Split / Scatter

**Gather / index selection.** Audit `Gather`, `GatherElements`, `GatherND`, `index_select`, advanced indexing, embedding lookup, boolean indexing. Prefer int32 indices only when: the exported op accepts int32 **and** the provider supports int32 for that op/opset **and** values cannot exceed int32 range **and** the exporter inserts no compensating cast **and** it actually reduces traffic or simplifies the dtype path. Otherwise use the provider-required dtype. Never add a cast inside `forward()` just to satisfy this preference. Precompute constant indices as buffers.

**Slice.** Audit starts/ends/axes/steps, negative steps, dynamic bounds, slice-to-end. Fold static slice controls into constants (int64 where required/favored). Delete whole-tensor slices. Merge compatible adjacent slices. Avoid dynamic bound computation for known dims.

**Split vs Slice.** Prefer one `Split` when the outputs partition one tensor along one axis, boundaries are known, `Split` is provider-supported, it removes repeated shape/boundary math, and it does not block a larger fusion. Not unconditional â€” keep Slice when its lowering is simpler/better supported. Use constant int64 split sizes when the opset requires.

**Scatter.** Treat `ScatterND`/`scatter_`/index_put as a full-tensor copy; prefer contract-legal alternatives (fixed-window Slice+Concat, preallocated buffer with static offsets, or moving the update outside the graph) and precompute constant index tensors.

### 9.11 `rotate_half()` and rotary embeddings

Compare at least: (1) split+neg+concat; (2) reshape+flip+sign/scale; (3) slice-based reversal; (4) an exporter-supported standard rotary/attention path; (5) precomputed permutation/sign application.

Choose the flip-based form only if it exports cleanly, creates no unsupported negative-step slicing, lowers entirely to target-compatible standard ops, reduces captured ops/nodes/traffic, and preserves the rotary convention.

Precompute sin/cos tables, inverse frequencies, position IDs, sign vectors, permutation indices, reshape metadata. Apply rotary jointly to compatible tensors before splitting when legal. No per-call sin/cos generation, no per-call position-dependent shape manipulation.

### 9.12 Minimize dtype casts

Build a full dtype-flow map. Remove same-dtype casts, cast chains, Aâ†’Bâ†’A round trips, repeated casts of one tensor, per-branch duplicated casts, unnecessary integer-width and bool conversions, runtime casts of constants.

Apply each needed cast once, at the point best for fusion and bandwidth, before a split when several consumers need the same dtype, and at init/export time for immutable tensors.

Distinguish storage dtype, I/O contract dtype, compute dtype, accumulation dtype, index dtype, shape dtype. Keep constants and buffers stored in their final runtime dtype. **Documented exception:** a static buffer sliced in `forward()` and cast back to the required dtype is acceptable when the storage dtype is deliberately narrower. Never force low precision through numerically sensitive ops without authorization.

### 9.13 Temporaries, reuse, zero-copy views

Precompute immutable buffers; reuse common intermediates; prefer safe metadata-only/zero-copy views; avoid temporary tensors, physical expansion, replication, unnecessary `clone`/`contiguous`/`repeat`, temporary constants, and mutation patterns that break SSA export or provider fusion.

ONNX is SSA: a PyTorch in-place op does not guarantee exported buffer reuse. Do not add mutation to "hint" preallocation. Output binding, allocator control, and device-buffer reuse are out of scope.

### 9.14 Minimize source-visible transfers and synchronization

Remove or hoist `.cpu()`, `.cuda()`, runtime `.to(device)`, `.item()`, tensor-value-driven Python control flow, cross-layout/cross-dtype copies, host-constant construction in `forward()`, state movement between devices, repeated dtype/device conversion of caches and masks.

Register constants and indices in their final dtype/device semantics before export. Keep inputs, outputs, and recurrent state on one consistent dtype/layout contract.

When an op is known to lower to something unsupported, resolve in order: fold/eliminate â†’ replace with a supported op â†’ hoist out of the repeated region â†’ precompute â†’ fuse into a neighbor â†’ change dtype/rank within contract â†’ controlled surgery to a supported standard/contrib/provider op â†’ custom op only if nothing else satisfies the contract and the library ships with deployment.

No provider-partition tuning. No placement or speed claims without evidence from the exact target environment.

### 9.15 Minimize memory and bandwidth (explicit pass)

For every intermediate: shape, dtype, lifetime, consumer count, materialized?, contiguous?, cheaply recomputable?

Optimize for: fewer large intermediates; shorter lifetimes; fewer full-tensor read/write passes; fewer materializing broadcasts; fewer transpose materializations; fewer concat copies; no duplicate caches; process before expansion; process before splitting; reduce before expensive layout change; fuse pointwise work into compute-heavy ops; narrower storage where authorized.

Do not cut arithmetic if it creates a substantially larger intermediate. Prefer the lower total-cost graph per Â§2; measure only when representative inputs and hardware already exist.

### 9.16 Minimize computation

CSE; strength reduction; invariant code motion; constant propagation; DCE; unrolling of small fixed structures; reduction fusion; cheaper equivalent formulations; combining repeated matrix ops; sharing normalized/transformed tensors; no repeated index/mask generation; no repeated scale/reciprocal computation; compute only requested outputs; drop unused auxiliary outputs.

Prioritize MatMul/GEMM, Conv, Softmax, normalization, transcendentals, big reductions, big Gather/Scatter, full-tensor layout changes â€” then bandwidth-bound elementwise chains.

### 9.17 ONNX-friendly coding

Avoid or rewrite: unsupported/data-dependent Python control flow; `.item()` on the graph path; Python lists built from tensor data; non-exportable custom autograd functions; unsupported complex indexing; unsupported in-place mutation; unnecessary custom ops; version-fragile exporter hacks; constructs that decompose into large primitive subgraphs when a supported fused op exists.

Prefer the modern exporter supported by the project's PyTorch version; use exporter diagnostics to understand lowering. Use static initializers (not graph inputs) for immutable weights. Keep `do_constant_folding` consistent between compared exports and do not rely on it to perform requested optimizations. Choose the opset by exporter support, provider support, required op availability, semantics, and deployment constraints â€” **not** by recency.

### 9.18 Dynamic-shape policy

Dynamic shapes are not automatically desirable. For each dynamic dim, prove it is required.

If not required: make it static; fold shape logic; precompute masks/indices; expose constants to capture; shrink shape/index subgraphs; enable more fusion.

If required: minimize symbolic dims; reuse symbolic values; avoid data-dependent dims; avoid unnecessary dynamic reshape targets; group deployment shapes into a few profiles; document when separate pre-export instances per profile are needed; use constructs that lower to well-supported dynamic-shape ops.

Never remove required dynamic behavior without documenting the contract change explicitly.

### 9.19 Attention and KV-cache

Evaluate: a single fused attention representation (standard SDPA-style export path, or a runtime-supported fused attention/MHA/GQA contrib op) versus the explicit matmul-scale-mask-softmax-matmul chain; folding the softmax scale into a projection weight or into one multiply on the smaller tensor; head reshape/transpose placement so QKV needs at most one layout pass; avoiding per-head loops (no unrolled per-head subgraphs); GQA/MQA handled by broadcast-friendly shapes instead of `repeat`/`Tile`; mask applied as a precomputed additive float mask rather than `Where` on a bool mask rebuilt per call.

**Cache strategy â€” choose deliberately and justify:** `Concat`-grow (copies the whole cache every step) vs preallocated max-length buffer with static-offset write vs fixed-window ring buffer. State the per-step bytes copied for the chosen scheme and keep cache dtype/layout identical between the input and output cache tensors so no cast/transpose sits on the state path.

### 9.20 Normalization canonicalization

Prefer a source form that lowers to one normalization operator (LayerNormalization at supporting opsets, or the runtime's fused RMS/Simplified-LayerNorm) instead of a mean/sub/pow/mean/add/sqrt/div/mul chain. Fold the affine weight/bias into the norm op or the adjacent GEMM where legal. Fuse residual-add into the normalization when a supported fused form exists. Keep epsilon and accumulation dtype semantics identical; never silently change the reduction dtype of a numerically sensitive norm.

### 9.21 Masking, Softmax, and reductions

Precompute causal/padding masks once (additive float, final dtype, correct broadcast rank); do not rebuild per call or per layer. Prefer one broadcastable mask over per-head materialized masks. Use `LogSoftmax` instead of `Log(Softmax)`; keep Softmax on its canonical axis so no transpose is needed around it; avoid `-inf` patterns that change NaN behavior versus the reference. Merge consecutive reductions over compatible axes; prefer one reduction over reduce-then-reduce; keep `keepdims` choices that avoid follow-up reshapes.

### 9.22 Control-flow elimination (If / Loop / Scan)

Resolve export-time-known conditions in Python so no `If` node appears. Unroll small fixed-trip loops when it removes subgraph overhead without exploding graph size; keep a loop only when trip count is genuinely dynamic and the op is well supported. Never keep a branch whose predicate is a config constant. Remove dead cache/state branches. Prefer separately exported specialized modules over a single graph with mode-selecting `If` nodes.

### 9.23 Embedding, LM head, and vocabulary paths

Prefer `Gather` for embedding lookup over one-hot matmul; keep tied embedding/LM-head weights as a single deduplicated initializer (transpose-folded, not stored twice); avoid runtime transpose of the LM head; consider computing logits only for the required positions (e.g. last token) instead of the whole sequence when the contract allows; keep index dtype consistent to avoid casts.

### 9.24 Rank and batch canonicalization

Fold leading batch dims into the M dimension so a batched 3-D MatMul becomes one 2-D GEMM when semantics allow and the reshape is metadata-only; keep a single canonical rank across a block so no squeeze/unsqueeze pairs appear; rely on implicit ONNX broadcasting instead of explicit `Expand`/`Tile`; avoid rank changes purely for readability.

### 9.25 Initializer and model-size hygiene

Deduplicate identical initializers; delete dead initializers; prefer initializers over per-call `Constant` nodes for large tensors and inline small scalars; store each constant exactly once in its final dtype and layout; avoid keeping both pre- and post-fold weight copies; keep external-data usage deliberate and self-consistent in the final artifact; ensure no temporary external-data shards survive (Â§4.7).

### 9.26 Provider-friendly layout and shape hygiene (source-level only)

When the target EP is known, respect its documented preferences as a *source* constraint: channels-last/NHWC for conv-heavy NPU/QNN-style targets, alignment-friendly head dims and hidden sizes, avoidance of ops with known unsupported dtypes/ranks, and shapes that keep the whole model in one partition. This is graph shaping before export â€” not post-export partition tuning.

### 9.27 Export granularity and multi-profile strategy

Consider exporting a submodule boundary that keeps runtime-controllable inputs (e.g. the penalty/sampling stage with `penalty_range`) as graph inputs, isolates a decomposition that needs surgery, or lets each shape profile be specialized. Document each exported unit's I/O contract and confirm the composition preserves end-to-end behavior.

### 9.28 Determinism and reproducibility

Fix seeds for validation inputs; make weight folding numerically deterministic and order-independent where possible; make node/tensor naming deterministic and collision-free; make the pipeline re-runnable to the identical raw graph (required by Â§4.8); make the rewrite idempotent.

### 9.29 Graph metrics instrumentation

Implement a small in-process metrics helper that, for any `ModelProto`, returns: operator histogram; node count; initializer count and total bytes; Cast / shape-op / layout-op / indexing-op / elementwise-op counts; count of materializing copy ops (Concat, Pad, Tile, Expand, Transpose, ScatterND, non-metadata Reshape); estimated bytes moved for representative shapes; graph I/O names, dtypes, and dynamic dims; opset imports and custom domains; any ATen/fallback/custom node. Run it on the raw and final graphs **before cleanup** and report the diff.

### 9.30 Enforce the surgery boundary

Before finishing, audit every command and code change: everything the exporter can express lives pre-export; the raw graph is treated as an ephemeral in-process baseline and the final model is written to its own path; only the explicit targeted rewrite runs â€” no quantizer, simplifier, generic optimizer, runtime transformer, or offline optimizer; no session/allocator/I-O-binding/provider/thread policy is delivered as an optimization; the rewrite touched only the intended subgraph and metadata; every inserted op's domain, opset, runtime/provider support, library requirement, and fallback is documented; unrelated runtime-only ideas are listed as out of scope; **all temporaries are gone**.

---

## 10. Rewrite Rules

**When modifying `forward()`:** give complete runnable replacements; preserve the required signature; move invariant initialization into `__init__`/an export-prep method; register immutable tensors correctly as parameters or buffers; no recomputation; reuse derived dims and intermediates; comment only non-obvious graph-level decisions; no abstraction that hides export behavior; favor graph transparency over style; never leave pseudocode.

**When changing module state or weight representation:** provide a migration/conversion function; state whether the original `state_dict` still loads; explain checkpoint compatibility; never change parameter semantics silently.

**When rewriting the ONNX graph:**

- complete, runnable, deterministic script or function;
- structured ONNX APIs only â€” no byte-level or text substitution;
- match semantics and topology, never autogenerated node names alone;
- require the exact expected match count; on zero/partial/ambiguous matches, write nothing and exit clearly;
- verify op types, domains, versions, attributes, dtypes, ranks, shapes, and initializer contents before editing;
- preserve graph input/output names and order unless a contract change is authorized;
- generate collision-free node/tensor/initializer names;
- keep topological order and valid producer/consumer relations;
- update opset imports, custom domains, value_info, metadata deliberately;
- be idempotent, or detect an already-rewritten model and exit clearly;
- **hold the raw and intermediate models in memory where possible; create any needed temporary automatically with no user-facing path and delete it unconditionally on success or failure (Â§4)**;
- compute all comparison metrics before cleanup;
- write the final model atomically only after every structural check passes;
- emit a concise report of matched, inserted, rewired, transformed, and deleted objects, plus confirmation that no temporary artifact remains.

---

## 11. Validation Requirements

**Source:** parses/compiles; module constructs; optimized `forward()` runs on representative inputs (when deps/weights exist); required signatures, shapes, dtypes, and cache updates preserved; all exporter-representable logic sits before `torch.onnx.export()`; no unsupported control flow, data extraction, or mutation added.

**Numerical:** compare original vs optimized PyTorch first. If surgery is used, raw-vs-final ONNX comparison is **mandatory** (in-process, before cleanup); otherwise it is recommended. Cover min/typical/max shapes, odd/even dims, boundary sequence lengths, several batch sizes, cache-empty/populated, all relevant dtypes, random and real inputs. Report max abs error, max rel error, mean abs error, cosine similarity where relevant, exact equality for int/bool outputs, and any NaN/Inf behavior change. Approximate rewrites require explicit authorization.

**Graph:** same exporter, opset, inputs, dynamic-axis policy, and constant-folding setting on both sides of any comparison; checker + shape inference on both graphs where supported; scan for unexpected custom/ATen/fallback nodes; diff operator histograms, initializers, graph interfaces, and the targeted node families; compare raw vs final outputs with runtime graph optimizations disabled when the runtime permits; load required custom-op libraries explicitly. Reproducibility (deterministic re-export) replaces artifact retention. No latency ablation or runtime tuning; placement/performance claims require target-environment evidence.

**Cleanup:** the final report must state that only the final model (plus its external data, if any) exists on disk, and that temporaries were removed on both the success and failure paths.

---

## 12. Required Deliverables

1. **Assumptions** â€” versions, exporter path, opset, shape policy, dtype policy, immutability, checkpoint constraints, tolerance, provider.
2. **Baseline audit** â€” expensive/redundant work in construction and `forward()`, its expected lowering, and every decomposition that is a surgery candidate with the exporter-limitation evidence.
3. **Optimization plan** â€” pre-export changes ranked by expected impact on Â§2 (Aâ€“E) and correctness risk; then each post-export rewrite separately with evidence, replacement op, runtime support, expected benefit, validation plan.
4. **Complete optimized code** â€” runnable module/helper/export-script changes, plus the deterministic rewrite script (including the ephemeral-artifact handling of Â§4) when surgery is justified.
5. **Patch / change map** â€” every source and graph change; for surgery: matched subgraph, replacement op, rewired edges, transformed initializers, deleted nodes, metadata updates.
6. **Before/after graph report** â€” for source, raw export, and final graph: invariant work moved out of `forward()`; matrix/conv/norm/reduction ops removed or combined; Cast count; shape-op count; layout-op count; indexing-op count; elementwise-op count; materializing-copy-op count; estimated bytes moved and peak live bytes; large temporaries removed/introduced; buffers/packed params added; node and initializer counts and bytes; inserted fused ops and removed decompositions; graph I/O and dynamic-shape contract changes (normally none); custom domains, opset imports, runtime library requirements.
7. **Numerical validation report** â€” measured comparisons across original PyTorch, optimized PyTorch, raw ONNX, final ONNX; state clearly when deps, weights, custom-op libraries, hardware, or inputs prevented execution.
8. **No-skill-lost checklist** â€” every item in Â§9 and Â§13 labeled per Â§6 with a concrete reason.
9. **Surgery justification and scope confirmation** â€” pre-export was attempted first; why the exporter could not express the graph; why the replacement is supported; how the rewrite is constrained.
10. **Artifact and cleanup confirmation** â€” the single final artifact; the list of temporaries that were created and deleted; the mechanism guaranteeing deletion on success and failure; confirmation that no raw/scratch path is exposed or configurable.

---

## 13. Mandatory Checklist (evaluate every item)

Precomputation of immutable state Â· reusable constants/buffers registered Â· common intermediates reused Â· zero-copy/metadata-only views where export-safe Â· minimum source-visible transfer/sync Â· minimum casts Â· minimum cached-buffer casts Â· minimum transpose Â· permute Â· reshape Â· view Â· squeeze Â· unsqueeze Â· flatten Â· unflatten Â· deep shape inference Â· minimum `.shape` Â· minimum `.size()` Â· known dims reused Â· minimum computation Â· minimum temporaries Â· minimum peak memory Â· minimum bytes moved Â· minimum copy ops (Concat/Pad/Tile/Expand/Transpose/ScatterND) Â· minimum position-wise ops Â· minimum elementwise ops Â· minimum bitwise ops Â· fused weights Â· fused biases Â· fused scales Â· folded operators Â· folded processes Â· constant folding Â· CSE Â· DCE Â· process-together-then-split Â· int32 Gather/index-select where beneficial and supported Â· int64 Slice controls where required Â· Split instead of repeated Slice where better Â· ONNX-friendly constructs Â· stable export patterns Â· provider-compatible standard ops Â· flip-based `rotate_half()` candidate evaluated Â· rotary precomputation and source-level fusion Â· attention fusion evaluated Â· KV-cache update scheme justified with per-step copy cost Â· normalization canonicalized to a single op Â· residual+norm fusion evaluated Â· precomputed additive masks Â· LogSoftmax vs Log(Softmax) Â· reduction merging Â· `If`/`Loop` elimination or justification Â· embedding/LM-head weight tying and dedup Â· logits computed only where required Â· batch/rank canonicalization and implicit broadcasting Â· initializer dedup and dead-initializer removal Â· deliberate external-data handling Â· provider-friendly layout at source level Â· export granularity and shape-profile strategy Â· runtime-controllable params (`penalty_range`) kept as graph inputs Â· graph metrics captured before cleanup Â· explicit decision on whether each remaining decomposition needs surgery Â· strict pattern/attribute/dtype/rank/shape/initializer preconditions Â· fail-closed and reproducible rewrite Â· idempotent rewrite Â· single persisted final artifact Â· **all temporary/raw files auto-deleted on success and failure** Â· **no configurable or advertised scratch path** Â· dead-node/initializer removal limited to the rewritten region Â· graph interface, value_info, opset, custom-domain validation Â· custom-op library and provider support documented Â· no post-export quantization or mixed-precision pass Â· no blanket simplifier/optimizer/cleanup pass Â· no runtime graph-transformer or offline-optimization step Â· no I/O-binding, allocator, thread, or session tuning Â· atomic final write with no partial artifact left behind.

---

## 14. Decision Standard

Apply an optimization only when at least one of these is demonstrated:

fewer ops executed by `forward()` Â· invariant work moved into construction/export prep Â· fewer or smaller temporaries Â· fewer bytes moved or fewer full-tensor copies Â· fewer casts or layout conversions Â· fewer shape reads and dynamic-shape computations Â· fewer repeated indexing/masking/elementwise processes Â· fused immutable weights/biases/scales Â· compatible branches combined before splitting Â· simpler, more stable exporter lowering Â· fewer nodes in the exported graph achieved through source optimization Â· an unavoidable decomposition replaced by a supported fused operator Â· fewer final nodes, provider boundaries, transfers, or large intermediates after targeted surgery Â· lower measured PyTorch cost on representative inputs without changing the contract.

Reject a source rewrite that only makes the code look nicer while the graph is equal or worse. Reject surgery whose justification, preconditions, runtime support, numerical equivalence, or graph-level benefit is not demonstrated. Never make latency or provider-placement claims without target-runtime evidence.

**Tie-break order:**

1. numerical equivalence and contract preservation in PyTorch;
2. numerical equivalence and contract preservation in the final ONNX graph;
3. successful exporter capture and clean validation of the exported graph;
4. exact target-runtime operator support (and placement evidence when available);
5. lower total cost per Â§2 (bytes moved, then FLOPs, then peak memory);
6. the final graph's structure and its reproducible surgery report;
7. source-level tensor-size, op-count, and dtype/layout analysis;
8. PyTorch source appearance (lowest priority).

Be aggressive. Be evidence-driven. Leave one file behind.
