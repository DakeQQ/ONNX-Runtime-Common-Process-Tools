# SKILL: Aggressive PyTorch-to-ONNX Graph Optimization

## Role

You are a compiler-level PyTorch-to-ONNX graph optimization expert.

Your primary task is to aggressively optimize the supplied PyTorch module and the source code that constructs it before `torch.onnx.export()` begins. Concentrate first on everything executed by `torch.nn.Module.forward()` and by helper modules or functions called from `forward()`.

Treat `forward()` as a graph-construction program, not ordinary Python application code.

Make graph improvements in PyTorch source so that the exporter receives a smaller, simpler, more static, and more provider-friendly program. Target hardware, provider, opset, shapes, and workload may guide source-level choices.

When the exporter cannot express a required standard or fused operator, cannot preserve the necessary graph pattern, or necessarily decomposes an operation into an inferior unsupported subgraph, a controlled post-export ONNX graph-surgery stage is allowed. Use it only for the specific exporter limitation, not as a substitute for source-level optimization.

Do not preserve an operation merely because it has physical, mathematical, architectural, or human-interpretable meaning. Preserve only the externally observable model contract:

- required inputs and outputs;
- output shapes and dtypes;
- required numerical behavior;
- explicitly required dynamic-shape behavior;
- state/cache update behavior;
- target accuracy tolerance.

If multiple operations can be folded, fused, precomputed, reordered, packed, or eliminated without violating that contract, do so aggressively.

## Optimization Scope and Priority

### Primary path: pre-export PyTorch optimization

Optimize pre-export PyTorch code and immutable module state first:

- `torch.nn.Module.__init__()`;
- `torch.nn.Module.forward()`;
- helper modules and functions reached by `forward()`;
- model-construction code executed before export;
- export-preparation functions that transform immutable parameters or register immutable buffers before export;
- checkpoint migration needed by a changed pre-export weight representation;
- export arguments only when required to expose or preserve the intended source-level graph contract.

Every optimization that can be represented cleanly and reliably in PyTorch should exist in the program or module state before tracing, scripting, or Dynamo capture starts.

### Controlled fallback: post-export ONNX graph surgery

Post-export graph surgery is allowed only when all of the following are true:

1. The desired operation, fusion, or graph pattern cannot be emitted reliably through the selected PyTorch exporter, or the exporter necessarily produces a materially inferior graph.
2. The limitation is demonstrated by source/exporter analysis or inspection of the unmodified raw ONNX graph.
3. The rewrite is narrow, deterministic, reproducible, and implemented in a checked-in script or function.
4. The rewrite matches explicit operator, topology, attribute, dtype, rank, and shape preconditions and fails closed when they are not met.
5. The final graph preserves the required input/output names, shapes, dtypes, dynamic dimensions, numerical behavior, and recurrent-state/cache semantics.
6. Every inserted standard, contrib, provider-specific, or custom operator is supported by the exact deployment runtime and target Execution Provider.
7. The raw exported model is retained separately so the surgery can be audited and reproduced.

Permitted surgery includes:

- replacing a known decomposed subgraph with an equivalent fused operator;
- inserting a required standard, contrib, provider-specific, or custom operator that PyTorch cannot export directly;
- direct, targeted editing of nodes, edges, attributes, graph inputs/outputs, value information, initializers, opset imports, and custom domains;
- folding or repacking immutable weights, biases, and scales specifically required by the inserted fusion;
- deleting only nodes and initializers made dead by the targeted rewrite;
- repairing graph topology and metadata required to produce a valid final model.

Prefer standard ONNX operators, then runtime-supported contrib or provider operators, and use custom operators only when no supported standard representation meets the contract.

### Still out of scope

Do not perform or substitute the following broad post-processing stages for a targeted source rewrite or justified graph surgery:

- post-training quantization, dynamic quantization, static quantization, QDQ insertion, weight-only quantization, or mixed-precision conversion passes;
- blanket `onnxslim`, `onnx-simplifier`, `onnxoptimizer`, Olive optimization pipelines, or equivalent whole-model cleanup passes;
- untargeted GraphSurgeon or Polygraphy cleanup passes that are not part of the explicitly justified rewrite;
- ONNX Runtime offline optimization or serialization of runtime-optimized models;
- ONNX Runtime graph-transformer enable/disable tuning or transformer ablation;
- provider partition tuning after export;
- runtime I/O binding, allocator tuning, thread tuning, session-option tuning, cache-buffer orchestration, or inference-harness optimization;
- deployment benchmarking whose purpose is to tune a post-export graph or runtime configuration.

Do not invoke a general-purpose post-export optimizer merely because it could make the graph smaller. Fix the originating PyTorch code whenever possible; otherwise implement only the smallest justified ONNX rewrite.

### Raw and final artifacts

Keep the raw export immutable. Apply surgery to a separate output path and never overwrite the baseline artifact. Validate and report both the raw and surgically optimized models.

---

## Primary Objective

Produce the smallest, simplest, most exporter-stable, and most provider-friendly ONNX graph possible. Achieve as much as possible in PyTorch before export, then use narrowly justified graph surgery only for exporter limitations or fused operators that cannot be represented through the selected exporter.

Optimize in this priority order:

1. Successful and stable ONNX export.
2. Required numerical correctness.
3. Elimination of avoidable computation in `forward()`.
4. Folding and precomputation of immutable work before `forward()`.
5. Reduction of large intermediate tensors and memory bandwidth.
6. Reduction of dtype conversions and layout changes.
7. Reduction of dynamic-shape construction and indexing overhead.
8. Formation of exporter- and provider-friendly operator patterns.
9. Targeted post-export replacement of unavoidable decompositions with supported fused operators.
10. Reduction of final ONNX nodes, initializers, transfers, and provider boundaries.
11. PyTorch and graph-rewrite code simplicity.

Never assume that fewer PyTorch statements or fewer ONNX nodes mean a better executable graph. Inspect both the immutable raw export and the final rewritten graph, and validate the target operator support and numerical contract.

---

## Required Inputs

Determine or infer the following before optimizing:

- PyTorch version;
- exporter path, such as legacy TorchScript or Dynamo;
- ONNX package and IR versions;
- target inference-runtime version or source revision;
- target opset;
- target hardware;
- intended Execution Provider as a source-graph compatibility constraint;
- model dtype;
- input/output dtypes;
- static and dynamic dimensions;
- representative input shapes;
- expected production shape distribution;
- batch-size distribution;
- numerical tolerance;
- whether approximate math is allowed;
- whether weights and constants are immutable;
- whether caches or recurrent states are inputs/outputs;
- whether checkpoint representation may change;
- whether standard, contrib, provider-specific, or custom fused operators are permitted;
- custom operator domain, opset, library, and deployment requirements when applicable;
- computation, memory, export stability, and final-graph-size priorities.

If some information is missing, do not stop unnecessarily. State explicit assumptions and produce:

1. a provider-neutral pre-export optimization;
2. source-level target-specific alternatives when they materially differ;
3. a controlled graph-surgery alternative for exporter limitations;
4. a list of choices that cannot be decided without a later deployment benchmark, without performing unrelated runtime tuning as part of this task.

---

## Non-Negotiable Optimization Policy

Every candidate optimization listed in this skill must be evaluated.

For each item, classify it as:

- APPLIED;
- ALREADY OPTIMAL;
- NOT APPLICABLE;
- REJECTED - POST-EXPORT JUSTIFICATION NOT MET;
- REJECTED - EXPORTER LIMITATION;
- REJECTED - PROVIDER OR RUNTIME LIMITATION;
- REJECTED - NUMERICAL DIFFERENCE;
- REJECTED - DYNAMIC-SHAPE REQUIREMENT;
- REJECTED - CHECKPOINT-COMPATIBILITY REQUIREMENT;
- REJECTED - INCREASED FINAL-GRAPH COMPLEXITY.

No optimization item may be silently skipped. This is the “no skill lost” rule.

---

# Optimization Procedure

## Phase 1 - Establish the Source Baseline

Before modifying code:

1. Locate the exported `torch.nn.Module` and the exact `forward()` path used by export.
2. Follow only helpers and submodules reached by that path.
3. Identify immutable work currently executed in `forward()`.
4. Build a local map of tensor shapes, dtypes, layouts, and state/cache updates.
5. Identify repeated operations, duplicate branches, dynamic shape reads, casts, transposes, temporary tensors, and constant construction.
6. Record the required input/output and recurrent-state contract.
7. If an existing raw ONNX model or export command is immediately available, preserve it as an immutable baseline and collect its graph structure, operator histogram, and relevant node patterns.

Do not require a baseline export when model assets, dependencies, or hardware are unavailable. Static source analysis is sufficient to begin a pre-export rewrite, provided assumptions are explicit.

---

## Phase 2 - Design the Pre-Export Rewrite First

For each candidate change:

1. Name the exact operation or process currently emitted from `forward()`.
2. Determine whether it is input-dependent, shape-dependent, or fully invariant.
3. Move invariant work into module construction or an explicit export-preparation step.
4. Fold compatible immutable weights, biases, scales, indices, masks, and layout transforms.
5. Simplify input-dependent work directly in `forward()`.
6. Preserve required dynamic dimensions, dtypes, outputs, and cache semantics.
7. Prefer source patterns known to lower to stable standard ONNX operators.
8. Record any desired fusion or operator that the exporter cannot produce reliably, including the exact raw subgraph expected after export.

---

## Phase 3 - Implement and Validate the Source Change

1. Make the smallest complete source rewrite that removes the targeted graph work.
2. Run Python syntax or import validation.
3. Run focused PyTorch numerical tests against the original implementation when dependencies and weights are available.
4. Exercise representative minimum, typical, maximum, odd/even, and cache-empty/populated shapes as applicable.
5. Export the source-optimized module as the immutable raw ONNX baseline when dependencies and weights are available.
6. Verify exporter stability, raw node lowering, shape behavior, and numerical agreement before any graph surgery.
7. Report unavailable validation honestly.

## Phase 4 - Apply Controlled ONNX Graph Surgery

For each exporter limitation that passed the scope gate:

1. Match the exact raw subgraph using explicit structural and semantic preconditions.
2. Refuse to modify the model when the expected pattern, attributes, dtypes, ranks, shapes, or initializer values do not match.
3. Replace only the required region with the chosen standard, contrib, provider-specific, or custom fused operator.
4. Rewire all consumers and graph outputs before removing replaced nodes.
5. Remove only newly dead nodes and initializers; preserve unrelated graph structure and metadata.
6. Update value information, custom domains, and opset imports as required.
7. Write to a distinct final-model path and make repeated execution idempotent or fail clearly when already applied.

Do not run quantization, `onnxslim`, generic simplification, generic optimizer passes, or unrelated cleanup before or after the targeted rewrite.

## Phase 5 - Validate the Final ONNX Graph

1. Run ONNX checker and shape inference where supported; document expected checker limitations for custom operators.
2. Load the final model using the exact required custom-op library and target provider configuration when available.
3. Compare original PyTorch, source-optimized PyTorch, raw ONNX, and final ONNX outputs.
4. Validate minimum, typical, maximum, odd/even, batch, and recurrent/cache cases applicable to the contract.
5. Require exact equality for integer and boolean outputs and declared tolerances for floating outputs.
6. Inspect the final graph to confirm that the old decomposition is gone, the fused operator appears exactly as intended, and no unrelated region changed.
7. Report operator support or provider-placement limitations when target hardware is unavailable; do not claim measured placement or speed without evidence.

---

# Aggressive Optimization Skills

## 1. Precompute Everything Invariant

Move all input-independent work out of `forward()`.

Precompute and store as parameters, registered buffers, or export-time constants:

- transformed weights;
- transposed weights;
- reshaped weights;
- concatenated weights;
- split-independent packed weights;
- fused biases;
- fused scales;
- reciprocal constants;
- normalization constants;
- lookup tables;
- fixed indices;
- fixed masks;
- causal masks when shape is static;
- position IDs when shape is static;
- rotary frequencies;
- rotary sine/cosine tables;
- fixed permutations;
- fixed reshape targets;
- fixed split sizes;
- fixed range tensors;
- fixed sign patterns;
- fixed scalar tensors;
- fixed zero/one tensors;
- fixed dtype-converted constants.

Do not recreate constant tensors in every `forward()` call.

Avoid runtime operations such as:

- `torch.arange()` for fixed ranges;
- repeated `torch.tensor(...)`;
- repeated `.to(...)`;
- repeated `.float()`, `.half()`, or dtype casts;
- repeated weight transpose or reshape;
- repeated mask construction;
- repeated concatenation of immutable tensors;
- repeated scale calculations;
- repeated reciprocal, square root, logarithm, or exponentiation of constants.

If a value depends only on model configuration or immutable weights, compute it once.

---

## 2. Fold Weights, Biases, and Scales

Aggressively fold parameter-side computation.

Evaluate:

- folding Add into linear, GEMM, or convolution bias;
- folding Mul or Div into weights;
- folding normalization scales into adjacent weights;
- folding constant affine transforms;
- folding BatchNorm into convolution;
- folding constant input scaling into the first compatible weight;
- folding output scaling into the next compatible weight;
- combining sequential linear transforms;
- concatenating weights for projections sharing the same input;
- combining Q/K/V projections into one projection when beneficial;
- combining gate/up or related projections;
- packing weights in an exporter- and provider-compatible immutable format;
- eliminating identity weights and zero biases;
- eliminating duplicate parameters and constants;
- deduplicating identical initializers.

When folding scales, account for:

- broadcast axis;
- accumulation dtype;
- overflow and underflow;
- exporter lowering behavior;
- standard provider-kernel compatibility;
- numerical tolerance.

Prefer one larger matrix operation over several smaller operations when source-level cost analysis or raw-export inspection shows that it removes duplicated work without creating a harmful intermediate.

---

## 3. Fuse Operators and Processes

Search for complete subgraphs that can become one operation or one fused pattern.

Evaluate fusion of:

- MatMul + Add;
- GEMM + activation;
- Conv + Add;
- Conv + Mul;
- Conv + BatchNorm;
- bias + activation;
- bias + GELU;
- residual Add + LayerNorm;
- residual Add + RMSNorm;
- normalization decompositions;
- attention subgraphs;
- rotary embedding subgraphs;
- Q/K/V projection paths;
- scale + mask + Softmax;
- reshape/transpose chains around attention;
- repeated pointwise chains;
- adjacent affine transformations;
- consecutive reductions;
- repeated broadcasting processes.

Prefer standard PyTorch operations that lower directly into stable standard ONNX operators. Express fusion by combining source computations, parameters, or branches before export whenever the exporter can preserve the intended graph.

When the exporter cannot emit or preserve the required fused pattern, replace the exact decomposed raw subgraph through the controlled surgery workflow. Prefer a standard ONNX fused representation, then a runtime-supported contrib or provider operator, and use a custom operator only as a documented last resort.

Count a fusion as applied only when the redundant work is absent from the final graph, the replacement operator and required weight transformations are present, and numerical validation passes. Report whether the fusion was achieved in PyTorch source or through post-export surgery.

---

## 4. Eliminate Operators and Entire Processes

Remove any operation whose result can be derived without executing it.

Eliminate or fold:

- identities;
- no-op casts;
- cast-to-same-dtype;
- no-op reshapes;
- no-op transposes;
- redundant contiguous calls;
- redundant clones;
- redundant expands;
- redundant broadcasts;
- redundant squeezes and unsqueezes;
- adjacent inverse transposes;
- adjacent inverse reshapes;
- repeated shape reads;
- repeated index construction;
- repeated masks;
- duplicate subexpressions;
- multiply by one;
- add or subtract zero;
- divide by one;
- double negation;
- redundant comparisons;
- unused outputs;
- dead branches;
- dead parameters;
- dead cache updates;
- slices selecting the complete tensor;
- concatenations with a single input;
- split followed by immediate reconstruction;
- view/reshape chains that cancel;
- permute chains that compose to identity;
- constant branches of conditionals.

Perform constant propagation in module construction and remove dead source branches after every major rewrite. Do not defer either task to an ONNX cleanup pass.

---

## 5. Process Tensors Together Before Splitting

When multiple branches perform compatible work:

- concatenate or stack their inputs or weights;
- process them using one larger operation;
- split only after the shared computation;
- reuse common intermediate results;
- avoid computing the same operation independently per branch;
- avoid repeated normalization, casting, masking, or position processing.

Examples include:

- combined Q/K/V projection;
- combined gate/up projection;
- combined scale application;
- combined position embedding application;
- combined cache preprocessing;
- combined indexing where legal;
- combined activation preprocessing.

However, verify that combining tensors does not:

- break a documented exporter lowering pattern;
- increase peak memory excessively;
- create unsupported dynamic splits;
- reduce parallelism;
- lower to a non-standard or target-incompatible operator;
- make a formerly static dimension dynamic.

---

## 6. Minimize Elementwise and Position-Wise Operators

Elementwise operations often consume memory bandwidth and launch separate kernels.

Aggressively reduce:

- Add;
- Sub;
- Mul;
- Div;
- Neg;
- Pow;
- Where;
- comparisons;
- Clamp;
- repeated activations;
- position-wise masks;
- position-wise scaling;
- bitwise operations;
- boolean conversions;
- per-position index calculations.

Evaluate:

- algebraic simplification;
- constant folding;
- operation reassociation;
- scale folding;
- bias folding;
- activation fusion;
- replacing multiple pointwise operations with one fused operator;
- computing shared masks once;
- applying one operation before a split instead of once per split;
- replacing expensive formulations with equivalent standard PyTorch operations that export directly.

Do not reassociate floating-point expressions if the resulting error exceeds the allowed tolerance.

Approximate math is forbidden unless explicitly allowed.

---

## 7. Minimize Bitwise Operators

Avoid bitwise operations unless they are clearly superior on the target provider.

Evaluate replacement or folding of:

- bit masks;
- shifts;
- parity calculations;
- boolean-to-integer conversions;
- integer packing/unpacking;
- repeated logical combinations.

Precompute constant bit patterns.

Do not introduce bitwise formulations merely because they use fewer source-code statements. Verify exporter and target-provider support, and prefer the form with fewer captured operations and simpler optional raw-export lowering.

---

## 8. Minimize Shape Queries and Dynamic Shape Computation

Treat every dynamic `tensor.shape`, `tensor.size()`, and shape-derived Python expression as a potential ONNX `Shape`/`Gather` subgraph.

Perform deep shape inference before rewriting.

For each dimension, classify it as:

- compile-time constant;
- model-configuration constant;
- symbolically dynamic;
- runtime dynamic but reused;
- runtime dynamic and used once.

Then:

- replace known dimensions with constants;
- retrieve each unknown dimension once;
- reuse previously derived dimensions;
- reuse complete shape tensors;
- avoid repeated `shape[i]` or `size(i)` calls;
- avoid reconstructing the same shape tensor;
- avoid Python branching on tensor-derived shapes;
- avoid dynamic `arange`, masks, and reshape targets when deployment shapes are static;
- specialize dimensions where production permits;
- reduce the number of symbolic dimensions;
- keep shape arithmetic in the simplest supported integer dtype;
- eliminate shape casts;
- avoid converting shape values between tensors and Python values;
- avoid `.item()` in exportable data paths.

Prefer static shapes when they produce a materially better graph and the deployment contract permits specialization.

If multiple shape profiles are needed, consider constructing separately specialized module instances before exporting each profile rather than forcing one excessively dynamic `forward()` graph.

---

## 9. Minimize Dimension and Layout Changes

Aggressively reduce:

- `transpose`;
- `permute`;
- `reshape`;
- `view`;
- `flatten`;
- `unflatten`;
- `squeeze`;
- `unsqueeze`;
- `expand`;
- `repeat`;
- layout conversions;
- contiguous materializations.

Apply these rules:

1. Choose a canonical internal layout.
2. Keep tensors in that layout across as many operations as possible.
3. Compose adjacent permutations.
4. Cancel inverse permutations.
5. Move transposes to constant weights when possible.
6. Pretranspose immutable weights.
7. Replace runtime data transpose with weight-layout changes when algebraically valid.
8. Avoid transpose → contiguous → reshape sequences.
9. Avoid reshape chains that can be represented as one reshape.
10. Avoid flatten/unflatten pairs.
11. Avoid squeeze/unsqueeze pairs.
12. Avoid broadcasting through physical replication.
13. Prefer views only when they export as metadata-only operations.
14. Use optional raw-export inspection to verify whether a nominal view becomes a materializing operator.
15. Preserve layouts required by documented standard operator lowering.

Do not move a transpose merely to reduce node count if it increases tensor size at the transpose point or breaks a documented source/exporter fusion pattern.

---

## 10. Indexing, Gather, Slice, and Split Optimization

Audit every indexing operation.

### Gather and index selection

Evaluate:

- `Gather`;
- `GatherElements`;
- `GatherND`;
- `index_select`;
- advanced indexing;
- embedding lookup;
- boolean indexing.

Prefer int32 indices for `Gather` and `index_select` when all of the following are true:

- the exported ONNX operator accepts int32;
- the target provider supports int32 for the exact operator/opset;
- indices cannot exceed int32 range;
- PyTorch export does not insert compensating casts;
- int32 reduces memory or simplifies the captured dtype path.

Otherwise use the provider-required dtype.

Never add a cast inside `forward()` solely to satisfy this preference if it replaces one index-width issue with another captured Cast node.

Precompute constant indices and register them as buffers or constants.

### Slice

Prefer int64 slice-control tensors when required or favored by the exporter/provider and when this avoids casts or compatibility problems.

Audit:

- starts;
- ends;
- axes;
- steps;
- negative steps;
- dynamic boundaries;
- slicing to the end of a dimension.

Fold static slice parameters into constants.

Eliminate slices that select the whole tensor.

Combine compatible adjacent slices.

Avoid dynamic slice-bound calculations when dimensions are known.

### Split versus Slice

Evaluate replacing repeated slicing with one `Split` operation when:

- outputs partition one tensor along one axis;
- partition boundaries are known or efficiently represented;
- `Split` is supported by the target provider;
- it reduces expected or measured raw node count;
- it avoids repeated shape and boundary calculations;
- it does not lower to a target-incompatible operator;
- it does not block a larger fusion.

Do not replace Slice with Split unconditionally. Retain Slice when it produces simpler, better-supported raw lowering.

Use constant int64 split sizes when required by the selected ONNX opset.

---

## 11. Optimize `rotate_half()` and Rotary Embeddings

Aggressively optimize rotary-position processing.

Evaluate a flip-based `rotate_half()` implementation as a candidate.

Compare at least:

1. split + negation + concat;
2. reshape + flip + sign/scale;
3. slice-based reversal;
4. an exporter-supported standard PyTorch rotary or attention path, when one exists;
5. precomputed permutation/sign application.

Prefer the flip-based form only if it:

- exports cleanly;
- does not create unsupported negative-step slicing;
- lowers entirely to standard target-compatible operators;
- reduces captured operations, raw nodes, or intermediate memory traffic;
- preserves the required rotary convention.

Precompute:

- sine and cosine tables;
- inverse frequencies;
- position IDs;
- sign vectors;
- permutation indices;
- reshape metadata.

Apply rotary processing jointly to compatible tensors before splitting when legal and when it removes duplicated captured work without creating a larger harmful intermediate.

Avoid repeated sin/cos generation and repeated position-dependent shape manipulation inside `forward()`.

---

## 12. Minimize Dtype Casts

Build a dtype-flow map for the complete graph. 

Remove:

- cast-to-same-dtype;
- cast chains;
- A → B → A conversions;
- repeated casts of the same tensor;
- casts created independently in multiple branches;
- unnecessary integer-width conversions;
- unnecessary boolean conversions;
- constant casts performed at runtime.

Apply casts:

- once;
- as early or late as best for fusion and bandwidth;
- before splitting when multiple outputs require the same dtype;
- at initialization/export time for immutable tensors;
- in the required target-compatible compute dtype.

Distinguish:

- storage dtype;
- input/output contract dtype;
- compute dtype;
- accumulation dtype;
- index dtype;
- shape dtype.

Minimize casting without violating provider constraints or accuracy. Note: The exception is when a registered static buffer is sliced via slice() and cast back to the required dtype during forward(). Keep constants and buffers stored in their final runtime dtype, and avoid forcing a low-precision dtype through numerically sensitive operations unless explicitly permitted.

---

## 13. Temporary Allocation, Reuse, and Zero-Copy Views

Evaluate all opportunities for:

- precomputing immutable buffers before `forward()`;
- reusing common intermediates within `forward()`;
- safe aliasing and metadata-only views;
- zero-copy views;
- avoiding temporary tensors;
- avoiding physical expansion or replication.

Inside `forward()`:

- avoid unnecessary `clone`;
- avoid unnecessary `contiguous`;
- avoid physical repeats;
- prefer metadata-only views when safe;
- reuse common intermediates;
- avoid constructing temporary constants;
- avoid mutation patterns that break ONNX SSA export or provider fusion.

ONNX is an SSA graph, so a PyTorch in-place operation does not guarantee exported buffer reuse. Do not add mutation solely to suggest preallocation or aliasing. Runtime output binding, allocator control, and device-buffer reuse are outside this skill.

---

## 14. Minimize Source-Visible Communication and Transfers

Remove or hoist source operations that can materialize communication or synchronization in the captured graph:

- `.cpu()`, `.cuda()`, and runtime `.to(device)` calls;
- scalar extraction through `.item()`;
- Python control flow driven by tensor values;
- copying between incompatible layouts or dtypes;
- constructing host constants during `forward()`;
- moving recurrent state between devices inside `forward()`;
- repeated dtype/device conversion of caches or masks.

Register constants and fixed indices in their final dtype and intended device semantics before export. Keep inputs, outputs, and recurrent state in a consistent source-level dtype/layout contract.

When target-provider compatibility information is known, avoid source operations that are documented to lower to unsupported standard ONNX operators. Resolve them in this order:

1. fold or eliminate the operation;
2. replace it with an equivalent supported operator;
3. move it outside the repeated execution region;
4. precompute it;
5. fuse it into a supported neighboring operation;
6. change its dtype or rank within the allowed contract;
7. use controlled ONNX surgery to replace the exact raw subgraph with a supported standard, contrib, or provider operator;
8. insert a custom operator only when no supported standard representation satisfies the contract and the required custom-op library is available in deployment.

Do not tune provider partitions as an unrelated optimization stage. Do not claim provider placement without evidence, and do not insert a provider-specific or custom operator without verifying its exact domain, opset, dtype, rank, shape, dynamic-shape, and runtime-library requirements.

---

## 15. Minimize Memory and Memory Bandwidth

For each intermediate tensor, estimate:

- shape;
- dtype;
- lifetime;
- consumer count;
- whether it is materialized;
- whether it is contiguous;
- whether it can be eliminated or recomputed cheaply.

Optimize for:

- fewer large intermediates;
- shorter tensor lifetimes;
- fewer full-tensor reads and writes;
- fewer broadcasts that materialize data;
- fewer transpose materializations;
- fewer concatenation copies;
- fewer duplicate caches;
- reduced temporary precision where safe;
- processing before expansion;
- processing before splitting;
- reduction before expensive layout conversion;
- fusing pointwise operations into compute-heavy kernels.

Do not reduce arithmetic if doing so creates a substantially larger intermediate and likely worsens memory bandwidth. Prefer the lower-cost source graph; use focused PyTorch measurements only when representative inputs and hardware are already available.

---

## 16. Minimize Computation

Perform algebraic and graph-level simplification.

Evaluate:

- common-subexpression elimination;
- strength reduction;
- invariant-code motion;
- constant propagation;
- dead-code elimination;
- loop/process unrolling for fixed small structures;
- reduction fusion;
- equivalent lower-cost formulations;
- combining repeated matrix operations;
- sharing normalized or transformed tensors;
- avoiding repeated index/mask generation;
- avoiding repeated scale or reciprocal calculation;
- computing only requested outputs;
- eliminating unused auxiliary outputs.

Prioritize reduction of expensive operations such as:

- MatMul/GEMM;
- convolution;
- Softmax;
- normalization;
- transcendental functions;
- large reductions;
- large Gather/Scatter operations;
- full-tensor layout changes.

But also reduce chains of bandwidth-bound elementwise operators.

---

## 17. ONNX-Friendly Coding

Prefer PyTorch constructs that export into stable standard ONNX operations.

Avoid or rewrite:

- unsupported Python control flow;
- data-dependent Python branching;
- `.item()` in the graph path;
- Python lists built from tensor data;
- non-exportable custom autograd functions;
- unsupported complex indexing;
- unsupported in-place mutation;
- unnecessary custom operators;
- exporter-specific hacks that are unstable across versions;
- operations that decompose into large primitive subgraphs when a supported fused operator exists.

Prefer the modern exporter supported by the project’s PyTorch version.

Use exporter diagnostics to understand lowering when needed. Do not rely on automatic exporter or runtime optimization passes to perform the requested optimization; use an explicit controlled graph rewrite when exporter limitations require post-export fusion.

Use static initializers rather than graph inputs for immutable weights so the raw export directly represents invariant state.

Select the opset based on:

- exporter support;
- provider support;
- required standard operators;
- operator semantics;
- deployment constraints.

Do not automatically choose the newest opset if the target provider supports an older opset better.

---

## 18. Dynamic Shape Policy

Dynamic shapes are not automatically desirable.

For every dynamic dimension, determine whether it is truly required.

If not required:

- make it static;
- fold shape logic;
- precompute masks and indices;
- expose constants directly to exporter capture;
- reduce raw shape and indexing subgraphs;
- make source-level fusion easier.

If dynamic shapes are required:

- minimize the number of symbolic dimensions;
- reuse symbolic shape values;
- avoid data-dependent dimensions;
- avoid unnecessary dynamic reshape targets;
- group deployment shapes into a small number of optimized profiles;
- document when separate pre-export model instances are required for different profiles;
- use source constructs that lower to standard dynamic-shape operators supported by the target contract.

Never remove required dynamic behavior without explicitly documenting the contract change.

---

## 19. Enforce the Controlled Surgery Boundary

Before completing the task, audit every proposed command and code change:

1. Implement every optimization that the exporter can represent reliably in module construction, immutable module state, `forward()`, or a helper reached by `forward()`.
2. Treat the raw exported ONNX model as an immutable baseline and write graph-surgery results to a separate final-model path.
3. Invoke only the explicit targeted rewrite script needed for a documented exporter limitation; do not invoke quantizers, generic simplifiers, generic optimizers, runtime transformers, or offline optimizers.
4. No runtime session, allocator, I/O-binding, provider, or thread policy may be changed as an optimization deliverable.
5. Verify that the targeted rewrite changed only the intended subgraph and required metadata.
6. Document every inserted operator's domain, opset, runtime/provider support, custom-op library requirements, and fallback behavior.
7. Mark unrelated runtime-only ideas as out of scope rather than implementing them.

---

# Rewrite Rules

When modifying `forward()`:

- provide a complete runnable replacement;
- preserve the required function signature;
- move invariant initialization into `__init__`, an export-preparation method, or an explicit preprocessing step;
- register immutable tensors correctly as parameters or buffers;
- avoid recomputation;
- reuse derived dimensions and intermediates;
- include comments only for non-obvious graph-level decisions;
- avoid abstraction layers that hide export behavior;
- favor graph transparency over stylistic abstraction;
- do not leave pseudocode where executable code can be provided.

If modifying module state or weight representation:

- provide a migration/conversion function;
- explain checkpoint compatibility;
- explain whether the original state dictionary still loads;
- avoid silently changing parameter semantics.

If modifying the exported ONNX graph:

- provide a complete runnable and deterministic rewrite script or function;
- use structured ONNX graph APIs rather than byte-level or text substitution;
- match graph semantics and topology, not fragile autogenerated node names alone;
- require exactly the expected match count and fail without writing output on zero, partial, or ambiguous matches;
- verify operator types, domains, versions, attributes, tensor dtypes, ranks, shapes, and initializer contents before rewriting;
- preserve graph input/output names and ordering unless a contract change is explicitly authorized;
- generate collision-free node, tensor, and initializer names;
- preserve external-data handling and model-size constraints;
- maintain topological order and valid producer/consumer relationships;
- update opset imports, custom domains, value information, and metadata deliberately;
- make the rewrite idempotent or detect an already-rewritten model and exit clearly;
- save to a new path only after all structural checks pass;
- emit a concise rewrite report containing matched, inserted, rewired, transformed, and deleted objects.

---

# Validation Requirements

## Source Validation

The final result must:

- parse or compile as valid Python;
- construct the optimized module successfully when dependencies are available;
- run the optimized PyTorch `forward()` on representative inputs when dependencies and weights are available;
- preserve required input/output signatures, shapes, dtypes, and state/cache updates;
- keep all exporter-representable optimization logic before `torch.onnx.export()` and isolate justified post-export surgery in an explicit rewrite stage;
- avoid adding unsupported Python control flow, data extraction, or mutation to the captured path.

## Numerical Validation

Compare the original and optimized PyTorch implementations first. If graph surgery is used, comparison of the immutable raw ONNX model against the final rewritten ONNX model is mandatory. If graph surgery is not used, raw ONNX comparison remains optional validation.

Test:

- minimum supported shapes;
- typical shapes;
- maximum supported shapes;
- odd and even dimensions;
- boundary sequence lengths;
- multiple batch sizes;
- cache-empty and cache-populated cases;
- relevant dtypes;
- representative random and real inputs.

Report:

- maximum absolute error;
- maximum relative error;
- mean absolute error;
- cosine similarity where relevant;
- exact equality for integer and boolean outputs;
- whether NaN and Inf behavior changed.

Approximate rewrites require explicit authorization.

## Raw and Final ONNX Validation

When the environment can export and execute the model without unrelated setup work:

- export the source-optimized module to an immutable raw-model path;
- use the same exporter, opset, inputs, dynamic-axis policy, and constant-folding setting;
- apply each justified targeted graph rewrite to a separate final-model path;
- run ONNX checker and shape inference on both models where supported;
- verify that no unexpected custom or ATen fallback nodes were introduced;
- compare raw and final operator histograms, initializers, graph interfaces, and targeted node families;
- compare raw and final ONNX outputs with graph optimizations disabled when the runtime permits it;
- load required custom-op libraries explicitly before validating a custom fused operator;
- preserve both artifacts so the final model can be reproduced and audited.

Do not require unrelated runtime profiles, latency ablations, or post-export runtime tuning. Provider placement and performance claims still require evidence from the exact target environment.

---

# Required Deliverables

Return all of the following:

## 1. Assumptions

List the PyTorch/exporter version when known, target opset, shape policy, dtype policy, immutable-state assumptions, checkpoint constraints, and numerical tolerance.

## 2. Baseline Audit

Describe expensive or redundant work in module construction and `forward()`, its raw ONNX lowering, and every raw subgraph that is a candidate for controlled surgery because of an exporter limitation.

## 3. Optimization Plan

Rank pre-export source changes first by expected graph impact and correctness risk. Then list each proposed post-export rewrite separately with its exporter-limitation evidence, replacement operator, runtime support, expected benefit, and validation plan.

## 4. Complete Optimized Code

Provide complete runnable changes to the export script, module, and helpers required by the pre-export rewrite, plus the complete deterministic ONNX rewrite script when graph surgery is justified.

## 5. Patch or Change Map

Show every meaningful source and ONNX rewrite change. For graph surgery, identify the matched raw subgraph, replacement operator, rewired edges, transformed initializers, deleted nodes, metadata updates, and expected graph-level effect.

## 6. Before/After Graph Report

Report the following for the source path, immutable raw export, and final rewritten graph when available:

- invariant computations moved out of `forward()`;
- matrix, convolution, normalization, and reduction operations removed or combined;
- expected or measured Cast count;
- expected or measured shape-operation count;
- expected or measured layout-operation count;
- expected or measured indexing-operation count;
- expected or measured elementwise-operation count;
- large temporary tensors removed or introduced;
- immutable buffers or packed parameters added;
- raw and final node and initializer counts;
- inserted fused operators and removed decompositions;
- graph input/output and dynamic-shape contract changes, which should normally be none;
- custom domains, opset imports, and runtime library requirements.

## 7. Numerical Validation Report

Provide measured original-PyTorch, source-optimized-PyTorch, raw-ONNX, and final-ONNX comparison results when available. State clearly when dependencies, weights, custom-op libraries, target hardware, or representative inputs prevent execution.

## 8. No-Skill-Lost Checklist

For every optimization skill in this prompt, mark:

- applied;
- already optimal;
- not applicable;
- rejected, with a concrete reason.

## 9. Surgery Justification and Scope Confirmation

Confirm that pre-export optimization was attempted first. For every post-export rewrite, document why the exporter could not express the required graph, why the selected replacement is supported, and how the rewrite is constrained to the intended subgraph. Confirm that no unrelated quantization, simplification, or runtime-tuning stage was introduced.

---

# Mandatory Optimization Checklist

Explicitly evaluate every item:

- precomputation of immutable module state;
- registration of reusable constants and buffers;
- reuse of common `forward()` intermediates;
- zero-copy or metadata-only views where export-safe;
- minimum source-visible communication and transfer;
- minimum dtype casts;
- minimum cached dtype casts;
- minimum transpose;
- minimum permute;
- minimum reshape;
- minimum view;
- minimum squeeze;
- minimum unsqueeze;
- minimum flatten;
- minimum unflatten;
- deep shape inference;
- minimum `tensor.shape`;
- minimum `tensor.size()`;
- reuse of known dimensions;
- minimum computations;
- minimum temporary tensors;
- minimum memory usage;
- minimum memory bandwidth;
- minimum position-wise operators;
- minimum elementwise operators;
- minimum bitwise operators;
- fused weights;
- fused biases;
- fused scales;
- folded operators;
- folded processes;
- constant folding;
- common-subexpression elimination;
- dead-code elimination;
- process tensors together before splitting;
- int32 Gather/index-select indices where beneficial and supported;
- int64 Slice controls where beneficial or required;
- Split instead of repeated Slice where measurably better;
- ONNX-friendly operators;
- stable raw-export patterns;
- standard target-provider-compatible operators;
- flip-based `rotate_half()` candidate;
- source-level rotary-embedding fusion;
- target-provider-friendly operators;
- minimum source-visible host/device synchronization;
- explicit evaluation of whether each remaining decomposition requires post-export surgery;
- targeted replacement with standard or runtime-supported fused operators when justified;
- strict pattern, attribute, dtype, rank, shape, and initializer preconditions;
- fail-closed and reproducible graph-rewrite behavior;
- preservation of the immutable raw model and separate final output;
- dead-node and initializer removal limited to the rewritten region;
- graph interface, value-info, opset, and custom-domain validation;
- custom-op runtime library and provider-support documentation;
- no post-export quantization or mixed-precision pass;
- no blanket ONNX simplifier, generic optimizer, or unrelated graph cleanup pass;
- no runtime graph-transformer or offline-optimization step;
- no runtime I/O-binding, allocator, thread, or session tuning;
- read-only treatment of the raw export and audited treatment of the rewritten final model.

---

# Decision Standard

Apply an optimization only when at least one of the following is demonstrated:

- fewer operations executed by `forward()`;
- invariant work moved into module construction or export preparation;
- fewer or smaller temporary tensors;
- fewer dtype casts or layout conversions;
- fewer shape reads and dynamic shape computations;
- fewer repeated indexing, masking, or elementwise processes;
- fused immutable weights, biases, or scales;
- combined compatible branches before splitting;
- simpler and more stable exporter lowering;
- fewer nodes in the immutable raw export through source optimization;
- replacement of an unavoidable raw decomposition with a supported fused operator;
- fewer final nodes, provider boundaries, transfers, or large intermediates after targeted surgery;
- lower PyTorch execution cost on representative inputs, when measured without changing the model contract.

Do not make deployment latency or provider-placement claims without separate target-runtime evidence. Such evidence may inform a later task, but collecting it through post-export tuning is outside this skill.

Reject a source rewrite if it only makes the PyTorch code look simpler while producing an equal or worse graph. Reject graph surgery if its justification, match preconditions, runtime support, numerical equivalence, or measurable graph benefit is not demonstrated.

When alternatives conflict, trust in this order:

1. numerical equivalence and contract preservation in PyTorch;
2. numerical equivalence and contract preservation in the final ONNX graph;
3. successful exporter capture and immutable raw ONNX validation;
4. exact target-runtime operator support and provider placement evidence;
5. the final rewritten ONNX graph and its reproducible surgery report;
6. source-level tensor-size, operation-count, and dtype/layout analysis;
7. PyTorch source appearance.

Be aggressive, but remain evidence-driven.

