# ONNX Runtime Android Arm64-v8.2-A Contrib CPU Full-Corpus Refactor and Rebuild Task

## TOPMOST GOAL — LOWER BIT FIRST

**"Lower bit first" is the highest optimization priority:** prefer Q4
(`uint4`, `int4`, `fp4`), then Q8 (`int8`, `uint8`, `fp8`), then Q16 (`FP16`,
`BF16`, `int16`, `uint16`), then Q32 (`int32`, `uint32`); use 64-bit variables
only as the last legal choice. Apply this ordering to tensor storage, packed
weights, activation tiles, arithmetic paths, accumulators, scales/zero points,
prepared metadata, dimensions, strides, offsets, counters, and **ordinary
`for`-loop indices**.

This priority never permits semantic loss, overflow, pointer truncation,
unsupported formats, or extra conversion traffic. For every value, choose the
lowest width that is proven to cover its complete registered range and preserve
the operator's exact rounding/saturation/tolerance contract. If a narrower
candidate would add repeated extension/conversion instructions, a full-tensor
Cast, extra cache/RAM traffic, or slower code, measure/document that rejection
and try the next width. Do not jump directly to 32 or 64 bits out of habit.

The enforceable width ladder is:

1. **4-bit payload first.** Keep supported `uint4`/`int4`/`fp4` values packed
  through storage, prepacking, cache tiles, and the leaf microkernel. Standard
  C++ has no addressable native 4-bit scalar or 4-bit loop counter: represent
  Q4 payloads with the repository's exact packed format (normally two nibbles
  per byte) and never invent a fake 4-bit C++ control type or silently
  reinterpret one FP4 encoding as another.
2. **8-bit next.** Prefer `int8_t`/`uint8_t` and the exact registered FP8 format
  for legal payload/compute paths. An 8-bit counter/index is allowed only when
  its full bound is proven and integer promotion/extension does not erase the
  benefit.
3. **16-bit next.** Prefer FP16/BF16/`int16_t`/`uint16_t` when the contract and
  accumulator bound permit it. Keep FP16 and BF16 distinct. A 16-bit
  counter/index requires a proven bound and favorable generated code.
4. **32-bit next.** Use `int32_t`/`uint32_t` for the normal prepared dimension,
  stride, tile, thread, and loop-index path when 8/16 bits are unsuitable.
  INT32 is the default quantized accumulator unless a smaller accumulator is
  mathematically proven safe. FP32 remains permitted only where the registered
  floating contract, accuracy, scaling, or accumulation requires it.
5. **64-bit last.** Use `int64_t`/`uint64_t`/`size_t`/`ptrdiff_t` only for public
  ABI/schema values, pointers/address generation, allocation/byte sizes,
  checked overflow validation, atomics/counters whose range requires it, and a
  valid large-tensor fallback. Narrow once after checked preparation; never run
  the common inner loop or normal `for` induction variable in 64 bits merely
  because a shape API returned 64 bits.

Every primary-file milestone MUST include a **width audit** listing hot payloads,
accumulators, metadata, and each performance-relevant loop induction variable;
their proven ranges; the selected width; narrower candidates rejected and why;
and disassembly evidence for any remaining 64-bit hot arithmetic. This goal
takes precedence over all performance recommendations below, subject only to
the no-semantic-loss and ABI/safety invariants.

Act as a principal Android CPU performance engineer with expert command of ONNX
Runtime's CPU Execution Provider (CPU EP), the MLAS / KleidiAI microkernel
layer, Android NDK/Clang, CMake, ELF shared-library packaging, Android ABIs,
heterogeneous mobile SoCs, portable and architecture-specific SIMD (NEON, FP16,
dot-product, I8MM, SVE/SVE2), graph fusion, arenas, memory planning, `adb`,
Perfetto, simpleperf, and hardware-counter-guided tuning.

`MUST`, `MUST NOT`, and `SHOULD` are normative.

**This is an execution task, not an audit.** The deliverable is *a concrete,
behavior-preserving refactor of every recursive `*.cc` and `*.h` under
`onnxruntime/contrib_ops/cpu`, plus a rebuilt Android `libonnxruntime.so`* — not
a report about them. Process the corpus one file at a time. Inventories,
coverage tables, checkpoints, ledgers, and benchmarks are **supporting evidence
that trails the edits**; they MUST NOT become the work itself. A run that
inspects, catalogs, "verifies," or measures files but leaves any in-scope source
or header without a meaningful source-level refactor has FAILED this task. Do
not stop after discovery, after one subtree, after one context window, or after
the first successful build.

**"Without loss" is a hard two-part constraint.** (1) *No file or coverage
loss* — you MUST refactor every recursive `*.cc` and `*.h` in the authoritative
root and preserve every registered contrib CPU operator. Skipping a file
because it is a header, "not hot," absent from a benchmark model, difficult, or
mostly registration code is forbidden. (2) *No semantic loss* — every edit
MUST preserve output values within the registered numerical tolerance and
preserve every registered opset, dtype, rank, shape, attribute, optional
input/output, aliasing rule, and generic fallback exactly.

"Zero memory traffic" here means **zero avoidable traffic**, not the impossible
removal of compulsory input reads and logical output writes. For each kernel
derive the semantic minimum bytes that must cross each memory boundary and drive
actual traffic toward that bound. Any zero-copy, zero-temporary, single-pass, or
cache-resident claim MUST show that no hidden allocation, materialization,
repack, writeback, or fallback moved the work elsewhere.

## 0. Authoritative Scope and First Blocking Milestone

The repository to modify is exactly:

```text
/home/iamj/Downloads/onnxruntime
```

Operate on the **current worktree at that path**, including all existing tracked
and untracked user changes. Start by reading `git status`, the current diff, and
the current implementation. Never reset, clean, checkout over, revert, or replace
the current kernel with an upstream copy. Existing edits are the baseline input
to continue, review, test, and improve; merely describing or reapplying them is
not progress.

### 0.1 Full recursive file scope

The authoritative source root is exactly:

```text
/home/iamj/Downloads/onnxruntime/onnxruntime/contrib_ops/cpu
```

The manifest MUST contain every regular file returned by this command, with no
manual exclusions:

```bash
find onnxruntime/contrib_ops/cpu -type f \
  \( -name '*.cc' -o -name '*.h' \) -print0 | sort -z
```

At the time this prompt was written, the root contains **195 files: 89 `*.cc`
and 106 `*.h`**. This count is only a sanity check; the command above is
authoritative and newly added files MUST be appended to the manifest. The
current recursive groups are the root-level files plus `aten_ops/`, `attnlstm/`,
`bert/`, `math/`, `moe/`, `quantization/`, `sparse/`, `tensor/`,
`transformers/`, and `utils/`.

Every manifest file MUST receive a meaningful non-comment source diff and its
own focused review/checkpoint. For a declaration-only header, perform a safe
structural refactor that supports prepared execution, tighter ownership,
compile-time work, compact metadata, or a corresponding implementation
optimization. Comment-only edits, formatting-only edits, include sorting,
renaming without runtime or maintainability value, generated churn, and adding
tests without changing the in-scope file do not count. `already-optimal` is not
a completion status for this user-requested full rewrite. If no safe change is
immediately evident, keep the row pending, inspect its callers/implementations,
and return after gaining evidence; do not manufacture a semantic change merely
to alter the file.

Files outside this root are **dependency/support scope**, not substitute scope.
MLAS, KleidiAI, standard CPU EP, framework, CMake, and tests MAY and SHOULD be
edited when an in-scope contrib file dispatches there or requires validation,
but changing a dependency does not mark any untouched contrib file complete.
Conversely, each contrib file remains individually accountable even when many
of them share one optimized MLAS primitive.

The first blocking implementation target is the complete `MatMulNBits` /
Q4-Q8-Q16-Q32 call chain below. These are real paths in this worktree, not
examples. Trace the actual runtime-selected path from the operator through MLAS
dispatch to the leaf Arm64 C++/intrinsics/assembly or KleidiAI ukernel. **Do not
stop at the operator wrapper.**

### 0.2 Exact source paths for the first milestone

**Operator front ends and prepacking**

- `onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc`
- `onnxruntime/contrib_ops/cpu/quantization/matmul_nbits_impl.cc`
- `onnxruntime/contrib_ops/cpu/quantization/matmul_nbits_impl.h`
- `onnxruntime/contrib_ops/cpu/quantization/matmul_nbits_helper.h`
- `onnxruntime/contrib_ops/cpu/quantization/matmul_bnb4.cc`
- `onnxruntime/contrib_ops/cpu/matmul_fpq4.cc`
- `onnxruntime/contrib_ops/cpu/quantization/matmul_integer16.cc`
- `onnxruntime/contrib_ops/cpu/quantization/matmul_integer16.h`
- `onnxruntime/core/providers/cpu/quantization/quantize_linear.cc`
- `onnxruntime/core/providers/cpu/quantization/dynamicquantizelinear.cc`
- `onnxruntime/core/providers/cpu/quantization/matmul_integer.cc`
- `onnxruntime/core/providers/cpu/quantization/quantize_linear_matmul.cc`
- `onnxruntime/core/providers/cpu/quantization/qlinearconv.cc`

**Lowest in-tree MLAS / Arm64 implementation**

- `onnxruntime/core/mlas/lib/qnbitgemm.cpp`
- `onnxruntime/core/mlas/lib/qnbitgemm.h`
- `onnxruntime/core/mlas/lib/qnbitgemm_kernel_neon.cpp`
- `onnxruntime/core/mlas/lib/qnbitgemm_kernel_neon.h`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_int8.cpp`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_int8_i8mm.cpp`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_fp32.cpp`
- `onnxruntime/core/mlas/lib/hqnbitgemm_kernel_neon_fp16.cpp`
- `onnxruntime/core/mlas/lib/hqnbitgemm_kernel_neon_fp16_8bit.cpp`
- `onnxruntime/core/mlas/lib/sqnbitgemm_q8_block.h`
- `onnxruntime/core/mlas/lib/qgemm.cpp`
- `onnxruntime/core/mlas/lib/qgemm.h`
- `onnxruntime/core/mlas/lib/qgemm_kernel_neon.cpp`
- `onnxruntime/core/mlas/lib/qgemm_kernel_sdot.cpp`
- `onnxruntime/core/mlas/lib/qgemm_kernel_udot.cpp`
- `onnxruntime/core/mlas/lib/qgemm_kernel_smmla.cpp`
- `onnxruntime/core/mlas/lib/qgemm_kernel_ummla.cpp`
- `onnxruntime/core/mlas/lib/aarch64/HalfGemmKernelNeon.S`
- `onnxruntime/core/mlas/lib/aarch64/QgemmS8S8KernelNeon.S`
- `onnxruntime/core/mlas/lib/aarch64/QgemmS8S8KernelSdot.S`
- `onnxruntime/core/mlas/lib/aarch64/QgemmS8S8KernelSmmla.S`
- `onnxruntime/core/mlas/lib/aarch64/QgemmU8X8KernelNeon.S`
- `onnxruntime/core/mlas/lib/aarch64/QgemmU8X8KernelUdot.S`
- `onnxruntime/core/mlas/lib/aarch64/QgemmU8X8KernelUmmla.S`
- `onnxruntime/core/mlas/lib/aarch64/SymQgemmS8KernelNeon.S`
- `onnxruntime/core/mlas/lib/aarch64/SymQgemmS8KernelSdot.S`
- `onnxruntime/core/mlas/lib/quantize.cpp`
- `onnxruntime/core/mlas/lib/dequantize.cpp`
- `onnxruntime/core/mlas/lib/halfgemm.cpp`
- `onnxruntime/core/mlas/lib/halfgemm_kernel_neon_fp16.cpp`
- `onnxruntime/core/mlas/lib/platform.cpp`
- `onnxruntime/core/mlas/lib/kai_ukernel_interface.cpp`
- `onnxruntime/core/mlas/lib/kai_ukernel_interface.h`
- `onnxruntime/core/mlas/lib/kleidiai/qgemm_kleidiai.cpp`
- `onnxruntime/core/mlas/lib/kleidiai/halfgemm_kleidiai.cpp`

**Prepacked-weight ownership and sharing, only when the measured waste crosses
the kernel/framework boundary**

- `include/onnxruntime/core/framework/op_kernel.h`
- `onnxruntime/core/framework/prepacked_weights_container.h`
- `onnxruntime/core/framework/prepacked_weights_container.cc`
- `onnxruntime/core/framework/session_state.cc`

**Build and dispatch wiring**

- `cmake/onnxruntime_mlas.cmake`
- `cmake/CMakeLists.txt`

The front-end files define contract, shape, and prepack lifetime; they are not
the final optimization destination. For every hot call, write down the exact
function chain and selected dispatch entry. If the leaf is an in-tree MLAS
kernel, edit that leaf. If the leaf is a KleidiAI ukernel, optimize the in-tree
packing, tiling, dispatch, workspace, or epilogue around it and prove the
external ukernel is not being fed through an avoidable conversion or copy. A
change confined to comments, registration lists, benchmark code, tests, build
flags, or an operator wrapper does **not** satisfy this phase.

After mechanically creating the full manifest—but before reading or auditing
later rows—the agent MUST finish at least one complete vertical optimization in
this stack: edit the current in-scope implementation and, where required, its
lowest-level dependency; remove measured pack/copy/allocation waste; add or
strengthen focused tests; rebuild affected targets; relink the Android shared
library; checkpoint; and inspect generated AArch64 code. Planning, profiling,
or a baseline build without a subsequent source edit is not a deliverable or a
stopping point.

### 0.3 Required mixed-width paths

Use `Q4`, `Q8`, `Q16`, and `Q32` as the mandatory lower-bit-first width classes,
not as permission to invent operator semantics:

- **Q4 (`uint4`, `int4`, `fp4`):** preserve each registered packed encoding,
  signedness, FP4 layout, scales, and optional zero points exactly. Consume
  packed nibbles directly in the selected microkernel or unpack only the current
  register/cache tile. A full `K*N` dequantized weight tensor is forbidden in
  repeated execution. Unsupported FP4/int4/uint4 variants remain on the correct
  generic path; never reinterpret formats to claim coverage.
- **Q8 (`int8`, `uint8`, `fp8`):** use exact registered integer or FP8 storage
  and arithmetic. Route integer paths to NEON dot-product or I8MM when HWCAP
  permits. Quantize an activation row/tile at most once and reuse it across the
  largest legal output tile; do not repeatedly scan `A` for each `N` panel.
  Preserve each FP8 encoding and its NaN/Inf/saturation behavior.
- **Q16 (`FP16`, `BF16`, `int16`, `uint16`):** do not conflate these formats.
  Prefer native FP16 vector/FMLA storage and compute on the guaranteed
  Armv8.2-A+FP16 path; runtime-gate optional BF16 instructions and preserve
  integer signedness, overflow, rounding, and saturation semantics. Do not
  create a whole-tensor FP32↔FP16/BF16 conversion merely to enter a microkernel.
- **Q32 (`int32`, `uint32`):** keep integer accumulators, prepared dimensions,
  strides, offsets, counters, and loop indices at 32 bits when narrower legal
  widths are unsuitable. Keep accumulators in registers through the epilogue
  and store each logical output once. FP32 is a separate floating contract and
  remains only where registered output, scale, or accuracy-preserving
  accumulation requires it; do not widen integer control state to 64 bits.

The optimized implementation MUST exercise and compare at least these direct
pipelines when the schema supports them:

1. FP32 activation × Q4 weight → FP32 output.
2. FP16 activation × Q4/Q8 weight → FP16 or FP32 output as registered.
3. FP32/FP16 activation → tile-local Q8 activation × Q4 weight → INT32
   accumulation → fused scale/zero-point/bias → final FP16/FP32 output.
4. Q8 activation × Q8 weight → INT32 accumulation → fused requantization or
   final registered output.
5. INT16 × INT16 → the exact registered `MatMulInteger16` output, with a proven
   overflow-safe accumulator and unchanged empty/tail behavior.

For each pipeline, retain a correct generic fallback for unsupported block
sizes, tails, alignment, dtypes, asymmetry, CPU features, and large dimensions.
Selection MUST be based on reusable shape/dtype/attribute/HWCAP predicates, not
on a model name or device vendor.

### 0.4 "No 64-bit processing" means no avoidable 64-bit hot arithmetic

Android `arm64-v8a` is an AArch64 ABI: pointers, `size_t`, address generation,
ELF objects, and ONNX/ORT public shape APIs are necessarily 64-bit. Removing all
64-bit instructions is physically impossible and truncating them is a
correctness/security bug. In this task, **without 64-bit processing** has this
precise, enforceable meaning:

- No avoidable `int64_t`, `uint64_t`, `size_t`, `ptrdiff_t`, `long`, or `double`
  arithmetic in a quantization, packing, tile, dot-product, reduction, or
  epilogue inner loop.
- Validate dimensions, products, byte sizes, offsets, and pointer ranges once
  with checked wide arithmetic at the preparation boundary. If they fit, store
  immutable prepared extents, strides, block counts, tile indices, and loop
  bounds at the lowest profitable fixed width: first 8-bit, then 16-bit, then
  `uint32_t`/`int32_t`. Keep the existing checked 64-bit fallback for valid
  large tensors and record any narrower candidate rejected due to repeated
  extension/address-generation cost.
- In generated AArch64 code, expect `xN` registers for pointers and address
  generation, but require `wN` registers or vector lanes for prepared counters,
  metadata, and integer compute. FP64 scalar/vector arithmetic is forbidden in
  FP16/FP32 kernels unless the public numerical contract explicitly requires
  double.
- Do not add a full metadata-conversion pass. Narrow once while preparing the
  kernel/request and consume the prepared record through all tiles.
- Inspect the final leaf-kernel disassembly and classify every remaining 64-bit
  arithmetic instruction as compulsory pointer/ABI work, checked preparation,
  or a defect to remove. A grep of C++ type names alone is not proof.

### 0.5 Cache and memory-traffic acceptance for the primary stack

For each Q4/Q8/Q16/Q32 kernel shape class, state the compulsory bytes for `A`,
packed `B`, scale/zero-point metadata, bias, and `C`, then count actual passes.
The steady-state target is:

- Constant `B` is packed exactly once in `PrePack()` and shared safely when
  allowed; `Compute()` never hashes, repacks, or dequantizes the whole weight.
- Each required `A` element is read once per mathematically necessary reuse
  domain. Dynamic quantization, min/max, scale, and row/block sums are fused into
  one tile-preparation pass where exact semantics allow.
- Packed Q4/Q8 panels, scales, and zero-point correction data use one
  cache-contiguous format consumed directly by the leaf kernel. Do not create
  parallel packed layouts unless measurements prove the extra persistent bytes
  buy a net end-to-end win.
- `M == 1` token/GEMV and `M > 1` prefill/GEMM have distinct measured tiling when
  their reuse differs. Neither may fall through an obviously unsuitable kernel
  merely because the other path is fast.
- The live `A` tile, packed `B` panel, accumulators, and metadata fit the stated
  L1D budget with associativity/headroom; the next blocking level fits L2. Treat
  L3/LLC as optional because many mobile SoCs expose only a shared system cache.
- No heap allocation, mutex, backend selection, error-string construction, or
  scratch-size calculation occurs per tile. Repeated execution has zero
  avoidable allocation, copy, Cast, repack, and full-tensor temporary bytes.
- Accumulators stay in NEON registers for the micro-tile, correction/bias/scale
  is fused into the epilogue, and each logical `C` element is written once.
- Prefetch is accepted only after counters show a gain and it never crosses a
  valid allocation. Do not use speculative prefetch or non-temporal stores as a
  substitute for correct blocking.

Where no physical Arm device is connected, source edits, cross-builds, tests,
static byte-pass accounting, and AArch64 disassembly remain mandatory. Only
hardware-counter and thermal A/B numbers may be marked externally blocked.

## 1. Mission, Source Baseline, and Deliverables

Starting from the **current source worktree**, directly refactor and deeply
optimize every recursive C++ source/header in `onnxruntime/contrib_ops/cpu`,
then rebuild the Android native library. A successful run concretely produces:

- A meaningful, validated source diff in every manifest `*.cc` and `*.h` under
  the authoritative root, preserving semantics and registration coverage.
- A rebuilt Android `arm64-v8a` `libonnxruntime.so`, compiled for an explicit
  Armv8.2-A+FP16 minimum CPU contract, with matching public C/C++ headers and a
  preserved `OrtApi` ABI.
- Focused correctness/build evidence and a compact durable checkpoint for every
  processed file (Sections 4 and 10).

Existing user changes in the worktree are part of the input state and MUST NOT
be discarded or silently overwritten. Before the first edit, fingerprint the
baseline as `HEAD + tracked diff + relevant untracked files + submodule
revisions + build config` so the candidate can be diffed against it; a clean
commit alone is insufficient when the worktree is dirty. Build baseline and
candidate into separate output directories.

The primary optimization scope is the **entire recursive contrib CPU C++
corpus**, not a hot subset: every source/header, every contrib CPU registration,
and every registered opset, dtype, rank, shape, attribute, optional-I/O variant,
and fallback represented there. Shared CPU compute beneath it—MLAS/KleidiAI,
architecture dispatch, prepacking, thread partitioning, allocation, and limited
framework/build wiring—is supporting scope whenever tracing an in-scope file
reaches it. Standard CPU operator files are not separate completion rows.

Android `arm64-v8a`, with Armv8.2-A+FP16 as the documented minimum CPU contract,
is the sole production artifact and performance target. Host `x86_64` tests may
be used to validate generic semantics, but x86 Android artifacts and AVX tuning
are not deliverables. `armeabi-v7a`, Android `x86`, and Android `x86_64` are
**out of scope** — do NOT build, target, benchmark, or optimize them. Physical
device numbers are for the production Arm ABI; emulator/host results validate
behavior only.

**C/C++ API only.** MUST NOT pass `--build_java`, build or package
`libonnxruntime4j_jni.so`, or emit an ORT Java/Kotlin JAR, Maven, or AAR
artifact. Integrate `libonnxruntime.so` through CMake / ndk-build / linker
paths. Prefer static C++ runtime linkage; if the app intentionally uses
`libc++_shared.so`, treat it as an external dependency, not an ORT deliverable.

Keep two artifact classes separate: a **full-contract build** that proves every
operator and fallback still exists, and an optional **reduced production build**
specialized to the app's model corpus. Removing registrations is binary
specialization, not kernel optimization, and MUST NOT be reported as — or
applied to — the full-contract artifact.

The objective is maximum useful CPU work per cycle and per byte moved: raise
operator throughput and cut latency, dispatch overhead, peak RSS/PSS, allocator
traffic, cache/TLB misses, and bandwidth. Drive avoidable DRAM, LLC,
private-cache, and L1 traffic toward zero; keep operands in registers or the
nearest cache; write each logical output once, directly to its destination.

CPU kernels, CPU contrib/custom ops, MLAS, KleidiAI, architecture dispatch, CPU
graph transforms, memory planning, thread pools, and CPU session init MAY be
changed. Shared runtime code changes only with proof other EPs are behaviorally
unaffected. Do NOT alter the behavior, registration, partitioning, kernel
selection, or compilation of any non-CPU EP; unused EPs may be omitted by build
config but their source stays intact.

Generating an inventory or checkpoint is not an optimization. Passing existing
tests is not a performance gain. Optimizing one operator, model, subtree, CPU,
or backend is a milestone, never completion. Continue until every manifest file
has its own accepted source diff; only genuine external hardware/toolchain gaps
may block device evidence, and they never excuse skipping a source refactor that
can be made and validated locally.

## 2. Authoritative Contrib CPU Corpus and Work Order

The manifest generated by Section 0.1 is the sole completion list. The current
snapshot is distributed as follows; regenerate counts at execution time:

| Group | Current `*.cc` + `*.h` count |
|---|---:|
| `quantization/` | 34 |
| `bert/` | 37 |
| `transformers/` | 42 |
| `attnlstm/` | 10 |
| `moe/` | 8 |
| `tensor/` | 6 |
| `sparse/` | 4 |
| `utils/` | 4 |
| `aten_ops/` | 3 |
| `math/` | 1 |
| root-level files | 46 |
| **Total snapshot** | **195** |

Process one primary manifest file at a time in this deterministic milestone
order, using lexicographic path order inside each group unless an already-active
worktree edit establishes the first file:

1. `quantization/` (begin with the active `matmul_nbits` stack in Section 0).
2. `bert/`.
3. `transformers/`.
4. `moe/`.
5. `attnlstm/`.
6. `tensor/`, `sparse/`, `math/`, `aten_ops/`, and `utils/`.
7. Root-level `onnxruntime/contrib_ops/cpu/*.cc` and `*.h`.

A source/header pair may require atomic companion edits to stay buildable, but
each path keeps a separate manifest row and receives a separate primary-file
review and checkpoint. Touching a companion while processing another row does
not make it complete. Do not hold multiple unrelated implementations in active
context, skip ahead to easier files, or mark an entire family complete because
one shared primitive was optimized.

**Coverage rule.** A file becomes complete only after that exact file has a
non-cosmetic source diff, its full registered contract and callers have been
reviewed, the smallest affected target builds, focused correctness tests pass,
and its durable checkpoint is written. Dependencies outside the root may be
changed, but they have no manifest status. Re-enumerate the authoritative `find`
command at each subtree boundary and immediately append new paths. Completion
requires a final set comparison with zero missing, extra, duplicate, pending,
or active rows.

## 3. Non-Negotiable Invariants (What "Without Loss" Forbids)

Every edit MUST preserve:

- Registered operator and graph semantics; numerical results within the
  operator's approved tolerance (exact equality for integer, index, shape,
  string, sequence, control-flow, and other deterministic outputs).
- Every registered opset range, dtype, rank, shape, attribute, optional
  input/output, broadcasting rule, empty-tensor case, aliasing rule,
  determinism, and concurrency contract.
- Initialized logical outputs (never expose uninitialized or hashed padding
  bytes), memory/capacity bounds, request isolation, and state ownership.
- The generic fallback for every valid input a new fast path does not cover.

Coverage is itself an invariant: the full-contract build MUST keep registering
and running every operator it did before. Dropping, gating, or `#if 0`-ing a
registration to shrink a build is allowed only in the *separate* reduced
production artifact (Section 1), never in the full-contract library.

**Never delete a safety check to save cycles — move it earlier.** Prove each
check's condition at the earliest stage it can be proven, in this order:
(1) build config, (2) offline graph compile, (3) optimized-model generation,
(4) session creation, (5) kernel creation / prepacking, (6) request/cache
creation, (7) runtime only when earlier proof is impossible. The hot path
(per-run, per-node, per-token, per-row, per-tile, inner-loop) MUST NOT repeat
invariant rank/shape/type/axis/opset/attribute/optional-input checks,
capacity/alignment/contiguity/ISA checks, kernel or backend selection, fallback
selection, allocator lookup, scratch allocation, tensor-shape construction, or
diagnostics/logging/error-string construction. Resolve CPU capabilities once per
process/session, not per invocation.

**EP isolation.** Classify each file you touch as CPU-only, shared-runtime,
non-CPU-EP, or build-config. CPU kernels, CPU contrib/custom ops, MLAS,
KleidiAI, CPU dispatch, CPU graph transforms, CPU memory planning/threading, and
CPU session init are in scope. Shared-runtime code changes only with an explicit
argument for why every other EP is behaviorally unaffected. Never change the
behavior, registration, partitioning, kernel selection, or compilation of a
non-CPU EP. Compiling is not proof of isolation — reason about the callers.

**Dispatch predicates.** Fast-path and graph-rewrite selection MUST key on
reusable schema/attribute/dtype/layout/shape-class predicates and, for SIMD, on
HWCAP/OS-reported CPU features — never on model name, graph hash, a single
model's fixed dimensions, or phone/SoC/vendor/build-fingerprint identity. A
feature-gated path MUST stay correct on every CPU in the thread's Android
scheduling pool (big/middle/little migration safe).

**Do not accelerate avoidable work — remove it.** An avoidable Cast, copy,
transpose, cache concat, or temporary should be eliminated, made metadata-only,
fused, or written straight to its final destination — not merely made faster.

**Missing device ≠ stop.** If the Android project or a physical production-ABI
device is absent, you MUST still perform and keep every source optimization, the
native build, unit-test parity, and (where possible) emulator functional
validation. Report only the *device-measured* numbers as a blocker naming the
missing device and the exact resume commands. Absence of a benchmarking device
NEVER converts an un-attempted kernel optimization into "done," and emulator or
host numbers MUST NOT be presented as Android device performance.

## 4. One-File Transaction and Durable Context-Compaction Protocol

This section is the engine of the task. The corpus is intentionally too large
for one context window. Progress MUST be reconstructable from disk without chat
history, and the active reasoning set MUST stay bounded to one primary file.

### 4.1 Durable state directory

Before the first source edit, create and continuously maintain this state root:

```text
/home/iamj/Downloads/onnxruntime/.agent_state/contrib_ops_cpu_refactor/
```

It MUST contain:

```text
BASELINE.md                 # immutable HEAD/diff/submodule/toolchain fingerprint
MANIFEST.tsv                # one row for every authoritative *.cc/*.h
CURRENT.md                  # compact resume capsule, at most 12 KiB / 200 lines
DECISIONS.md                # concise cross-file ABI/layout/dispatch decisions
milestones/M####_<slug>.md  # one bounded checkpoint per primary file
families/F##_summary.md     # compressed subtree summaries
diffs/M####.patch           # exact accepted milestone diff
logs/M####/                 # raw build/test/bench/disassembly output
```

The state directory is an execution aid, not product source. Do not add it to a
git commit or package it in `libonnxruntime.so`; do not delete it between
sessions. Raw logs and patches go under `logs/` and `diffs/`, never into
`CURRENT.md`. If a persistent memory facility is available, mirror only
`CURRENT.md`; the on-disk files remain authoritative.

`MANIFEST.tsv` MUST have these columns:

```text
sequence  group  path  baseline_sha256  current_sha256  status  milestone  tests  build  notes
```

Allowed source status values are `pending`, `active`, `edited_unverified`, and
`verified`. Exactly zero or one row may be `active`. There is no
`already-optimal`, `skipped`, or context-related `blocked` status. A missing
phone may block a separate device-measurement field in `notes`; it never blocks
the locally actionable source row. Update state files atomically and verify the
manifest has unique paths after every change.

`CURRENT.md` is overwritten—not endlessly appended—at every checkpoint. It
MUST contain only:

- baseline fingerprint ID and current worktree-diff hash;
- manifest totals by status and completed group;
- current/next sequence and exact path;
- last accepted milestone and links to its patch/log/checkpoint files;
- active build directories, pinned SDK/NDK/CMake/Ninja, and important flags;
- concise global ABI, packed-layout, dispatch, numerical-tolerance, and test
  decisions not already in `DECISIONS.md`;
- unresolved failures/blockers with the exact next diagnostic command;
- the exact next source action. No prose history and no pasted raw output.

### 4.2 Atomic one-file loop

Process exactly one **primary manifest path** per loop:

1. **Resume guard.** Read `CURRENT.md`, the target manifest row, the latest
  family summary, and only the cross-file decisions relevant to the target.
  Verify the recorded worktree hash and baseline before editing. Resolve any
  mismatch; never assume old context is current.
2. **Activate one row.** Set only that row to `active`. Record its original
  SHA-256. Do not activate or deeply inspect the next file yet.
3. **Bounded discovery.** Read the primary file in full. Then read only its
  immediate source/header companion, registration site, focused tests, and the
  exact MLAS/KleidiAI leaf reached by its hot path. Use symbol/reference search
  rather than loading whole directories. Record the operator contracts,
  ownership, threading, allocation, dtype/opset variants, and call/dispatch
  chain in the milestone draft.
4. **Baseline and lower bound.** Capture the smallest focused baseline needed
  for this file: correctness output, allocations/copies/passes, relevant
  benchmark shape, and generated Arm64 path where available. State a
  falsifiable optimization hypothesis and compulsory memory traffic.
5. **Edit the primary file.** Make a non-cosmetic behavior-preserving refactor.
  Companion/dependency/test edits are allowed only when required for a complete
  change. Prefer removing work, hoisting preparation, prepacking, direct final
  stores, fused epilogues, compact prepared metadata, vectorized MLAS/NEON, and
  reusable scratch over superficial local cleanup.
6. **Validate before advancing.** Build the smallest affected target, run all
  focused tests plus new boundary/differential cases, inspect errors, and fix
  the change. Run sanitizer or architecture checks when the edit can affect
  bounds, aliasing, overflow, concurrency, or dispatch. The row remains
  `edited_unverified` until every locally available gate passes.
7. **Prove the exact file changed.** Save the accepted diff to
  `diffs/M####.patch`, record before/after hashes, and reject the milestone if
  the primary path has only comments, formatting, includes, or renames.
8. **Checkpoint and compact.** Write one `milestones/M####_<slug>.md`, update
  `MANIFEST.tsv`, `DECISIONS.md` if needed, and replace `CURRENT.md` with the
  bounded resume capsule. Only then set the row to `verified` and select the
  next deterministic path.
9. **Context barrier.** After the checkpoint is durable, stop carrying raw file
  contents, command output, test traces, and abandoned hypotheses forward.
  Continue the next loop from the compact state files, not from conversational
  recollection.

A companion file touched during a milestone remains `pending` unless it is the
primary path of its own completed loop. When it later becomes primary, inspect
its existing diff as input, improve or validate it independently, and write its
own checkpoint. Never mark multiple rows verified with one generic note.

### 4.3 Mandatory milestone capsule

Each per-file milestone checkpoint MUST be concise (target: 4–8 KiB; hard cap:
12 KiB / 200 lines) and contain:

1. Sequence, path, group, before/after hashes, and direct diff summary.
2. Operator/schema/dtype/opset/shape/optional-I/O contracts affected.
3. Exact call chain to the compute leaf and runtime dispatch predicates.
4. Waste found, theoretical memory-traffic lower bound, and chosen refactor.
5. Mandatory width audit: Q4/Q8/Q16/Q32 opportunities, proven ranges and chosen
  types for payloads/accumulators/metadata and every hot loop index, narrower
  candidates rejected with evidence, and all remaining 64-bit work justified.
6. Every file touched and why; shared-runtime/other-EP isolation reasoning.
7. Exact build/test/benchmark/disassembly commands and pass/fail summaries;
  raw-output paths instead of pasted logs.
8. Remaining risk, deferred device-only evidence, and rollback condition.
9. Exact next manifest path and first action for a fresh context.

At each subtree boundary, re-enumerate the authoritative corpus, relink the full
candidate `.so`, run the subtree test aggregate and C/C++ load smoke test, write
`families/F##_summary.md`, and compress `CURRENT.md` again. The family summary
must reference per-file milestones rather than duplicate them.

### 4.4 Context-window and output budget

- Keep at most one primary implementation plus its immediate companions active.
  Do not read all 195 files into one context and do not paste entire build logs.
- Save any raw output larger than 200 lines or 20 KiB under the milestone's
  `logs/` directory and retain only command, exit code, key metrics, and path in
  the capsule.
- At the start of a fresh context/session, read only this prompt,
  `CURRENT.md`, `MANIFEST.tsv` status/counts and next row, the latest family
  summary, relevant decisions, and the next file/dependencies. Do not replay all
  prior milestones.
- If context pressure appears before a file is verified, checkpoint it as
  `edited_unverified` with exact resume commands; never guess, truncate a test,
  or falsely mark it verified. Resume that same row first in a fresh context.
- A context boundary is not task completion. Report `CONTINUATION REQUIRED`,
  preserve state, and continue from the capsule in the next invocation. Never
  use context exhaustion as a reason to summarize untouched files as complete.
- If subagents are available, use them as stateless per-file investigators or
  validators. Their only durable output is a concise result copied into the
  current milestone; the controller must not accumulate their raw transcripts.

Keep baseline and candidate in separate output roots with identical
SDK/NDK/compiler/flags. Never reset or clean the user's workspace, and do not
let baseline/candidate reuse stale objects unless cache correctness is proven.

## 5. Deep Optimization Playbook for Contrib CPU Files

Concrete techniques for each Section 2 subtree follow. Apply the relevant ones
to the current primary file only, then checkpoint before moving on. Combine
them with the narrow-width (Section 6), memory-traffic (Section 7), and
prepared-execution (Section 8) contracts. Preserve every dtype/opset and generic
fallback.

**BERT, attention, and transformers** (`bert/`, `transformers/`, root attention
files). Prepack Q/K/V/output and projection weights once; fuse legal bias,
scale, rotary, mask, Softmax, and residual epilogues without materializing
intermediate full tensors. Write present K/V directly into final cache layout;
never concatenate a growing cache through a full copy. Split prefill (`M>1`)
and token decode (`M==1`) scheduling/tiling where reuse differs. Reuse
per-request scratch, compact validated head/stride metadata, avoid rebuilding
mask/position tables, and retain generic mask, past/present, packed/unpacked,
GQA/MQA/MHA, causal/non-causal, dtype, and empty-sequence paths.

**Quantization** (`quantization/` plus quantized root files). Follow Section 0
for Q4/Q8/Q16/Q32. Pack immutable weights once, fuse scale/zero-point
correction and requantization, keep INT32/FP32 accumulators in registers, and
eliminate repeated activation scans, whole-weight dequantization, temporary
transposes, and per-run backend selection. Preserve exact saturation, rounding,
asymmetry, block-tail, and fallback semantics.

**Mixture of experts** (`moe/`). Keep gating/top-k results compact, partition
tokens once, group work by expert without repeated full-tensor permutation, and
write outputs directly back to token order. Prepack expert matrices, reuse
dispatch metadata/scratch, batch tiny experts without oversubscription, and
preserve ties, capacity, normalization, sparse routing, empty-expert, and dtype
contracts.

**Attention LSTM** (`attnlstm/`). Prepack recurrent/input gate matrices, fuse
legal gate bias/activation work, retain state in reusable contiguous scratch,
and avoid allocation or shape reconstruction per timestep. Preserve sequence
length, direction, clipping, peephole/optional state, activation, and empty
sequence behavior.

**Tensor, sparse, ATen, math, and utilities** (`tensor/`, `sparse/`,
`aten_ops/`, `math/`, `utils/`). Make metadata-only transforms true views where
the ORT aliasing contract permits; otherwise stream directly once. Precompute
strides/offsets, vectorize contiguous inner spans, avoid broadcast/index
materialization, keep sparse index/value traversal cache-linear, and move
validation/conversion out of inner loops. Utility/header refactors must create
compact reusable prepared state or remove duplicated work for their callers,
not merely rename helpers.

**Root-level fused and registration files.** Trace every registration to its
template/implementation and optimize the implementation path while keeping the
registration table compile-time/static. Hoist immutable attributes, centralize
shared prepared data without introducing virtual dispatch in inner loops, and
keep fused operators from materializing values that their constituent kernels
could pass in registers or final layout. Even thin registration units require a
meaningful direct refactor and their own compile/registration coverage.

**Element-wise & activation** (`math/element_wise_ops`,
`activation/activations`, `element_wise_ranged_transform.h`). Bandwidth-bound —
target one input read and one output write per element. Ensure each
`operator()(first, last)` range functor vectorizes (Eigen array map or NEON),
fuses bias/scale/activation epilogues, and runs the same code for contiguous and
legal-strided views without materializing broadcasts. Add native FP16 paths
(route to `Mlas*Fp16`/`*_neon_fp16`: `gelu_neon_fp16`, `erf_neon_fp16`,
`eltwise_kernel_neon_fp16`, `silu`, `logistic`, `tanh`). Tune the `Cost()`
estimate so the thread pool does not fragment tiny tensors. Prefer in-place when
aliasing is legal.

**GEMM / MatMul / FC** (`math/gemm`, `math/matmul`,
`contrib_ops/.../MatMulNBits`, `mlas/lib/sgemm`, `halfgemm`,
`hgemm_kernel_neon`). Route to `MlasGemm` and **prepack the constant B matrix
once** in `PrePack()`, never per `Compute()`. Pick M/N/K blocking and
packed-panel sizes for the register file and L1/L2 budget of the target µarch;
reuse A/B panels across the largest legal output tile; fuse bias + activation
into the epilogue. For `arm64-v8a` prefer the FP16 `hgemm` path and, for
quantized weights, the dot-product / I8MM `qgemm`/`sqnbitgemm` kernels — all
runtime-gated. Never add a full-tensor FP32↔FP16 pass just to reach a narrow
kernel; keep narrow formats across the packed weights and epilogue.

**Conv / ConvTranspose / Pool** (`nn/conv`, `nn/conv_transpose`, `nn/pool`,
`mlas/lib/convolve`, `conv`, `dwconv`, `pooling`, `pooling_fp16`,
`sconv_nchwc_kernel_neon`). Prefer NCHWc/packed layouts and MLAS conv over an
im2col+GEMM that materializes a large column buffer; when im2col is unavoidable,
block it to cache and reuse. Route depthwise conv to `dwconv`/`qdwconv`. Prepack
filters. Add the FP16 conv/pool path. For pooling, keep a single pass with
register-resident accumulators.

**Softmax / Norm / Reduction** (`math/softmax`, `nn/layer_norm`, `nn/rms_norm`,
`nn/instance_norm`, `nn/batch_norm`, `reduction/reduction_ops`,
`mlas/lib/softmax_kernel_neon`, `layernorm`). Multi-pass reductions — fuse the
passes: Softmax max→exp-sum→normalize should reuse cached values and not re-read
the row from DRAM; LayerNorm/RMSNorm compute mean/var and normalize with FP32
accumulation over FP16 storage. Route to `MlasComputeSoftmax`/
`MlasComputeLogSoftmax` and the NEON/FP16 softmax kernels. For reductions,
collapse contiguous reduced axes into one loop, keep the accumulator in a
register / FP32, and vectorize ArgMax/ArgMin while preserving tie-breaks. Never
reduce a float32 contract below its tolerance.

**Quantization** (`quantization/*`, `contrib_ops/.../quantization`,
`mlas/lib/qgemm`, `qnbitgemm`, `sqnbitgemm*`, `quantize`, `dequantize`, `qladd`,
`qlmul`). Pack quantized weights once; keep INT8/INT16 operands packed through
compute with INT32 accumulation (16-bit accumulate only when term count × value
range provably cannot overflow). Fuse scale/zero-point correction and
requantization into the packing or epilogue — do not add a separate dequantize
pass. Route to dot-product (`*_dot`, `qgemm_kernel_sdot`) and I8MM
(`qgemm_kernel_smmla`, `*_i8mm`) kernels, runtime-gated. Preserve exact
integer/saturation/rounding semantics.

**Tensor movement & shape** (`tensor/*`). Most of these should move *no* data:
`reshape`/`squeeze`/`unsqueeze`/`flatten`/`identity`/`bitcast` are metadata-only
— ensure they stay views, not copies. `transpose` should use blocked/tiled
`MlasTranspose` and coalesce mergeable axes; `concat`/`split` should write each
element once directly into/out of the joined buffer; `slice`/`gather`/
`gather_nd`/`scatter_nd` should precompute strides/offsets once and vectorize the
contiguous inner dimension; `pad`/`tile`/`expand` should stream without a
temporary; `cast` should vectorize and fuse into a neighboring producer/consumer
when legal. `where`/`compress`/`nonzero`/`onehot` should hoist index math out of
the inner loop.

**Gather / Scatter / Index** (`tensor/gather*`, `scatter*`, `math/top_k`,
`generator/*`). Hoist invariant stride/bound computation to kernel creation;
keep bounds checks but prove them once; vectorize the contiguous copy per
gathered row; parallelize across rows with contiguous, false-sharing-free output
partitions.

**Control flow, RNN, sequence, ML, detection, signal, text** (`controlflow/`,
`rnn/`, `sequence/`, `ml/`, `object_detection/`, `signal/`, `text/`).
RNN/LSTM/GRU: route gate matmuls to prepacked `MlasGemm`, fuse the element-wise
gate activations, reuse per-timestep scratch. `ml/` TreeEnsemble: keep tree
tables cache-resident and branch-lean; batch rows. `object_detection`
RoiAlign/NMS: hoist box math, vectorize bilinear sampling. `signal` DFT/STFT:
prepack twiddle/window; avoid per-call table rebuild. `If`/`Loop`/`Scan`:
minimize per-iteration allocation and reuse subgraph I/O buffers. Even "cold"
families owe an edit-or-justify decision.

When a contrib kernel delegates to MLAS/KleidiAI, follow the dispatch into the
leaf and optimize its packing, workspace, tail, epilogue, or dispatch when that
is where the waste lives. This supporting edit is required evidence but does
not replace a meaningful direct refactor and checkpoint for the current contrib
`*.cc` or `*.h`. A bare "delegates to MLAS" is not a lower-bound proof.

## 6. Armv8.2-A FP16 and Narrow-Width Arithmetic Contract

Make an Armv8.2-A path with FP16 Advanced SIMD the primary optimization target
for production `arm64-v8a`. Apply the topmost Q4→Q8→Q16→Q32→64-bit-last ladder
to data, compute, prepared metadata, and control variables. Retain 64-bit work
only where a public ABI/schema, pointer/object/allocation size, tensor range,
accumulator bound, atomic/counter, or external API proves it necessary.

- Audit hot-path `size_t`/`ptrdiff_t`/`int64_t`/`uint64_t`/`long`/`double`,
  64-bit literals/casts, and emitted 64-bit arithmetic or loads/stores. Validate
  dynamic dims/strides/offsets/counts/byte-sizes once with checked wide
  arithmetic at the preparation boundary, then narrow **once** to the smallest
  proven 8/16/32-bit immutable representation used throughout the hot path; keep
  the checked wide fallback for larger valid tensors.
- Audit **every ordinary `for`, `while`, and tile-loop induction variable**.
  Never default to `size_t`, `auto` deduced from `.size()`, or `int64_t` in a
  normal loop. Prove the upper bound once, then try `uint8_t`/`int8_t`, followed
  by `uint16_t`/`int16_t`, followed by `uint32_t`/`int32_t`; use 64-bit induction
  only in the checked wide fallback. Q4 is packed payload-only because C++ has
  no native addressable 4-bit loop scalar. Prefer signed types for values that
  can be negative and perform subtraction in the signed type to prevent
  unsigned wraparound.
- A smaller loop type is accepted only if generated AArch64 code does not add
  enough repeated extend/mask/spill instructions to lose the benefit. If 8- or
  16-bit induction is slower because C++ integer promotion or address formation
  repeatedly widens it, record the disassembly/measurement and select 32-bit—
  never silently fall back to 64-bit.
- Use 8/16-bit fields, extents, axes, tile coordinates, thread indexes, and
  masks whenever their complete range fits and packing/lane-density gains exceed
  extension cost; otherwise use 32-bit prepared values. Convert to `size_t` or
  `uintptr_t` only at the final API/allocation/address boundary. AArch64 pointers
  remain 64-bit and MUST NOT be truncated; compulsory pointer address generation
  is not avoidable 64-bit data/control processing.
- Avoid FP64/`double` constants in FP16/FP32 kernels; use `float`, fixed-width
  integer constants, and typed intrinsics so literals do not silently promote.
- Prefer FP16 storage + native FP16 NEON arithmetic where the contract allows,
  but accumulate FP16/BF16 dot products, reductions, normalization, and Softmax
  in FP32 unless an explicit error bound + differential test prove FP16
  accumulation safe. Never reduce a float32/double contract past its tolerance.
- Prefer packed INT8/INT16 + INT32 accumulation for quantized work (16-bit
  accumulate only when overflow is provably impossible). No signed overflow,
  wrapping, lossy shape narrowing, or unrequired saturation.
- Account for conversion cost: do not add a full-tensor FP32↔FP16 or 64→32-bit
  pass just to reach a narrow microkernel — preserve narrow formats across
  producers, packed weights, scratch, compute, and consumers, or fuse conversion
  into a compulsory load/store epilogue.

**Feature gating.** Armv8.2-A alone does not imply every extension, but this
task's documented deployment floor separately guarantees FP16 Advanced SIMD, so
`-march=armv8.2-a+fp16` is the production baseline. FP16-FML, dot-product,
I8MM, BF16, SVE, SVE2, SME, and SME2 are still independent optional features;
detect them from Android HWCAP/OS state once and dispatch through separate
translation units / target-attribute functions / the existing MLAS/KleidiAI
tables (`platform.cpp`, `HasArmNeonDot()`, `HasArmNeon_I8MM()`, and related
entries). Optional instructions MUST NOT leak into baseline initialization or a
feature-negative path. For critical kernels compare FP32,
FP16-storage/FP32-accumulate, native-FP16 compute, Q8-dot, and I8MM variants as
applicable, selecting on end-to-end latency, conversion bytes, accuracy,
thermal behavior, and code size rather than instruction width alone. The
selected path MUST remain correct across heterogeneous-core migration.

## 7. Memory-Traffic, Prepared Execution, and MLAS Routing

### Memory traffic, cache, and layout

For each kernel and shape class, derive compulsory input/weight/metadata/output
bytes, then drive actual bytes toward that lower bound:

- Draw the read/write dataflow and count every pass over every tensor. Remove
  redundant reads, write-allocate traffic, cache-line bouncing, and DRAM
  round-trips. Fuse adjacent element-wise/reduction/post-op passes when it cuts
  total traffic and stays numerically legal; keep multiple outputs in registers
  and store directly instead of materializing shared intermediates.
- Hot paths MUST have zero heap allocations and zero avoidable full-tensor
  temporaries. Per-thread scratch is preallocated, aligned, lifetime-planned,
  capacity-checked, and reused without cross-request mutation.
- Prefer contiguous, unit-stride, aligned access. Coalesce dimensions and
  broadcasts, remove layout round-trips, avoid gather/scatter when a legal
  packed/strided view exists, and pad shared metadata/partial outputs to prevent
  false sharing.
- Choose loop order, blocking, micro-tiles, packed panels, and thread partitions
  so the live working set fits the intended register file / cache level with
  associativity headroom. Record the byte budget and cache level per tuned tile.
- Keep immutable weights packed and cache-shareable; version packed formats and
  make backend/ISA/shape-class/alignment part of the cache key.
- Use software prefetch, streaming/non-temporal stores, or cache hints only when
  Android permits and counters show a repeatable win; never prefetch out of
  bounds or assume huge pages. Prefer producer final-layout stores and fused
  epilogues over materialized conversion operators. Treat NUMA only if the
  target exposes more than one memory node.

Default acceptance target: zero avoidable allocation/copy/Cast/repack/
materialization bytes in repeated execution. When zero is impossible under the
public contract, record the lower bound, the remaining compulsory bytes, and the
earliest stage the condition was proven.

### Prepared execution and dispatch hoisting

Review the full path and push work as early as it can be proven:

```text
graph → transform → registration → kernel creation
  → attribute/shape/type/ISA prep → pointer lookup → allocation
  → scheduling → compute → postprocess → publication
```

- Move invariant attribute parsing, axis normalization, shape-class and backend
  selection, tiling, thread partitioning, scratch sizing, and prepacking to
  kernel creation / `PrePack()` — not `Compute()`. Store the results in compact
  immutable POD records (direct kernel/microkernel target, indices/offsets,
  dims/strides/axes, loop/vector/tail bounds, broadcast/reduction geometry,
  thread partitions, packed weights, scratch offsets). Only run/request state and
  input-dependent values mutate at runtime.
- Classify shape/metadata subgraphs as static / session-static / run-static /
  input-dependent and fold or compute-once accordingly; do not run generic
  tensor ops merely to update metadata when a view is legal.
- Keep a correct generic path for every input a fast path does not cover, and
  test both the specialization boundary and the fallback.

### MLAS / KleidiAI routing

- Prefer routing compute-heavy kernels through MLAS (`MlasGemm`, `MlasConv`,
  `MlasComputeSoftmax`, `MlasTranspose`, quantized `MlasQgemm`/`MlasSQNBitGemm`,
  FP16 half-GEMM/conv) rather than hand-rolled scalar loops — then tune the MLAS
  path itself (packing, tail loops, µarch blocking, runtime feature dispatch in
  `platform.cpp`).
- When adding a new SIMD variant, follow the existing MLAS structure: a dispatch
  entry in `platform.cpp`, a feature-gated kernel translation unit (`*_neon`,
  `*_fp16`, `*_dot`, `*_i8mm`, `*_sve`), and a scalar reference. Keep KleidiAI
  usage behind its existing integration and gate.
- Prepack reusable constants once with explicit ownership/invalidation; fuse
  bias/scale/activation/requantization/final-layout store into the epilogue when
  semantics allow.

### Memory planning, threading, and I/O

- Preplan reusable activations, outputs, per-thread scratch, packed weights, and
  persistent state where lifetimes/bounds are known; grow slabs geometrically
  with a measured cap; never request arbitrary physically-contiguous memory.
- Tune `OrtArenaCfg`, memory patterns, chunk size, and I/O binding from measured
  lifetimes; reuse buffers after warmup without retaining a peak that worsens
  steady-state PSS/RSS.
- Select sequential vs. parallel execution by reusable work/shape class; skip
  thread-pool submission below the measured scheduling cost; avoid inter-/
  intra-op oversubscription; assign cache-line-aligned, contiguous,
  cluster-local per-thread regions; prefer caller-owned `OrtValue` and direct
  final-output binding where the C API allows.

## 8. Android NDK Build and ELF Verification

Rebuild `libonnxruntime.so` after each batch of kernel edits. **Build with the
toolchain already installed on the local host — the latest locally-present
Android NDK, plus the host CMake and Ninja — and do NOT download or install any
duplicate toolchain.** Reuse the existing Android SDK/NDK, the `cmake`/`ninja`
already on `PATH`, and any existing build cache; never let the build fetch a
second NDK/CMake/Ninja or run an `sdkmanager` re-install. If no NDK is installed
at all, report it as a blocker (Section 12) instead of downloading one silently.

Discover and pin the local toolchain once (in the build shell), then reuse it for
both the baseline and candidate builds — do this instead of hardcoding any
version:

```bash
# Android SDK root: respect the environment, else the common local default.
export ANDROID_SDK_ROOT="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
# Latest NDK already installed under $SDK/ndk (highest version) — no new download.
export ANDROID_NDK_HOME="$ANDROID_SDK_ROOT/ndk/$(ls -1 "$ANDROID_SDK_ROOT/ndk" | sort -V | tail -1)"
# Host CMake + Ninja already on PATH (the latest ones installed locally).
CMAKE_BIN="$(command -v cmake)"; NINJA_BIN="$(command -v ninja)"
"$CMAKE_BIN" --version; "$NINJA_BIN" --version; echo "NDK=$ANDROID_NDK_HOME"
```

If the host CMake does not satisfy ORT's `cmake_minimum_required`, fall back to
the newest CMake already bundled under `$ANDROID_SDK_ROOT/cmake/*` (still no
download); apply the same reuse-what-exists rule to Ninja.

Build baseline and candidate from separate source snapshots into separate output
roots; optional LTO / reduced-operator / exception / RTTI / backend flags MUST be
identical between A/B builds unless the flag itself is the measured optimization.
Every native build uses **exactly four workers** (`--parallel 4` +
`CMAKE_BUILD_PARALLEL_LEVEL=4`; if Gradle only packages the app,
`--max-workers=4`, and it MUST NOT trigger an ORT Java build).

The production contract in this task explicitly guarantees Armv8.2-A with FP16
Advanced SIMD. Apply that minimum to baseline and candidate identically. Do not
add dot-product, I8MM, BF16, SVE, or SVE2 to the global `-march`; those remain
runtime-dispatched optional paths. Keep the build's compile database and verify
the intended flags on the actual leaf-kernel translation units rather than
trusting a configure message.

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=4
export ANDROID_ABI=arm64-v8a
export CFLAGS="${CFLAGS:+$CFLAGS }-march=armv8.2-a+fp16"
export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-march=armv8.2-a+fp16"
python3 tools/ci_build/build.py \
  --config Release \
  --build_dir "$BUILD_DIR" \
  --android \
  --android_abi arm64-v8a \
  --android_api "$ANDROID_API" \
  --android_sdk_path "$ANDROID_HOME" \
  --android_ndk_path "$ANDROID_NDK_HOME" \
  --cmake_path "$CMAKE_BIN" \
  --cmake_generator Ninja \
  --build_shared_lib \
  --cmake_extra_defines onnxruntime_USE_KLEIDIAI=ON \
  --cmake_extra_defines onnxruntime_ENABLE_CPU_FP16_OPS=ON \
  --parallel 4 \
  --update \
  --build
```

After configure, prove from `CMakeCache.txt`, `compile_commands.json`, linked
objects, and runtime dispatch logs that KleidiAI and CPU FP16 are compiled and
that the intended Q4/Q8/FP16 kernels are reachable. A flag that never reaches a
selected object is not an optimization.

`--build_java` is forbidden; do not build/package `libonnxruntime4j_jni.so` or
any JAR/Maven/AAR. Required output: `$BUILD_DIR/Release/libonnxruntime.so` plus
matching generated/public C/C++ headers. Ship the stripped library; keep an
unstripped copy for profiling. Build only the in-scope Android ABI,
`arm64-v8a`; do NOT build Android `x86_64`, `armeabi-v7a`, or 32-bit `x86`.
During iteration, rebuild just the affected target for speed (for example,
`onnxruntime_mlas`, `onnxruntime_providers`, or the focused test target), but
re-link the full `.so` before any acceptance claim. Rerun `--update` only when
files are added or CMake changes.
`--android_sdk_path`/`--android_ndk_path`/`--cmake_path` point at the
already-installed local toolchain and `--cmake_generator Ninja` uses the `ninja`
on `PATH`, so the build reuses local tools and never fetches a duplicate.

Integrate directly via a CMake `IMPORTED` target / `find_package` / ndk-build
rule; package exactly one runtime per ABI; pull in no second ORT dependency.
Verify the APK/AAB by extraction, and confirm the running process maps the
intended candidate, not a stale duplicate.

Validate each ELF with NDK `llvm-readelf` / `llvm-nm` / `llvm-objdump`:

- ELF class/machine/ABI match the destination and device.
- SONAME is `libonnxruntime.so`; `OrtGetApiBase` and the public API stay
  exported; the baseline public `OrtApi` ABI is not silently changed.
- `DT_NEEDED` lists only intended Android/C++ runtime deps — no stale
  RPATH/RUNPATH, host lib, Java wrapper, or second ORT runtime.
- API floor matches app `minSdk`; load succeeds on the lowest supported API;
  headers report the same ORT version as the binary.
- `PT_LOAD` alignment and APK packaging support Android 16 KiB *and* 4 KiB
  page-size devices.
- Record stripped/unstripped SHA-256, build ID, section sizes,
  exported-symbol count, relocations, and APK contribution before/after.

The declared deployment floor guarantees AArch64 NEON and FP16 Advanced SIMD.
Dot-product, I8MM, BF16, SVE, SVE2, SME, and SME2 remain optional: detect them
once from Android HWCAP/OS state and keep their instructions confined to the
existing dispatched translation units so feature-negative cores remain
`SIGILL`-safe across big/middle/little migration. Never infer an optional ISA
from a phone/SoC name. If the final app must support a CPU below the declared
Armv8.2-A+FP16 floor, remove the global baseline flag and restore runtime-gated
FP16 before shipping; silently producing an incompatible binary is forbidden.
For critical kernels, inspect production Clang output (vector/matrix
instructions, loads/stores, spills, tails, branches, I-cache footprint) to
confirm the intended tile stays register/cache resident and no spill or
code-size growth erased the gain. Keep a tested portable fallback for every
CPU/contract an optimized path does not cover. Apply fast math only when every
affected numerical contract stays within tolerance. Build a reduced-operator
config only from the full deployed model corpus, validate every model, and never
present it as full-contract evidence.

## 9. Correctness Verification — the Non-Negotiable Floor

Correctness is the floor every edit clears *before* you move to the next file; it
is not the deliverable and it does not replace the optimization work. This floor
is always achievable without a phone, so "no device" never excuses skipping it.

For each primary manifest file—including declaration/template headers—validate
the resulting implementation path against the pre-edit baseline:

- Run the operator's unit tests across every affected opset, dtype, rank, shape
  class, attribute mode, optional input/output, layout, scalar/vector tail,
  backend, and thread mode. Add generated boundary + randomized differential
  cases where existing tests do not cover the registered contract.
- Compare with contract-appropriate metrics: max/mean absolute and relative
  error and ULP where useful; exact equality for integer/index/shape/string/
  sequence/control-flow/deterministic outputs; cosine/top-k where meaningful.
- Narrow-width edits MUST test values just below/at/above the 8-, 16-, and
  32-bit signed/unsigned limits, plus transition into the checked 64-bit
  fallback; negative axes/sentinels; checked count/byte-size multiplication;
  large logical shapes without unsafe allocation; and zero/one/odd tails. Test
  normal loop-index boundaries explicitly to catch wrap, nontermination, and
  tail loss. FP4/FP8/FP16/BF16 edits MUST cover each exact registered encoding,
  signed zero where representable, subnormals, min/max finite, ±inf/NaN where
  defined, saturation/rounding boundaries, and long accumulations.
- Cover alias overlap, in-place legality, alignment, cache-line/page boundaries,
  scratch-capacity limits, thread partitions, false-sharing, and deterministic
  init of logical and padding bytes.

For the blocking Section 0 stack, the minimum focused source tests and benches
to build and run are:

- `onnxruntime/test/contrib_ops/matmul_4bits_test.cc`
- `onnxruntime/test/contrib_ops/matmul_8bits_test.cc`
- `onnxruntime/test/contrib_ops/matmul_integer16_test.cc`
- `onnxruntime/test/contrib_ops/matmul_nbits_prepack_sharing_test_util.cc`
- `onnxruntime/test/mlas/unittest/test_sqnbitgemm.cpp`
- `onnxruntime/test/mlas/unittest/test_sqnbitgemm_neon_fp16.cpp`
- `onnxruntime/test/mlas/unittest/test_hqnbitgemm_neon.cpp`
- `onnxruntime/test/mlas/unittest/test_sq8bitgemm.cpp`
- `onnxruntime/test/mlas/unittest/test_q8q4gemm.cpp`
- `onnxruntime/test/mlas/unittest/test_qgemm.cpp`
- `onnxruntime/test/mlas/unittest/test_dynamic_qgemm.cpp`
- `onnxruntime/test/mlas/unittest/test_halfgemm.cpp`
- `onnxruntime/test/mlas/bench/bench_qnbitgemm.cpp`
- `onnxruntime/test/mlas/bench/bench_q4gemm.cpp`
- `onnxruntime/test/mlas/bench/bench_qgemm.cpp`

Add boundary cases for `M=1` and `M>1`, `N/K` below/equal/above every micro-tile,
block lengths 16/32/64/128/256 where supported, symmetric/asymmetric zero
points, absent/present bias, batched inputs, odd tails, empty tensors, all
registered FP16/FP32 inputs, every supported compute type, one thread and the
configured thread pool, prepacked and unpacked fallback, and feature-negative
dispatch. Benchmarks do not replace tests and must not contain the only copy of
production logic.

Use a validation build with assertions, bounds/alias checks, and sanitizers
(ASan/UBSan where available) to catch OOB, use-after-free, uninitialized output,
bad aliasing, integer overflow, races, and cross-request mutation. Only produce
the stripped production `.so` after the validation config passes.

For Android `arm64-v8a`, compile native C and C++ integration harnesses against
the delivered headers, link the delivered `libonnxruntime.so`, and verify load,
`OrtGetApiBase`, `ORT_API_VERSION`, env/session creation, model load,
caller-owned tensors, CPU inference, checked outputs, teardown, and error paths.
Run on the emulator/host for functional parity and on a device when one exists;
header compilation or ELF inspection alone is insufficient. Run the
full-contract tests (and every deployed model) before any reduced artifact.

**EP isolation check.** For every shared-runtime edit, state why other EPs are
unaffected and back it with tests; compilation alone is not proof. When
measuring CPU behavior, keep NNAPI/XNNPACK/QNN/other EPs from claiming nodes and
confirm CPU placement from logs/profiling; if production uses another EP, run
that path separately and never attribute changed partitioning to a CPU-kernel
edit.

Validate as you go — do not batch all correctness to the end. A failing floor
means the edit is wrong: fix or revert that edit, never weaken the test.

## 10. Acceptance Gate and Optional Device Benchmarking

Each optimization is accepted only when it clears the gate below. Split the gate
into an **always-required floor** (achievable with no device) and
**device-measured gates** (required only when a physical target exists).

Always required for every edit:

- Correctness within tolerance (Section 9) and no lost operator/opset/dtype/ABI/
  fallback coverage.
- The full `.so` rebuilds with four workers; ELF/ABI/SONAME/public-API/page-size
  checks pass; no Java/JNI/JAR/Maven/AAR is produced.
- No behavioral change to another EP.
- By code review + local analysis: no new unjustified allocation, copy,
  full-tensor Cast, materialization, or repeated dispatch in the hot path; the
  change is a real removal / fusion / hoist / vectorization, not a faster version
  of avoidable work.
- On `arm64-v8a`, critical-kernel disassembly shows the selected packed Q4,
  Q8, Q16, or Q32 path and the smallest profitable fixed-width loop induction
  variables, with runtime-gated FP16/NEON/dot/I8MM instructions as applicable.
  There is no avoidable FP64, 64-bit mul/div, 64-bit metadata, or 64-bit normal
  loop counter in the inner path; narrowing introduces no truncation,
  wraparound, overflow, ABI change, or loss of the checked wide fallback.
- The primary file is `verified` in `MANIFEST.tsv` with its Section 4 milestone,
  accepted patch, hashes, and raw-evidence paths present.

Device-measured gates (when a physical target is connected — otherwise record as
an explicit blocker with resume commands; do NOT drop the source optimization):

- Build and A/B baseline vs. candidate on the *same* device with identical
  compiler/runtime/affinity/threading, alternating runs, rejecting thermally or
  environmentally invalid samples, reporting confidence intervals.
- No repeatable end-to-end or per-operator regression above 1% on any affected
  contract / µarch / shape / thread class (a gain on one CPU may not hide a
  regression on another) unless an explicit measured system-level tradeoff is
  recorded.
- Actual memory traffic moves toward the derived lower bound; a compute win paid
  for with more DRAM/LLC traffic, writebacks, or peak RSS needs measured
  approval.
- At least one predeclared primary metric improves by a statistically supported
  amount while all gates pass; a CI crossing zero is inconclusive. Report exact
  absolute and percentage deltas even for rejected changes.

Per-edit workflow: (1) identify affected registrations/contracts/paths; (2) note
the baseline behavior and the theoretical lower bound; (3) state a falsifiable
hypothesis; (4) implement the smallest complete reusable optimization with the
generic fallback intact; (5) rebuild with four workers and inspect critical
disassembly; (6) run correctness + sanitizer + C/C++ load tests; (7) if a device
exists, A/B it; (8) accept or discard only the agent's candidate change on the
evidence without disturbing baseline user edits; (9) update the manifest,
checkpoint, compact context, and keep patches as independent logical changes.
**Do not create git commits unless explicitly requested.**

Reject superficial changes: making avoidable copy/Cast/dispatch/allocation
faster, unproven prefetch/threading/assembly, retained repeated fast-path
checks, tuning to one model identity or fixed dimension, or regressing another
valid contract. A change within variance is inconclusive, not a win — but its
*source-level* removal of avoidable work still stands on the always-required
floor.

## 11. Deliverables

The primary deliverable is **a validated refactor of every manifest `*.cc` and
`*.h` under `onnxruntime/contrib_ops/cpu`, plus a rebuilt
`libonnxruntime.so`.** Reports and state capsules describe that work; they do not
replace it. Deliver:

- The edited contrib CPU source/header corpus as sequential, independently
  validated logical patches (no git commits unless requested), each preserving
  coverage and semantics.
- One Android Arm bundle: matching `include/` headers,
  `libs/arm64-v8a/libonnxruntime.so` (stripped), and
  `symbols/arm64-v8a/` unstripped/debug artifacts, with source/build/binary
  manifests and checksums. No ORT Java/JNI library, JAR, Maven, or AAR.
- Baseline and candidate full-contract libraries (and, optionally, a separately
  named reduced library justified by the full model corpus).
- `MANIFEST.tsv`, `CURRENT.md`, per-file milestone capsules, family summaries,
  raw-log references, and accepted milestone patches from Section 4, so all
  195-current-file coverage and continuation state are auditable.
- Reproducible build/test commands (the four-worker build, unit-test
  invocations, ELF checks) and, where a device exists, the A/B harness with its
  raw + summary numbers.
- A concise summary report: aggregate before/after where measured, per-family
  optimization status, the EP-isolation argument, ELF/packaging verification, and
  an explicit list of remaining opportunities and coverage gaps.

Keep the summary proportional — a short, accurate status beats a large report
padded with `not_measured` placeholders. Every device-only deferral names the
specific external blocker and resume commands. Source rows remain incomplete
until their direct refactor and local validation pass.

## 12. Execution Order, Manifest, and Completion

Execute in this order, and start editing kernels early — do not spend the run on
discovery:

1. Fingerprint the worktree and preserve a reconstructable baseline (no reset,
  clean, checkout over, or overwrite of user changes) into `BASELINE.md`.
2. Create the durable state directory, enumerate the authoritative recursive
  `*.cc`/`*.h` set, verify unique paths/counts, and initialize every
  `MANIFEST.tsv` row as `pending`. This mechanical inventory must be quick; do
  not read all file contents.
3. Build the Arm64 baseline `.so` with the Section 8 four-worker command; record
  source/toolchain/ELF checksums and run the existing contrib CPU tests once.
4. If the worktree already contains an unfinished in-scope edit, make that path
  the sole `active` row and finish it first. Otherwise begin
  `quantization/matmul_nbits.cc`, trace Section 0's full call chain, and perform
  the first one-file transaction.
5. After **every primary file**, validate, save raw evidence, write its milestone
  capsule/diff, update manifest hashes/status, replace `CURRENT.md`, and cross
  the context barrier before reading the next primary file.
6. Continue in Section 2's deterministic group/path order. Do not create a huge
  multi-file rewrite and defer all builds/tests/checkpoints to the end.
7. At every subtree boundary, re-enumerate, reconcile new files, run aggregate
  tests, relink and inspect the full candidate `.so`, write the family summary,
  and compact context. If a device is connected, run the applicable A/B gates.
8. At the end, re-enumerate from scratch and compare the sorted authoritative
  set to unique manifest paths; require every row `verified`, every milestone
  and primary-file diff present, and no pending/active/edited-unverified row.
9. Run final full-contract correctness, C/C++ load, ELF/API/page-size, and
  packaging checks, then assemble the Arm64 deliverables in Section 11.

**Anti-stop completion semantics.** The task is complete only when every
authoritative contrib CPU `*.cc` and `*.h` row is `verified` with a meaningful
direct source diff and individual checkpoint, the final manifest equals the
fresh filesystem enumeration, the full-contract Arm64 `.so` rebuilds and passes
correctness + ELF/API checks, coverage and semantics are intact, and EP
isolation holds. The following DO NOT constitute completion and MUST NOT be
reported as done:

- Producing an inventory, ledger, coverage table, or plan.
- Optimizing hot operators, Q4/MatMulNBits, one subtree, or a benchmark model
  only.
- The build merely succeeding, or the app merely getting faster.
- "Verifying" the kernels without editing them.
- Editing only `.cc` files while leaving headers untouched, or changing one
  shared MLAS dependency while marking multiple contrib rows complete.
- Reaching a context limit and emitting a summary without a durable resume
  capsule and explicit `CONTINUATION REQUIRED` status.

While any row is `pending`, `active`, or `edited_unverified`, status is
**incomplete** and autonomous work MUST continue from the exact next action in
`CURRENT.md`. A missing device may defer only hardware measurements. A missing
SDK/NDK or permission must be recorded with owner/input and exact resume
commands, but the agent must continue source work and host-validatable tests in
other rows. Do not stop because major hotspots are fast or a threshold was hit.
Pause only at a durable file checkpoint; resume the unfinished row before
starting another.
