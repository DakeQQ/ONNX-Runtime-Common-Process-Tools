## Agent Task: Adapt Fresh ONNX Runtime Source for Shared Weights and Fast Cross-Session Prepacking

You are modifying a fresh ONNX Runtime source tree, using ONNX Runtime v1.27.1 as the reference version.

Locate code by symbols and control flow rather than fixed line numbers. Implement the changes directly; do not stop at analysis.

Caller contract:

- The application creates one process-global OrtPrepackedWeightsContainer.
- The same container is passed to every relevant session through
  CreateSessionWithPrepackedWeightsContainer or
  CreateSessionFromArrayWithPrepackedWeightsContainer.
- Shared model weights are mapped once and exposed as process-global OrtValue objects.
- Every session receives the same OrtValue instances through SessionOptions::AddInitializer.
- The shared initializer backing memory, OrtValue objects, and prepacked container outlive all sessions.
- Relevant model operators are MatMulNBits with 4-bit weights, block size 32,
  accuracy level 4, constant scales, and asymmetric constant zero points.
- The target runtime is CPU EP on Android ARM64, with KleidiAI enabled.
- Prepacking must remain enabled.

### Objectives

1. Support sharing the same raw model weights across multiple sessions through `AddInitializer`.
2. Support cross-session sharing of `MatMulNBits` packed buffers through `OrtPrepackedWeightsContainer`.
3. Fix the case where `MatMulNBits::PrePack` reports `is_packed=true` without returning a cacheable buffer.
4. Let the first session perform normal packing and content hashing.
5. Let later sessions safely reuse an already verified packed buffer before calling `PrePack`.
6. Preserve existing behavior for non-shared initializers, non-CPU execution providers, FP16 kernels, dynamic zero points, and packed formats that are not self-contained.

### Scope

Modify only these files:

- `include/onnxruntime/core/framework/op_kernel.h`
- `onnxruntime/core/framework/prepacked_weights_container.h`
- `onnxruntime/core/framework/prepacked_weights_container.cc`
- `onnxruntime/core/framework/session_state.cc`
- `onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc`

Do not modify CMake files, application code, tests, execution providers, or build scripts. No compilation or verification work is required.

### 1. Extend `OpKernel`

Add this virtual method to `OpKernel`:

```cpp
virtual std::string GetPrePackedWeightsSourceKey(
    const Tensor& tensor,
    int input_idx) const;
```

The default implementation must return an empty string.

Semantics:

- An empty key means the kernel cannot safely identify its packed result before `PrePack`; retain the existing pack-then-hash behavior.
- A non-empty key is process-local and may only describe immutable initializer buffers.
- Do not require any other kernel to override this method.

### 2. Extend `PrepackedWeightsContainer`

Add:

```cpp
bool TryGetWeightKeyForSource(
    const std::string& source_key,
    std::string& weight_key) const;

void WriteWeightKeyForSource(
    const std::string& source_key,
    const std::string& weight_key);
```

Add a map:

```cpp
std::unordered_map<std::string, std::string> source_to_weight_key_;
```

The source key must not replace the existing packed-content hash key. It is only an alias to a canonical packed-weight key that has already passed normal `PrePack` and content hashing.

`TryGetWeightKeyForSource` must verify both:

- The source alias exists.
- Its target still exists in `prepacked_weights_map_`.

These methods are called while `PrepackedWeightsContainer::mutex_` is already held. Do not add another mutex.

### 3. Add the Fast Path in `SessionState`

Locate `SessionState::PrepackConstantInitializedTensors`.

Only change the branch where:

- The initializer was supplied through `AddInitializer`.
- A `PrepackedWeightsContainer` is present.
- The node is assigned to `CPUExecutionProvider`.

Before calling `kernel->PrePack`:

1. Call `kernel->GetPrePackedWeightsSourceKey(tensor, input_idx)`.
2. If the key is non-empty and its source alias exists:
   - Obtain the canonical `PrePackedWeights`.
   - Call `KernelUseSharedPrePackedBuffers`.
   - Set `is_packed=true`.
   - Increment `used_shared_pre_packed_weights_counter_`.
   - Call `ReplaceWithReferenceIfSaving`.
   - Do not call `PrePack`.
3. On a miss:
   - Preserve the normal `PrePack` call.
   - Preserve packed-buffer content hashing.
   - Preserve the existing `HasWeight`, `GetWeight`, and `WriteWeight` behavior.
4. Publish the source alias only after normal packing, hashing, and container insertion or lookup have succeeded.
5. On an alias hit, leave `is_packed=true` so initializer use-count and release behavior remain identical to normal packing.
6. Do not change empty-key, non-shared, or non-CPU paths.

### 4. Repair the `MatMulNBits` Cache Contract

When `MatMulNBits::PrePack` successfully packs input B and `prepacked_weights` is non-null, move `packed_b_` and its size into:

```cpp
prepacked_weights->buffers_
prepacked_weights->buffer_sizes_
```

The shared container becomes the owner. The kernel must subsequently obtain a non-owning reference through `UseSharedPrePackedBuffers`.

Add:

```cpp
bool packed_b_finalized_{false};
```

Before filling a buffer intended for shared caching, zero the entire allocation:

```cpp
std::memset(packed_b_.get(), 0, packed_b_size_);
```

This is required because padding or unwritten bytes participate in content hashing. Uninitialized bytes would produce unstable hashes for identical weights.

### 5. Finalize Scales and Zero Points Before Sharing B

For self-contained `SQNBIT_CompInt8` or KleidiAI packed B:

- Complete B, scale, and zero-point packing while the constant inputs are still available.
- For `HQNBIT_CompInt8`, use `SQNBIT_CompInt8` as the effective compute type.
- Convert constant FP16 scales to FP32 when required by the packing implementation.
- If `has_zp_input_` is true, obtain the constant zero-point and include its correction.
- Set `packed_b_finalized_=true` after all required information is baked into B.
- Guard later scale and zero-point `PrePack` branches with `!packed_b_finalized_`.
- Do not allow later initializer processing to modify a buffer already owned by the shared container.
- Scale and zero-point branches must keep `is_packed=false`; they are not independently cached packed initializers.

If a zero-point input exists but `TryGetConstantInput` fails, do not use this fast path. Preserve the safe fallback.

### 6. Implement the `MatMulNBits` Source Key

Only `MatMulNBits<float>` may return a non-empty source key.

Required conditions:

- `input_idx` is B.
- No `g_idx` is present.
- Scales are constant.
- Zero points, when present, are constant.
- Packed B is self-contained.

A packed B is self-contained only for:

- FP32 LUT GEMM; or
- `SQNBIT_CompInt8` where:
  - The applicable AMD64/x86 format is fully packed; or
  - On non-x86 platforms, `nbits == 8`; or
  - `MlasQNBitGemmScalesPacked(...)` returns true.

Return an empty key for:

- `MatMulNBits<MLFloat16>`.
- Dynamic scales or zero points.
- `g_idx`.
- Non-self-contained formats.
- Paths requiring separate per-kernel `scales_fp32_` or `bias_fp32_` state.

The key must include:

- B `DataRaw()` pointer and element count.
- Scales `DataRaw()` pointer and element count.
- Zero-point `DataRaw()` pointer and element count, or null and zero.
- `K`, `N`, `block_size`, and `nbits`.
- `compute_type_`.
- `has_zp_input_`.
- `has_unquantized_zero_point_`.
- `column_wise_quant_`.
- An explicit version prefix such as `MatMulNBitsPackedV2`.

This key is process-local. Never serialize or reuse it across processes. Do not use only initializer names because different models may contain different weights with identical names.

### 7. Update `UseSharedPrePackedBuffers`

`MatMulNBits::UseSharedPrePackedBuffers` must:

- Accept buffers only when `input_idx` is B and the buffer list is non-empty.
- Adopt the first shared buffer as a non-owning reference.
- Restore `packed_b_size_` from `prepacked_buffer_sizes[0]`.
- Set `used_shared_buffers=true`.
- Reinitialize LUT kernel configuration for LUT-packed buffers.
- Set `packed_b_finalized_=true` for LUT-packed buffers.
- For `SQNBIT_CompInt8`, restore `scales_are_packed_` and `packed_b_finalized_` using the same self-contained-format conditions used by the source-key implementation.

The kernel must never free the shared buffer. The shared container owns its storage.

### 8. Preserve Compatibility

The finished implementation must preserve these invariants:

- The first session still performs normal `PrePack` and content hashing.
- The canonical container key remains `op_type + packed-buffer content hash`.
- Kernels without a safe source key retain pack-then-hash behavior.
- Non-shared sessions remain unchanged.
- Non-CPU execution providers remain unchanged.
- `AddInitializer` backing memory and `OrtValue` lifetime requirements remain unchanged.
- `PrepackedWeightsContainer` must outlive every session using it.
- Dynamic zero-point, FP16, and non-self-contained formats must never enter the source-key fast path.

Keep the changes minimal and localized. Do not introduce a generic pointer-key optimization for kernels that have not explicitly proven that their packed output is fully determined by immutable source identity and packing parameters.

