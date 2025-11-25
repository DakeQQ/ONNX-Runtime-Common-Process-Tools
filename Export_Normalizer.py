import torch
from onnxslim import slim
import onnxruntime as ort
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "normalizer_type": "rms",           # ["rms", "max"] - Select which normalizer to export
    "dynamic": True,                    # True for dynamic shape, False for static
    "input_audio_length": 16000,        # Used for static shape
    "input_dtype": torch.int16,         # [torch.float32, torch.int16]
    "output_dtype": torch.int16,        # [torch.float32, torch.int16]
    "opset": 23,                        # ONNX opset version
    "target_value": 4096.0,             # Default target value for RMS (use ~16384.0 for MAX)
    "run_test": True,                   # Run ONNX Runtime test after export
    "test_audio_lengths": [8000, 16000, 32000],  # Test with different lengths (if dynamic)
}

# ============================================================================
# NORMALIZER MODULES
# ============================================================================

class NormalizerRMS(torch.nn.Module):
    """RMS-based audio normalizer"""
    def __init__(self, input_dtype, output_dtype):
        super(NormalizerRMS, self).__init__()
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.eps = torch.tensor([1e-6], dtype=torch.float32).view(1, 1, -1)

    def forward(self, audio, target_value):
        if self.input_dtype != torch.float32:
            audio = audio.float()
        
        rms_val = torch.sqrt(audio.pow(2).mean(dim=-1, keepdim=True))
        scaling_factor = (target_value / (rms_val + self.eps))
        normalized = (audio * scaling_factor).clamp(min=-32768.0, max=32767.0)
        
        if self.output_dtype != torch.float32:
            normalized = normalized.to(torch.int16)
        
        return normalized


class NormalizerMAX(torch.nn.Module):
    """MAX-based audio normalizer"""
    def __init__(self, input_dtype, output_dtype):
        super(NormalizerMAX, self).__init__()
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        
        if input_dtype != torch.int16:
            self.eps = torch.tensor([1e-6], dtype=input_dtype)
        else:
            self.eps = torch.tensor([1], dtype=input_dtype)
        self.eps = self.eps.view(1, 1, -1)

    def forward(self, audio, target_value):
        max_val, _ = torch.max(torch.abs(audio).float(), dim=-1, keepdim=True)
        scaling_factor = (target_value / (max_val + self.eps))
        
        if self.input_dtype == torch.int16:
            scaling_factor = scaling_factor.to(torch.int16)
        
        normalized = audio * scaling_factor
        
        if self.output_dtype != torch.float32 and self.input_dtype == torch.float32:
            normalized = normalized.to(self.output_dtype)
        
        return normalized


# ============================================================================
# EXPORT FUNCTION
# ============================================================================

def export_to_onnx(config):
    """
    Export the selected normalizer to ONNX format
    
    Args:
        config (dict): Configuration dictionary containing export settings
    
    Returns:
        model: The PyTorch model instance (for testing)
        save_path: Path to the exported ONNX model
    """
    # Extract config values
    normalizer_type = config["normalizer_type"].lower()
    dynamic = config["dynamic"]
    input_audio_length = config["input_audio_length"]
    input_dtype = config["input_dtype"]
    output_dtype = config["output_dtype"]
    opset = config["opset"]
    target_value = config["target_value"]
    
    # Set save path based on normalizer type
    save_path = f"normalizer_{normalizer_type}.onnx"

    # Select and instantiate the appropriate model
    if normalizer_type == "rms":
        model = NormalizerRMS(input_dtype, output_dtype)
    elif normalizer_type == "max":
        model = NormalizerMAX(input_dtype, output_dtype)
    else:
        raise ValueError(f"Unknown normalizer_type: {normalizer_type}. Must be 'rms' or 'max'")

    model.eval()
    
    # Create dummy inputs
    if input_dtype == torch.int16:
        dummy_input_0 = torch.randint(
            size=(1, 1, input_audio_length), 
            low=-32768, 
            high=32767, 
            dtype=input_dtype
        )
    else:
        dummy_input_0 = torch.randn(
            size=(1, 1, input_audio_length), 
            dtype=input_dtype
        )
    
    dummy_input_1 = torch.tensor([target_value], dtype=torch.float32)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input_0, dummy_input_1),
        save_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['audio', 'target_value'],
        output_names=['normalized_audio'],
        dynamic_axes={
            'audio': {2: 'audio_length'},
            'normalized_audio': {2: 'audio_length'}
        } if dynamic else None,
        dynamo=False
    )
    
    # Optimize with onnxslim
    slim(
        model=save_path,
        output_model=save_path,
        no_shape_infer=False,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=False,
        verbose=False
    )
    
    print(f"✓ {normalizer_type.upper()} normalizer exported successfully to {save_path}")
    print(f"  - Input dtype: {input_dtype}")
    print(f"  - Output dtype: {output_dtype}")
    print(f"  - Dynamic shape: {dynamic}")
    print(f"  - Target value: {target_value}")
    
    return model, save_path


# ============================================================================
# ONNX RUNTIME TEST
# ============================================================================

def test_onnx_model(model, onnx_path, config):
    """
    Test the exported ONNX model against the PyTorch model
    
    Args:
        model: PyTorch model instance
        onnx_path: Path to the ONNX model
        config: Configuration dictionary
    """
    print(f"\n{'='*70}")
    print(f"ONNX RUNTIME TEST - {config['normalizer_type'].upper()}")
    print(f"{'='*70}")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    
    # Print model info
    print(f"\n📋 Model Information:")
    print(f"  Inputs:")
    for inp in ort_session.get_inputs():
        print(f"    - {inp.name}: {inp.type}, shape={inp.shape}")
    print(f"  Outputs:")
    for out in ort_session.get_outputs():
        print(f"    - {out.name}: {out.type}, shape={out.shape}")
    
    # Determine test lengths
    if config["dynamic"]:
        test_lengths = config.get("test_audio_lengths", [16000])
    else:
        test_lengths = [config["input_audio_length"]]
    
    input_dtype = config["input_dtype"]
    output_dtype = config["output_dtype"]
    target_value = config["target_value"]
    
    print(f"\n🧪 Running tests on {len(test_lengths)} different audio lengths...")
    
    all_passed = True
    
    for idx, audio_length in enumerate(test_lengths, 1):
        print(f"\n--- Test {idx}/{len(test_lengths)}: Audio length = {audio_length} ---")
        
        # Generate test input
        if input_dtype == torch.int16:
            test_audio = torch.randint(
                size=(1, 1, audio_length),
                low=-32768,
                high=32767,
                dtype=input_dtype
            )
        else:
            test_audio = torch.randn(
                size=(1, 1, audio_length),
                dtype=input_dtype
            ) * 10000  # Scale for better testing
        
        test_target = torch.tensor([target_value], dtype=torch.float32)
        
        # PyTorch inference
        model.eval()
        with torch.no_grad():
            pytorch_output = model(test_audio, test_target)
        
        # ONNX Runtime inference
        ort_inputs = {
            ort_session.get_inputs()[0].name: test_audio.numpy(),
            ort_session.get_inputs()[1].name: test_target.numpy()
        }
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Convert PyTorch output to numpy
        pytorch_output_np = pytorch_output.numpy()
        
        # Compare outputs
        max_diff = np.abs(pytorch_output_np - ort_output).max()
        mean_diff = np.abs(pytorch_output_np - ort_output).mean()
        
        # Calculate relative error (avoid division by zero)
        pytorch_abs = np.abs(pytorch_output_np)
        mask = pytorch_abs > 1e-6
        if mask.any():
            relative_error = np.abs((pytorch_output_np - ort_output) / (pytorch_output_np + 1e-10))
            max_relative_error = relative_error[mask].max() if mask.any() else 0
        else:
            max_relative_error = 0
        
        # Determine if test passed
        # For int16, exact match expected; for float32, allow small numerical differences
        if output_dtype == torch.int16:
            tolerance = 1  # Allow 1 bit difference due to rounding
            passed = max_diff <= tolerance
        else:
            tolerance = 1e-4
            passed = max_diff < tolerance
        
        # Print results
        print(f"  PyTorch output - shape: {pytorch_output_np.shape}, "
              f"dtype: {pytorch_output_np.dtype}, "
              f"range: [{pytorch_output_np.min():.2f}, {pytorch_output_np.max():.2f}]")
        print(f"  ONNX output    - shape: {ort_output.shape}, "
              f"dtype: {ort_output.dtype}, "
              f"range: [{ort_output.min():.2f}, {ort_output.max():.2f}]")
        print(f"\n  Difference Statistics:")
        print(f"    Max absolute difference: {max_diff:.6f}")
        print(f"    Mean absolute difference: {mean_diff:.6f}")
        print(f"    Max relative error: {max_relative_error:.6e}")
        print(f"    Tolerance: {tolerance}")
        
        if passed:
            print(f"  ✅ Test PASSED")
        else:
            print(f"  ❌ Test FAILED (difference {max_diff} > tolerance {tolerance})")
            all_passed = False
            
            # Print sample values for debugging
            print(f"\n  Sample values (first 5):")
            print(f"    PyTorch: {pytorch_output_np.flatten()[:5]}")
            print(f"    ONNX:    {ort_output.flatten()[:5]}")
    
    # Final summary
    print(f"\n{'='*70}")
    if all_passed:
        print(f"✅ ALL TESTS PASSED - ONNX model matches PyTorch model")
    else:
        print(f"❌ SOME TESTS FAILED - Please review differences above")
    print(f"{'='*70}\n")
    
    return all_passed


# ============================================================================
# BENCHMARK (Optional)
# ============================================================================

def benchmark_onnx_model(onnx_path, config, num_iterations=100):
    """
    Benchmark the ONNX model inference speed
    
    Args:
        onnx_path: Path to the ONNX model
        config: Configuration dictionary
        num_iterations: Number of iterations for benchmarking
    """
    import time
    
    print(f"\n{'='*70}")
    print(f"PERFORMANCE BENCHMARK")
    print(f"{'='*70}")
    
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    
    audio_length = config["input_audio_length"]
    input_dtype = config["input_dtype"]
    target_value = config["target_value"]
    
    # Prepare input
    if input_dtype == torch.int16:
        test_audio = np.random.randint(
            -32768, 32767,
            size=(1, 1, audio_length),
            dtype=np.int16
        )
    else:
        test_audio = np.random.randn(1, 1, audio_length).astype(np.float32)
    
    test_target = np.array([target_value], dtype=np.float32)
    
    ort_inputs = {
        ort_session.get_inputs()[0].name: test_audio,
        ort_session.get_inputs()[1].name: test_target
    }
    
    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, ort_inputs)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        _ = ort_session.run(None, ort_inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
    throughput = audio_length / (avg_time / 1000)  # Samples per second
    
    print(f"\n  Audio length: {audio_length} samples")
    print(f"  Iterations: {num_iterations}")
    print(f"  Average inference time: {avg_time:.4f} ms")
    print(f"  Throughput: {throughput:,.0f} samples/second")
    print(f"  Real-time factor (16kHz): {throughput / 16000:.2f}x")
    print(f"\n{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Export model
    model, save_path = export_to_onnx(CONFIG)
    
    # Test ONNX model
    if CONFIG.get("run_test", True):
        test_passed = test_onnx_model(model, save_path, CONFIG)
        
        # Optional: Run benchmark
        if test_passed:
            benchmark_onnx_model(save_path, CONFIG, num_iterations=100)
    
    # ========================================================================
    # Optional: Export and test both normalizers
    # ========================================================================
    # print("\n" + "="*70)
    # print("EXPORTING BOTH NORMALIZERS")
    # print("="*70)
    # 
    # print("\n[1/2] Exporting RMS normalizer...")
    # CONFIG["normalizer_type"] = "rms"
    # CONFIG["target_value"] = 4096.0
    # model_rms, path_rms = export_to_onnx(CONFIG)
    # if CONFIG["run_test"]:
    #     test_onnx_model(model_rms, path_rms, CONFIG)
    #     benchmark_onnx_model(path_rms, CONFIG)
    # 
    # print("\n[2/2] Exporting MAX normalizer...")
    # CONFIG["normalizer_type"] = "max"
    # CONFIG["target_value"] = 16384.0
    # model_max, path_max = export_to_onnx(CONFIG)
    # if CONFIG["run_test"]:
    #     test_onnx_model(model_max, path_max, CONFIG)
    #     benchmark_onnx_model(path_max, CONFIG)
