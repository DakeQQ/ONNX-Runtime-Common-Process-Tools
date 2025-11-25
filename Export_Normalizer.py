import torch
from onnxslim import slim

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
    "target_value": 4096.0              # Default target value for RMS (use ~16384.0 for MAX)
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
        input_name_target = 'target_rms'
    elif normalizer_type == "max":
        model = NormalizerMAX(input_dtype, output_dtype)
        input_name_target = 'target_value'
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
        input_names=['audio', input_name_target],
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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    export_to_onnx(CONFIG)
    
    # Optional: Export both normalizers
    # print("\nExporting RMS normalizer...")
    # CONFIG["normalizer_type"] = "rms"
    # CONFIG["target_value"] = 4096.0
    # export_to_onnx(CONFIG)
    # 
    # print("\nExporting MAX normalizer...")
    # CONFIG["normalizer_type"] = "max"
    # CONFIG["target_value"] = 16384.0
    # export_to_onnx(CONFIG)
