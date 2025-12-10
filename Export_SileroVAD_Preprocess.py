import torch
import torch.nn as nn
import numpy as np
from onnxslim import slim


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Export settings
    EXPORT_PATH = "./silero_vad_preprocess.onnx"        # Directory to save exported ONNX models
    OPSET_VERSION = 23                                  # ONNX opset version
    
    # Model settings
    DYNAMIC = False                                     # True: dynamic padding in forward(), False: static pre-allocated padding
    MAX_AUDIO_LENGTH = 16000                            # Required if DYNAMIC=False, maximum expected audio length
    USE_FLOAT16 = False
    
    # Buffer settings
    ZEROS_BUFFER_SIZE = 1000                            # Size of pre-allocated zeros buffer [ZEROS_BUFFER_SIZE, 64], 1000 for 30 seconds audio inputs
    
    # Reshape settings
    FRAME_SIZE = 512                                    # Reshape audio to [-1, FRAME_SIZE], 512 for 16kHz. 256 for 8kHz Silero_VAD
    CONCAT_ZEROS_SIZE = 64                              # Concat zeros with shape [N, CONCAT_ZEROS_SIZE]
    
    # Test settings
    TEST_AUDIO_LENGTHS = [16000, 24000, 32768, 48000]   # Audio lengths to test
    TOLERANCE = 1e-5                                    # Tolerance for output comparison

# ============================================================


class AudioProcessor(nn.Module):
    def __init__(self, config):
        """
        Args:
            config: Configuration object with all settings
        """
        super(AudioProcessor, self).__init__()
        self.config = config
        self.dynamic = config.DYNAMIC
        self.dtype = torch.float16 if config.USE_FLOAT16 else torch.float32
        self.normalization_factor = torch.tensor([1.0 / 32768.0], dtype=self.dtype).view(1, 1, -1)

        if self.dynamic:
            self.register_buffer('padding_zeros', torch.zeros((1, 1, config.FRAME_SIZE), dtype=torch.int8))
            self.register_buffer('zeros_buffer', torch.zeros(config.ZEROS_BUFFER_SIZE, config.CONCAT_ZEROS_SIZE, dtype=torch.int8))
        else:
            if config.MAX_AUDIO_LENGTH is None:
                raise ValueError("MAX_AUDIO_LENGTH must be provided when DYNAMIC=False")

            # Pre-compute the padded length (FRAME_SIZE integer times)
            self.total_length = ((config.MAX_AUDIO_LENGTH + config.FRAME_SIZE - 1) // config.FRAME_SIZE) * config.FRAME_SIZE
            self.padding_size = self.total_length - config.MAX_AUDIO_LENGTH

            # Pre-allocate zeros for padding
            self.register_buffer('padding_zeros', torch.zeros((1, 1, self.padding_size), dtype=self.dtype))
            self.register_buffer('zeros_buffer', torch.zeros(self.total_length // config.FRAME_SIZE, config.CONCAT_ZEROS_SIZE, dtype=self.dtype))
    
    def forward(self, audio):
        """
        Args:
            audio: Input audio tensor, shape [audio_length]
        
        Returns:
            output: Processed audio with concatenated zeros, shape [N, FRAME_SIZE + CONCAT_ZEROS_SIZE]
        """
        # Step 1: Normalize audio
        audio_normalized = audio.to(self.dtype) * self.normalization_factor

        # Step 2: Pad to FRAME_SIZE integer times
        if self.dynamic:
            # Compute necessary length dynamically
            audio_length = audio_normalized.shape[-1]
            total_length = ((audio_length + self.config.FRAME_SIZE - 1) // self.config.FRAME_SIZE) * self.config.FRAME_SIZE
            padding_size = total_length - audio_length
            audio_padded = torch.cat([audio_normalized, self.padding_zeros[..., :padding_size].to(audio_normalized.dtype)], dim=-1)
        else:
            # Use pre-allocated zeros
            padding_size = self.padding_size
            audio_padded = torch.cat([audio_normalized, self.padding_zeros], dim=-1)

        # Step 3: Reshape to [-1, FRAME_SIZE]
        audio_reshaped = audio_padded.reshape(-1, self.config.FRAME_SIZE)

        # Step 4: Concat zeros array with shape [audio.shape[0], CONCAT_ZEROS_SIZE]
        # Use pre-allocated int8 zeros buffer, slice and cast
        if self.dynamic:
            num_frames = audio_reshaped.shape[0]
            zeros_slice = self.zeros_buffer[:num_frames, :].to(self.dtype)
        else:
            zeros_slice = self.zeros_buffer
        # Step 5: Concatenate along dim=-1
        output = torch.cat([audio_reshaped, zeros_slice], dim=-1)
        if self.dynamic:
            effective_len = output.shape[0] - 1 if padding_size != 0 else output.shape[0]  # Drop the last state came from padding
            return output, effective_len
        split_output = output.split(split_size=1, dim=0)
        return split_output


def export_to_onnx(config):
    """
    Export the model to ONNX format.
    
    Args:
        config: Configuration object with all settings
    """
    import os
    
    # Create model
    model = AudioProcessor(config)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.ones((1, 1, config.MAX_AUDIO_LENGTH), dtype=torch.int16)
    
    # Generate filename
    onnx_filepath = config.EXPORT_PATH
    
    print(f"{'='*60}")
    print(f"Exporting model in {'DYNAMIC' if config.DYNAMIC else 'STATIC'} mode")
    print(f"{'='*60}")
    print(f"Export path: {onnx_filepath}")
    print(f"Opset version: {config.OPSET_VERSION}")
    print(f"Frame size: {config.FRAME_SIZE}")
    print(f"Concat zeros size: {config.CONCAT_ZEROS_SIZE}")
    if not config.DYNAMIC:
        print(f"Max audio length: {config.MAX_AUDIO_LENGTH}")
    print(f"{'='*60}\n")

    if config.DYNAMIC:
        output_names = ['output_frame', 'effective_len']
        dynamic_axes = {
            'audio': {2: 'audio_length'},
            'output': {0: 'num_frames'}
        }
    else:
        output_names = []
        dynamic_axes = None
        for i in range(model.zeros_buffer.shape[0]):
            output_names.append(f"output_frame_{i}")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_filepath,
        export_params=True,
        opset_version=config.OPSET_VERSION,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=output_names,
        dynamo=False,
        dynamic_axes=dynamic_axes
    )

    slim(
        model=onnx_filepath,
        output_model=onnx_filepath,
        no_shape_infer=False,    # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=False,
        verbose=False
    )

    print(f"✓ Model exported to {onnx_filepath}")
    
    # Verify the export
    import onnx
    onnx_model = onnx.load(onnx_filepath)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid!")
    
    # Test the PyTorch model
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✓ PyTorch model test:")
        print(f"  Input shape: {dummy_input.shape}")
        if config.DYNAMIC:
            print(f"  Output shape: {output[0].shape}")
        else:
            print(f"  Output shape: {output[0].shape} * {len(output)}")
    
    return model, onnx_filepath


if __name__ == "__main__":
    # Create config instance
    config = Config()
    
    # Export model with current configuration
    model, onnx_file = export_to_onnx(config)
