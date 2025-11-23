import torch
import torch.nn as nn
from onnxslim import slim

# Export to ONNX
DYANMIC = True                          # Set True for dynamic shape.
INPUT_AUDIO_LENGTH = 16000              # Set for static shape.
IN_DTYPE = torch.int16                  # [torch.float32, torch.int16]
OUT_DTYPE = torch.int16                 # [torch.float32, torch.int16]
save_path = "weighted_average.onnx"
OPSET = 23


class WeightedAverage(nn.Module):
    def __init__(self):
        super(WeightedAverage, self).__init__()

    def forward(self, original_audio, denoised_audio, wieght, weight_minus):
        if IN_DTYPE != torch.float32:
            original_audio = original_audio.float()
            denoised_audio = denoised_audio.float()
        mix_audio = original_audio * weight_minus + denoised_audio * wieght
        if OUT_DTYPE == torch.int16:
            return mix_audio.to(torch.int16)
        return mix_audio

def export_to_onnx():
    model = WeightedAverage()
    model.eval()
    dummy_input_0 = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=IN_DTYPE)
    dummy_input_1 = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=IN_DTYPE)
    dummy_input_2 = torch.tensor([0.4], dtype=torch.float32)
    dummy_input_3 = 1.0 - dummy_input_2

    torch.onnx.export(
        model,
        (dummy_input_0, dummy_input_1, dummy_input_2, dummy_input_3),
        save_path,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=['original_audio', 'denoised_audio', 'wieght', 'weight_minus'],
        output_names=['mix_audio'],
        dynamic_axes={
            'original_audio': {2: 'audio_length'},
            'denoised_audio': {2: 'audio_length'},
            'mix_audio': {2: 'audio_length'}
        } if DYANMIC else None,
        dynamo=False
    )

    slim(
        model=save_path,
        output_model=save_path,
        no_shape_infer=False,            # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=False,
        verbose=False
    )

    print(f"Model exported successfully to {save_path}")


if __name__ == "__main__":
    export_to_onnx()
