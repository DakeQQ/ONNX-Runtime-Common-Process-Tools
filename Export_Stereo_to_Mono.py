import torch
from onnxslim import slim


save_path = "stereo_to_mono.onnx"
DYNAMIC = False              # Set True for dunamic length input.
INPUT_AUDIO_LENGTH = 48000   # For staic input shape.
IN_DTYPE = torch.int16       # [torch.int16, torch.float32]
OUT_DTYPE = torch.int16      # [torch.int16, torch.float32]
OPSET = 23


class StereoToMono(torch.nn.Module):
    def __init__(self):
        super(StereoToMono, self).__init__()

    def forward(self, stereo: torch.Tensor) -> torch.Tensor:
        stereo = stereo.view(-1, 2)
        if IN_DTYPE != torch.int16:
            mono = stereo.mean(dim=-1, keepdim=False)
        else:
            stereo = stereo.float()
            if DYNAMIC:
                mono = stereo.mean(dim=-1, keepdim=False)
            else:
                mono = stereo.sum(dim=-1, keepdim=False)
                mono *= 0.5
        if OUT_DTYPE != torch.float32:
            mono = mono.to(torch.int16)
        return mono.view(1, 1, -1)


model = StereoToMono()
model.eval()

dummy_input = torch.ones(INPUT_AUDIO_LENGTH + INPUT_AUDIO_LENGTH, dtype=IN_DTYPE)

# Export
torch.onnx.export(
    model,
    dummy_input,
    save_path,
    export_params=True,
    opset_version=OPSET,
    do_constant_folding=True,
    input_names=["stereo_in"],
    output_names=["mono_out"],
    dynamic_axes={
        "stereo_in": {0: "num_interleaved_samples"},
        "mono_out": {-1: "num_mono_samples"},
    } if DYNAMIC else None,
    dynamo=False
)
# Optimize
print("Optimizing model...")
slim(
    model=save_path,
    output_model=save_path,
    no_shape_infer=False,
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)

print(f"Model exported and optimized.")
print(f"Exported ONNX model to {save_path}")
