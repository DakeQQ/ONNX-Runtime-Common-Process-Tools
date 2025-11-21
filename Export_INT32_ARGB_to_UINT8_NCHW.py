import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxslim import slim

# --- Configuration ---
INPUT_FORMAT = 'ARGB'   # Options: ['RGB', 'BGR', 'ARGB', 'ABGR', 'RGBA', 'BGRA']
OUTPUT_FORMAT = 'NCHW'  # Options: ['NCHW', 'NHWC']
KEEP_ALPHA = False      # Set True to keep alpha channel (ignored for 'RGB' and 'BGR')
DYNAMIC = False         # Set True for dynamic image size.
HEIGHT = 720            # Set for static shape.
WIDTH = 1280            # Set for static shape.
OPSET = 23              # Must >= 18
SAVE_FOLDER = './'      # Default to current folder

# Validate format
SUPPORTED_FORMATS = ['ABGR', 'BGR', 'ARGB', 'RGBA', 'BGRA', 'RGB']
SUPPORTED_OUTPUT_FORMATS = ['NCHW', 'NHWC']

if INPUT_FORMAT not in SUPPORTED_FORMATS:
    raise ValueError(f"INPUT_FORMAT must be one of {SUPPORTED_FORMATS}, got '{INPUT_FORMAT}'")

if OUTPUT_FORMAT not in SUPPORTED_OUTPUT_FORMATS:
    raise ValueError(f"OUTPUT_FORMAT must be one of {SUPPORTED_OUTPUT_FORMATS}, got '{OUTPUT_FORMAT}'")

# Determine if alpha channel will be included
HAS_ALPHA_INPUT = INPUT_FORMAT not in ['RGB', 'BGR']
INCLUDE_ALPHA = HAS_ALPHA_INPUT and KEEP_ALPHA

# Generate save path based on format
channels_suffix = "rgba" if INCLUDE_ALPHA else "rgb"
save_path = f"{SAVE_FOLDER}/flat_{INPUT_FORMAT.lower()}_to_{OUTPUT_FORMAT.lower()}_{channels_suffix}.onnx"


class FlatPackedInt32ToNCHW(torch.nn.Module):
    def __init__(self, height, width, format_type, output_format, keep_alpha):
        super().__init__()
        self.height = height
        self.width = width
        self.format_type = format_type
        self.output_format = output_format
        self.keep_alpha = keep_alpha

        # Define bit positions for each channel based on format
        self.format_config = {
            'ABGR': {'R': 0, 'G': 8, 'B': 16, 'A': 24, 'has_alpha': True},
            'BGR': {'R': 0, 'G': 8, 'B': 16, 'has_alpha': False},
            'ARGB': {'R': 16, 'G': 8, 'B': 0, 'A': 24, 'has_alpha': True},
            'RGBA': {'R': 24, 'G': 16, 'B': 8, 'A': 0, 'has_alpha': True},
            'BGRA': {'R': 8, 'G': 16, 'B': 24, 'A': 0, 'has_alpha': True},
            'RGB': {'R': 16, 'G': 8, 'B': 0, 'has_alpha': False},
        }

    def forward(self, flat_input):
        config = self.format_config[self.format_type]

        # Extract channels
        if self.format_type in ['BGR', 'RGBA', 'BGRA', 'RGB']:
            # Use the simpler shift-then-mask for formats that work correctly
            r = (flat_input >> config['R']) & 0xFF
            g = (flat_input >> config['G']) & 0xFF
            b = (flat_input >> config['B']) & 0xFF
            if config['has_alpha']:
                a = (flat_input >> config['A']) & 0xFF
        else:
            # For ABGR and ARGB: mask before shifting to avoid sign extension
            r_mask = 0xFF << config['R']
            g_mask = 0xFF << config['G']
            b_mask = 0xFF << config['B']

            r = (flat_input & r_mask) >> config['R']
            g = (flat_input & g_mask) >> config['G']
            b = (flat_input & b_mask) >> config['B']
            
            if config['has_alpha']:
                a_mask = 0xFF << config['A']
                a = (flat_input & a_mask) >> config['A']

        # Stack channels based on whether alpha should be kept
        if config['has_alpha'] and self.keep_alpha:
            planar = torch.stack((r, g, b, a), dim=0)
        else:
            planar = torch.stack((r, g, b), dim=0)

        planar = planar.to(torch.uint8)

        # Reshape based on output format
        if self.output_format == 'NCHW':
            return planar.view(1, -1, self.height, self.width)
        else:  # NHWC
            nchw = planar.view(1, -1, self.height, self.width)
            return nchw.permute(0, 2, 3, 1)  # Convert NCHW to NHWC


def get_packing_formula(format_type, val_r, val_g, val_b, val_a=255):
    """
    Returns the packed pixel value based on the format type.
    """
    packing_formulas = {
        'ABGR': (val_a << 24) | (val_b << 16) | (val_g << 8) | val_r,
        'BGR': (val_b << 16) | (val_g << 8) | val_r,
        'ARGB': (val_a << 24) | (val_r << 16) | (val_g << 8) | val_b,
        'RGBA': (val_r << 24) | (val_g << 16) | (val_b << 8) | val_a,
        'BGRA': (val_b << 24) | (val_g << 16) | (val_r << 8) | val_a,
        'RGB': (val_r << 16) | (val_g << 8) | val_b,
    }
    return packing_formulas[format_type]


# --- Export Process ---

# 1. Initialize Model
model = FlatPackedInt32ToNCHW(HEIGHT, WIDTH, INPUT_FORMAT, OUTPUT_FORMAT, KEEP_ALPHA)
model.eval()

# 2. Create Dummy Input (Size = H * W)
total_pixels = HEIGHT * WIDTH
dummy_input = torch.randint(
    low=-2147483648,
    high=2147483647,
    size=(total_pixels,),
    dtype=torch.int32
)

# 3. Export to ONNX
output_channels = 4 if INCLUDE_ALPHA else 3
channel_desc = "RGBA" if INCLUDE_ALPHA else "RGB"

print(f"Exporting {INPUT_FORMAT} model to {save_path} with {OUTPUT_FORMAT} output...")
print(f"Output channels: {output_channels} ({channel_desc})")

# Set up dynamic axes based on output format
if DYNAMIC:
    if OUTPUT_FORMAT == 'NCHW':
        dynamic_axes = {
            'flat_input_int32': {0: 'num_pixels'},
            'output_uint8': {2: 'height', 3: 'width'}
        }
    else:  # NHWC
        dynamic_axes = {
            'flat_input_int32': {0: 'num_pixels'},
            'output_uint8': {1: 'height', 2: 'width'}
        }
else:
    dynamic_axes = None

torch.onnx.export(
    model,
    dummy_input,
    save_path,
    export_params=True,
    opset_version=OPSET,
    do_constant_folding=True,
    input_names=['flat_input_int32'],
    output_names=['output_uint8'],
    dynamic_axes=dynamic_axes,
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

# --- Verification Step ---

print(f"\nVerifying logic for {INPUT_FORMAT} packing with {OUTPUT_FORMAT} output...")

# Create predictable pixel values
val_r, val_g, val_b, val_a = 10, 20, 30, 128

# Get packed pixel based on format
packed_pixel = get_packing_formula(INPUT_FORMAT, val_r, val_g, val_b, val_a)

# Convert to signed 32-bit integer representation for PyTorch input
if packed_pixel >= 2 ** 31:
    packed_pixel -= 2 ** 32

test_input = torch.tensor([packed_pixel] * total_pixels, dtype=torch.int32)

# Run ONNX Runtime
ort_session = ort.InferenceSession(save_path)
ort_inputs = {'flat_input_int32': test_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)
result = ort_outs[0]

print(f"Output Shape: {result.shape} ({OUTPUT_FORMAT})")
print(f"Packed pixel value: {packed_pixel} (0x{packed_pixel & 0xFFFFFFFF:08X})")

# Check the values of the first pixel based on output format
if OUTPUT_FORMAT == 'NCHW':
    out_r = result[0, 0, 0, 0]
    out_g = result[0, 1, 0, 0]
    out_b = result[0, 2, 0, 0]
    if INCLUDE_ALPHA:
        out_a = result[0, 3, 0, 0]
else:  # NHWC
    out_r = result[0, 0, 0, 0]
    out_g = result[0, 0, 0, 1]
    out_b = result[0, 0, 0, 2]
    if INCLUDE_ALPHA:
        out_a = result[0, 0, 0, 3]

if INCLUDE_ALPHA:
    print(f"\nExpected: R={val_r}, G={val_g}, B={val_b}, A={val_a}")
    print(f"Actual:   R={out_r}, G={out_g}, B={out_b}, A={out_a}")
    success = (out_r == val_r and out_g == val_g and out_b == val_b and out_a == val_a)
else:
    print(f"\nExpected: R={val_r}, G={val_g}, B={val_b} (Alpha dropped)")
    print(f"Actual:   R={out_r}, G={out_g}, B={out_b}")
    success = (out_r == val_r and out_g == val_g and out_b == val_b)

if success:
    print(f"✓ SUCCESS: ONNX model matches {INPUT_FORMAT} packing with {OUTPUT_FORMAT} output.")
else:
    print(f"✗ FAILURE: Data layout mismatch.")

# Print format information
print(f"\n--- {INPUT_FORMAT} Format Details ---")
config = model.format_config[INPUT_FORMAT]
print(f"Bit layout (MSB to LSB):")
if INPUT_FORMAT == 'ABGR':
    print("  [31:24] Alpha | [23:16] Blue | [15:8] Green | [7:0] Red")
elif INPUT_FORMAT == 'BGR':
    print("  [23:16] Blue | [15:8] Green | [7:0] Red")
elif INPUT_FORMAT == 'ARGB':
    print("  [31:24] Alpha | [23:16] Red | [15:8] Green | [7:0] Blue")
elif INPUT_FORMAT == 'RGBA':
    print("  [31:24] Red | [23:16] Green | [15:8] Blue | [7:0] Alpha")
elif INPUT_FORMAT == 'BGRA':
    print("  [31:24] Blue | [23:16] Green | [15:8] Red | [7:0] Alpha")
elif INPUT_FORMAT == 'RGB':
    print("  [23:16] Red | [15:8] Green | [7:0] Blue")

print(f"Output format: {OUTPUT_FORMAT}")
if INCLUDE_ALPHA:
    print(f"Output channels: RGBA (Alpha kept)")
else:
    print(f"Output channels: RGB (Alpha {'N/A' if INPUT_FORMAT in ['RGB', 'BGR'] else 'dropped'})")

