import numpy as np
import tensorflow as tf
import tf2onnx
from pathlib import Path

from model import LIAEModel

MODEL_DIR = Path("model")
OUT_DIR = Path("converter/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPORTING LIAE FP32 TO ONNX")
print("=" * 80)

# Load weights
encoder = np.load(MODEL_DIR / "new_SAEHD_encoder.npy", allow_pickle=True)
inter_ab = np.load(MODEL_DIR / "new_SAEHD_inter_AB.npy", allow_pickle=True)
inter_b = np.load(MODEL_DIR / "new_SAEHD_inter_B.npy", allow_pickle=True)
decoder = np.load(MODEL_DIR / "new_SAEHD_decoder.npy", allow_pickle=True)

print("✓ Weights loaded")

# Build original model
liae = LIAEModel(encoder, inter_ab, inter_b, decoder)

# Build once
dummy = tf.random.uniform((1, 128, 128, 3))
liae(dummy, dummy)

print("✓ Model built")

# ---------- Wrapper ----------
class ExportModel(tf.keras.Model):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def call(self, inputs):
        src, dst = inputs
        return self.base(src, dst)

export_model = ExportModel(liae)

# Build wrapper
src_in = tf.keras.Input(shape=(128, 128, 3), name="src")
dst_in = tf.keras.Input(shape=(128, 128, 3), name="dst")
export_model([src_in, dst_in])

onnx_path = OUT_DIR / "LIAE_128_80_48_16_fp32.onnx"

tf2onnx.convert.from_keras(
    export_model,
    input_signature=(
        tf.TensorSpec((1, 128, 128, 3), tf.float32, name="src"),
        tf.TensorSpec((1, 128, 128, 3), tf.float32, name="dst"),
    ),
    opset=17,
    output_path=str(onnx_path),
)

print()
print("=" * 80)
print("EXPORT COMPLETE")
print("=" * 80)
print(f"Saved: {onnx_path}")
