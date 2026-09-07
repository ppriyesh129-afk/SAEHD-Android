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

# -------------------------------------------------
# Load official DeepFaceLab weights
# -------------------------------------------------

encoder = np.load(MODEL_DIR / "new_SAEHD_encoder.npy", allow_pickle=True)
inter_ab = np.load(MODEL_DIR / "new_SAEHD_inter_AB.npy", allow_pickle=True)
inter_b = np.load(MODEL_DIR / "new_SAEHD_inter_B.npy", allow_pickle=True)
decoder = np.load(MODEL_DIR / "new_SAEHD_decoder.npy", allow_pickle=True)

print("✓ Weights loaded")

# -------------------------------------------------
# Build model
# -------------------------------------------------

model = LIAEModel(encoder, inter_ab, inter_b, decoder)

# Build once
src = tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)
dst = tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)
model(src, dst)

print("✓ Model built")

# -------------------------------------------------
# Export ONNX (weights embedded)
# -------------------------------------------------

onnx_path = OUT_DIR / "LIAE_128_80_48_16_fp32.onnx"

spec = (
    tf.TensorSpec((1, 128, 128, 3), tf.float32, name="src"),
    tf.TensorSpec((1, 128, 128, 3), tf.float32, name="dst"),
)

tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17,
    output_path=str(onnx_path),
)

print()
print("=" * 80)
print("EXPORT COMPLETE")
print("=" * 80)
print(f"Saved: {onnx_path}")
