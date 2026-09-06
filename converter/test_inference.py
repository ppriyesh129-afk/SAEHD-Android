import numpy as np
import tensorflow as tf
from pathlib import Path

from model import LIAEModel

MODEL_DIR = Path("model")

print("=" * 80)
print("LIAE FP32 INFERENCE TEST")
print("=" * 80)

# -------------------------------------------------
# Load official DeepFaceLab weights
# -------------------------------------------------

encoder = np.load(
    MODEL_DIR / "new_SAEHD_encoder.npy",
    allow_pickle=True,
)

inter_ab = np.load(
    MODEL_DIR / "new_SAEHD_inter_AB.npy",
    allow_pickle=True,
)

inter_b = np.load(
    MODEL_DIR / "new_SAEHD_inter_B.npy",
    allow_pickle=True,
)

decoder = np.load(
    MODEL_DIR / "new_SAEHD_decoder.npy",
    allow_pickle=True,
)

print("✓ Weights loaded")

# -------------------------------------------------
# Build model
# -------------------------------------------------

model = LIAEModel(
    encoder,
    inter_ab,
    inter_b,
    decoder,
)

print("✓ Model created")

# -------------------------------------------------
# Dummy inputs
# -------------------------------------------------

src = tf.random.uniform(
    (1, 128, 128, 3),
    dtype=tf.float32,
)

dst = tf.random.uniform(
    (1, 128, 128, 3),
    dtype=tf.float32,
)

print("✓ Dummy inputs created")

# -------------------------------------------------
# Forward pass
# -------------------------------------------------

rgb, mask = model(src, dst)

print()
print("=" * 80)
print("FORWARD PASS COMPLETE")
print("=" * 80)

print("RGB shape :", rgb.shape)
print("Mask shape:", mask.shape)

print()

if rgb.shape == (1, 128, 128, 3):
    print("✓ RGB output is correct")
else:
    print("✗ RGB output mismatch")

if mask.shape == (1, 128, 128, 1):
    print("✓ Mask output is correct")
else:
    print("✗ Mask output mismatch")

print()
print("=" * 80)
print("TEST FINISHED")
print("=" * 80)
