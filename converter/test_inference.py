import numpy as np
import tensorflow as tf
from pathlib import Path

from model import LIAEModel


MODEL_DIR = Path("model")


print("=" * 80)
print("LIAE FP32 INFERENCE TEST")
print("=" * 80)


# =====================================================
# LOAD WEIGHTS
# =====================================================

encoder = np.load(
    MODEL_DIR / "new_SAEHD_encoder.npy",
    allow_pickle=True
)

inter_ab = np.load(
    MODEL_DIR / "new_SAEHD_inter_AB.npy",
    allow_pickle=True
)

inter_b = np.load(
    MODEL_DIR / "new_SAEHD_inter_B.npy",
    allow_pickle=True
)

decoder = np.load(
    MODEL_DIR / "new_SAEHD_decoder.npy",
    allow_pickle=True
)

print("✓ Weights loaded")


# =====================================================
# CREATE MODEL
# =====================================================

model = LIAEModel(
    encoder,
    inter_ab,
    inter_b,
    decoder
)

print("✓ Model created")


# =====================================================
# DUMMY INPUT
# =====================================================

dst = tf.random.uniform(
    (1, 128, 128, 3),
    dtype=tf.float32
)

print("✓ Dummy input created")


# =====================================================
# INFERENCE
# =====================================================

swapped_face, dst_mask, src_mask = model(dst)


# =====================================================
# RESULTS
# =====================================================

print()
print("OUTPUT SHAPES")
print("-" * 80)

print(
    "Swapped face:",
    swapped_face.shape
)

print(
    "Destination mask:",
    dst_mask.shape
)

print(
    "Source mask:",
    src_mask.shape
)


# =====================================================
# VALIDATION
# =====================================================

assert swapped_face.shape == (
    1, 128, 128, 3
)

assert dst_mask.shape == (
    1, 128, 128, 1
)

assert src_mask.shape == (
    1, 128, 128, 1
)


# =====================================================
# RANGE
# =====================================================

print()
print("VALUE RANGES")
print("-" * 80)

print(
    "Swapped face:",
    float(tf.reduce_min(swapped_face)),
    "->",
    float(tf.reduce_max(swapped_face))
)

print(
    "Destination mask:",
    float(tf.reduce_min(dst_mask)),
    "->",
    float(tf.reduce_max(dst_mask))
)

print(
    "Source mask:",
    float(tf.reduce_min(src_mask)),
    "->",
    float(tf.reduce_max(src_mask))
)


# =====================================================
# FINISHED
# =====================================================

print()
print("=" * 80)
print("✓ INFERENCE TEST PASSED")
print("=" * 80)
