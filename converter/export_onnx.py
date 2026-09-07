import numpy as np
import tensorflow as tf
import tf2onnx

from pathlib import Path

from model import LIAEModel


# =====================================================
# PATHS
# =====================================================

MODEL_DIR = Path("model")

OUT_DIR = Path("converter/output")
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("EXPORTING DEEPFACELAB LIAE FP32 -> ONNX")
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


print("✓ Encoder loaded")
print("✓ Inter_AB loaded")
print("✓ Inter_B loaded")
print("✓ Decoder loaded")


# =====================================================
# BUILD MODEL
# =====================================================

liae = LIAEModel(
    encoder,
    inter_ab,
    inter_b,
    decoder
)


dummy = tf.random.uniform(
    (1, 128, 128, 3),
    dtype=tf.float32
)


outputs = liae(dummy)


print("✓ Model built")


# =====================================================
# VERIFY OUTPUTS
# =====================================================

swapped_face = outputs[0]
dst_mask = outputs[1]
src_mask = outputs[2]


print()
print("Tensor shapes:")
print(
    "swapped_face:",
    swapped_face.shape
)

print(
    "dst_mask:",
    dst_mask.shape
)

print(
    "src_mask:",
    src_mask.shape
)


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
# EXPORT WRAPPER
# =====================================================

class ExportModel(tf.keras.Model):

    def __init__(self, base):
        super().__init__()

        self.base = base

    def call(self, dst):

        swapped_face, dst_mask, src_mask = self.base(
            dst
        )

        return (
            dst_mask,
            swapped_face,
            src_mask
        )


export_model = ExportModel(liae)


# =====================================================
# BUILD EXPORT MODEL
# =====================================================

dst_in = tf.keras.Input(
    shape=(128, 128, 3),
    name="in_face"
)


export_model(dst_in)


# =====================================================
# OUTPUT
# =====================================================

onnx_path = (
    OUT_DIR /
    "LIAE_128_80_48_16_fp32.onnx"
)


# =====================================================
# TF2ONNX
# =====================================================

tf2onnx.convert.from_keras(
    export_model,

    input_signature=(
        tf.TensorSpec(
            (1, 128, 128, 3),
            tf.float32,
            name="in_face"
        ),
    ),

    opset=17,

    output_path=str(
        onnx_path
    )
)


# =====================================================
# DONE
# =====================================================

print()
print("=" * 80)
print("EXPORT COMPLETE")
print("=" * 80)

print(
    f"Saved: {onnx_path}"
)

print()
print("Expected ONNX:")
print("INPUT : in_face [1,128,128,3]")
print()
print("OUTPUT 0 : dst_mask  [1,128,128,1]")
print("OUTPUT 1 : swapped   [1,128,128,3]")
print("OUTPUT 2 : src_mask  [1,128,128,1]")
