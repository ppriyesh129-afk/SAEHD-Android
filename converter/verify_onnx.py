import numpy as np
import onnxruntime as ort
from pathlib import Path


MODEL_PATH = Path(
    "converter/output/LIAE_128_80_48_16_fp32.onnx"
)


print("=" * 80)
print("VERIFYING LIAE FP32 ONNX")
print("=" * 80)


# =====================================================
# LOAD
# =====================================================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)


print("✓ ONNX structure is valid")


# =====================================================
# INPUTS
# =====================================================

print()
print("Inputs:")

for inp in session.get_inputs():
    print(
        f"  {inp.name} -> {inp.shape}"
    )


# =====================================================
# OUTPUTS
# =====================================================

print()
print("Outputs:")

for out in session.get_outputs():
    print(
        f"  {out.name} -> {out.shape}"
    )


# =====================================================
# CREATE TEST INPUT
# =====================================================

inp = session.get_inputs()[0]

dst = np.random.rand(
    1,
    128,
    128,
    3
).astype(np.float32)


# =====================================================
# RUN
# =====================================================

outputs = session.run(
    None,
    {
        inp.name: dst
    }
)


# =====================================================
# VALIDATE
# =====================================================

assert len(outputs) == 3, (
    f"Expected 3 outputs, got {len(outputs)}"
)


dst_mask = outputs[0]
swapped_face = outputs[1]
src_mask = outputs[2]


assert dst_mask.shape == (
    1, 128, 128, 1
)

assert swapped_face.shape == (
    1, 128, 128, 3
)

assert src_mask.shape == (
    1, 128, 128, 1
)


# =====================================================
# RANGE
# =====================================================

print()
print("Output ranges:")
print("-" * 80)

print(
    "Destination mask:",
    float(dst_mask.min()),
    "->",
    float(dst_mask.max())
)

print(
    "Swapped face:",
    float(swapped_face.min()),
    "->",
    float(swapped_face.max())
)

print(
    "Source mask:",
    float(src_mask.min()),
    "->",
    float(src_mask.max())
)


# =====================================================
# NaN / INF CHECK
# =====================================================

assert np.isfinite(dst_mask).all()
assert np.isfinite(swapped_face).all()
assert np.isfinite(src_mask).all()


print()
print("=" * 80)
print("✓ ONNX INFERENCE TEST PASSED")
print("=" * 80)
