import numpy as np
import onnxruntime as ort
from pathlib import Path


MODEL_PATH = Path(
    "converter/output/LIAE_128_80_48_16_fp32.onnx"
)


print("=" * 70)
print("ONNX RUNTIME TEST")
print("=" * 70)


# =====================================================
# CREATE SESSION
# =====================================================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)


# =====================================================
# INPUTS
# =====================================================

print()
print("RUNTIME INPUTS")
print("-" * 70)

for inp in session.get_inputs():
    print("Name :", inp.name)
    print("Shape:", inp.shape)
    print("Type :", inp.type)


# =====================================================
# OUTPUTS
# =====================================================

print()
print("RUNTIME OUTPUTS")
print("-" * 70)

for out in session.get_outputs():
    print("Name :", out.name)
    print("Shape:", out.shape)
    print("Type :", out.type)


# =====================================================
# INPUT
# =====================================================

input_name = session.get_inputs()[0].name

dst = np.random.rand(
    1,
    128,
    128,
    3
).astype(np.float32)


# =====================================================
# INFERENCE
# =====================================================

print()
print("RUNNING TEST INFERENCE")
print("-" * 70)

outputs = session.run(
    None,
    {
        input_name: dst
    }
)


# =====================================================
# CHECK
# =====================================================

assert len(outputs) == 3

dst_mask = outputs[0]
swapped_face = outputs[1]
src_mask = outputs[2]


assert dst_mask.shape == (1, 128, 128, 1)
assert swapped_face.shape == (1, 128, 128, 3)
assert src_mask.shape == (1, 128, 128, 1)


assert np.isfinite(dst_mask).all()
assert np.isfinite(swapped_face).all()
assert np.isfinite(src_mask).all()


# =====================================================
# RESULTS
# =====================================================

print()
print("RESULTS")
print("-" * 70)

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


print()
print("=" * 70)
print("✓ ONNX RUNTIME TEST PASSED")
print("=" * 70)
