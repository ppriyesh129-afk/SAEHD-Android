import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

model_path = Path("converter/output/LIAE_128_80_48_16_fp32.onnx")

print("=" * 80)
print("VERIFYING LIAE FP32 ONNX")
print("=" * 80)

onnx_model = onnx.load(str(model_path))
onnx.checker.check_model(onnx_model)
print("✓ ONNX structure is valid")

session = ort.InferenceSession(str(model_path))

print("\nInputs:")
for inp in session.get_inputs():
    print(f"  {inp.name} -> {inp.shape}")

print("\nOutputs:")
for out in session.get_outputs():
    print(f"  {out.name} -> {out.shape}")

src = np.random.rand(1,128,128,3).astype(np.float32)
dst = np.random.rand(1,128,128,3).astype(np.float32)

outputs = session.run(None, {
    session.get_inputs()[0].name: src,
    session.get_inputs()[1].name: dst
})

print("\nOutput shapes:")
for i, o in enumerate(outputs):
    print(f"  Output {i}: {o.shape}")

print("=" * 80)
print("ONNX VERIFIED")
print("=" * 80)
