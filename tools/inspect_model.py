from pathlib import Path
import sys
import numpy as np

DeepFaceLab source is downloaded into third_party/DeepFaceLab

DFL_ROOT = Path("third_party/DeepFaceLab")

sys.path.insert(0, str(DFL_ROOT))

from core.leras import nn

MODEL_DIR = Path("model")

FILES = [
"new_SAEHD_encoder.npy",
"new_SAEHD_inter_AB.npy",
"new_SAEHD_inter_B.npy",
"new_SAEHD_decoder.npy",
]

def count_weights(weights):
total = 0

for w in weights:
    try:
        total += int(np.prod(w.shape))
    except Exception:
        pass

return total

def inspect_file(filename):
path = MODEL_DIR / filename

print()
print("=" * 100)
print(filename)
print("=" * 100)

if not path.exists():
    print("MISSING:", path)
    return

size_mb = path.stat().st_size / 1024 / 1024

print(f"File size: {size_mb:.2f} MB")

try:
    obj = np.load(path, allow_pickle=True)

    print("Top-level dtype :", obj.dtype)
    print("Top-level shape :", obj.shape)

    if obj.dtype != object:
        print("Not an object array.")
        return

    flat = obj.reshape(-1)

    print("Object count:", len(flat))

    total = 0

    for i, item in enumerate(flat):

        print()
        print(f"[OBJECT {i}]")
        print("Type:", type(item))

        if isinstance(item, dict):

            for key, value in item.items():

                if isinstance(value, np.ndarray):

                    params = int(np.prod(value.shape))
                    total += params

                    print(
                        f"  {key}: "
                        f"shape={value.shape}, "
                        f"dtype={value.dtype}, "
                        f"params={params:,}"
                    )

                else:
                    print(
                        f"  {key}: "
                        f"type={type(value)}"
                    )

        elif isinstance(item, np.ndarray):

            params = int(np.prod(item.shape))
            total += params

            print(
                "  array:",
                f"shape={item.shape},",
                f"dtype={item.dtype},",
                f"params={params:,}"
            )

    print()
    print("-" * 100)
    print(f"TOTAL PARAMETERS IN {filename}: {total:,}")
    print("-" * 100)

except Exception as e:

    print()
    print("ERROR")
    print(repr(e))

print()
print("=" * 100)
print("DeepFaceLab SAEHD MODEL INSPECTOR")
print("=" * 100)

print()
print("Python:", sys.version)
print("NumPy:", np.version)

try:
print("TensorFlow:", nn.tf.version)
except Exception as e:
print("TensorFlow version unavailable:", repr(e))

print()
print("Model directory:", MODEL_DIR.resolve())

for filename in FILES:
inspect_file(filename)

print()
print("=" * 100)
print("INSPECTION COMPLETE")
print("=" * 100)
