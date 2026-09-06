from pathlib import Path
import sys
import numpy as np

# DeepFaceLab source directory
DFL_ROOT = Path("third_party/DeepFaceLab")

# Add DeepFaceLab to Python import path
sys.path.insert(0, str(DFL_ROOT))

MODEL_DIR = Path("model")

FILES = [
    "new_SAEHD_encoder.npy",
    "new_SAEHD_inter_AB.npy",
    "new_SAEHD_inter_B.npy",
    "new_SAEHD_decoder.npy",
]


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

        print("NumPy type :", type(obj))
        print("dtype      :", obj.dtype)
        print("shape      :", obj.shape)

        if obj.dtype != object:
            print("Not an object array.")
            return

        flat = obj.reshape(-1)

        print("Object count:", len(flat))

        total_params = 0

        for i, item in enumerate(flat):

            print()
            print(f"[OBJECT {i}]")
            print("Type:", type(item))

            if isinstance(item, dict):

                for key, value in item.items():

                    if isinstance(value, np.ndarray):

                        params = int(np.prod(value.shape))
                        total_params += params

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
                total_params += params

                print(
                    f"  array: "
                    f"shape={item.shape}, "
                    f"dtype={item.dtype}, "
                    f"params={params:,}"
                )

        print()
        print("-" * 100)
        print(
            f"TOTAL PARAMETERS IN {filename}: "
            f"{total_params:,}"
        )
        print("-" * 100)

    except Exception as e:
        print()
        print("ERROR:", repr(e))


print()
print("=" * 100)
print("DEEPFACELAB SAEHD MODEL INSPECTOR")
print("=" * 100)

print()
print("Python:", sys.version)
print("NumPy:", np.__version__)

print()
print("Model directory:", MODEL_DIR.resolve())

print()
print("Checking DeepFaceLab source...")

archi_file = (
    DFL_ROOT
    / "core"
    / "leras"
    / "archis"
    / "DeepFakeArchi.py"
)

if archi_file.exists():
    print("DeepFaceLab source found.")
else:
    print("ERROR: DeepFaceLab source NOT FOUND")
    sys.exit(1)

print()

for filename in FILES:
    inspect_file(filename)

print()
print("=" * 100)
print("INSPECTION COMPLETE")
print("=" * 100)
