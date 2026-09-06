from pathlib import Path
import sys
import numpy as np

DFL_ROOT = Path("third_party/DeepFaceLab")
MODEL_DIR = Path("model")

sys.path.insert(0, str(DFL_ROOT))

FILES = [
    "new_SAEHD_encoder.npy",
    "new_SAEHD_inter_AB.npy",
    "new_SAEHD_inter_B.npy",
    "new_SAEHD_decoder.npy",
]


def inspect_value(name, value, indent="  "):
    if isinstance(value, np.ndarray):
        params = int(np.prod(value.shape))

        print(
            f"{indent}{name}: "
            f"shape={value.shape}, "
            f"dtype={value.dtype}, "
            f"params={params:,}"
        )

        return params

    if isinstance(value, dict):
        print(f"{indent}{name}: DICT")

        total = 0

        for key, child in value.items():
            total += inspect_value(str(key), child, indent + "  ")

        return total

    print(
        f"{indent}{name}: "
        f"type={type(value).__name__}, "
        f"value={repr(value)[:200]}"
    )

    return 0


def inspect_file(filename):
    path = MODEL_DIR / filename

    print()
    print("=" * 100)
    print(filename)
    print("=" * 100)

    if not path.exists():
        print("MISSING:", path)
        return

    print(
        "File size:",
        f"{path.stat().st_size / 1024 / 1024:.2f} MB"
    )

    try:
        obj = np.load(path, allow_pickle=True)

        print("Loaded type:", type(obj).__name__)

        if isinstance(obj, dict):

            print("Dictionary entries:", len(obj))

            total_params = 0

            for key, value in obj.items():
                total_params += inspect_value(
                    str(key),
                    value
                )

            print()
            print("-" * 100)
            print(
                f"TOTAL NUMPY PARAMETERS: "
                f"{total_params:,}"
            )
            print("-" * 100)

        elif isinstance(obj, np.ndarray):

            print("Array shape:", obj.shape)
            print("Array dtype:", obj.dtype)

        else:

            print("Unexpected object:")
            print(repr(obj)[:1000])

    except Exception as e:

        print()
        print("ERROR:", repr(e))


print()
print("=" * 100)
print("DEEPFACELAB SAEHD EXACT WEIGHT INSPECTOR")
print("=" * 100)

print()
print("Python:", sys.version)
print("NumPy:", np.__version__)

archi_file = (
    DFL_ROOT
    / "core"
    / "leras"
    / "archis"
    / "DeepFakeArchi.py"
)

print()
print("Checking DeepFaceLab source...")

if not archi_file.exists():
    print("ERROR: DeepFaceLab source not found")
    sys.exit(1)

print("DeepFaceLab source found.")

print()
print("Model directory:", MODEL_DIR.resolve())

grand_total = 0

for filename in FILES:
    inspect_file(filename)

print()
print("=" * 100)
print("INSPECTION COMPLETE")
print("=" * 100)
