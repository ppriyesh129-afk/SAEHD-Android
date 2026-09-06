from pathlib import Path
import numpy as np

MODEL_DIR = Path("model")

FILES = [
    "new_SAEHD_encoder.npy",
    "new_SAEHD_inter_AB.npy",
    "new_SAEHD_inter_B.npy",
    "new_SAEHD_decoder.npy",
]

def inspect(path):
    print("\n" + "=" * 80)
    print(path.name)
    print("=" * 80)

    if not path.exists():
        print("MISSING")
        return

    print("File size:", f"{path.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        obj = np.load(path, allow_pickle=True)

        print("NumPy type :", type(obj))
        print("dtype      :", obj.dtype)
        print("shape      :", obj.shape)

        if obj.dtype == object:
            flat = obj.reshape(-1)

            print("objects    :", len(flat))

            for i, item in enumerate(flat):
                print(f"\nObject {i}:")
                print("  type:", type(item))

                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, np.ndarray):
                            print(
                                f"  {key}: "
                                f"shape={value.shape}, "
                                f"dtype={value.dtype}, "
                                f"elements={value.size:,}"
                            )
                        else:
                            print(f"  {key}: {type(value)}")

                elif isinstance(item, np.ndarray):
                    print(
                        "  shape:",
                        item.shape,
                        "dtype:",
                        item.dtype,
                        "elements:",
                        f"{item.size:,}",
                    )

    except Exception as e:
        print("ERROR:", repr(e))


for filename in FILES:
    inspect(MODEL_DIR / filename)
