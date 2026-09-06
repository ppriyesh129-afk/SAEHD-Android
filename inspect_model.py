import numpy as np
from pathlib import Path

MODEL_DIR = Path("model")

files = [
    "new_SAEHD_encoder.npy",
    "new_SAEHD_inter_AB.npy",
    "new_SAEHD_inter_B.npy",
    "new_SAEHD_decoder.npy",
]

total_params = 0

for filename in files:
    path = MODEL_DIR / filename

    print("\n" + "=" * 80)
    print(filename)
    print("=" * 80)

    if not path.exists():
        print("MISSING:", path)
        continue

    try:
        data = np.load(path, allow_pickle=True)

        print("Container type :", type(data))
        print("dtype          :", getattr(data, "dtype", None))
        print("shape          :", getattr(data, "shape", None))

        if isinstance(data, np.ndarray) and data.dtype == object:
            print("Object array detected")

            flat = data.reshape(-1)

            for i, item in enumerate(flat):
                print(f"\n[{i}]")
                print("  type:", type(item))

                if isinstance(item, dict):
                    for key, value in item.items():
                        if hasattr(value, "shape"):
                            n = int(np.prod(value.shape))
                            total_params += n
                            print(
                                f"  {key}: "
                                f"shape={value.shape}, "
                                f"dtype={value.dtype}, "
                                f"params={n:,}"
                            )
                        else:
                            print(f"  {key}: {type(value)}")

                elif hasattr(item, "shape"):
                    n = int(np.prod(item.shape))
                    total_params += n
                    print(
                        f"  shape={item.shape}, "
                        f"dtype={item.dtype}, "
                        f"params={n:,}"
                    )

        elif isinstance(data, np.ndarray):
            n = int(np.prod(data.shape))
            total_params += n
            print("Parameters    :", f"{n:,}")

    except Exception as e:
        print("ERROR:", repr(e))

print("\n" + "=" * 80)
print("TOTAL")
print("=" * 80)
print(f"Total counted elements: {total_params:,}")
