from pathlib import Path

print("=" * 100)
print("DEEPFACELAB LIAE EXACT ARCHITECTURE")
print("=" * 100)

# Official checkpoint configuration
resolution = 128
ae_dims = 128
e_dims = 80
d_dims = 48
d_mask_dims = 16

print()
print("CONFIG")
print("-" * 100)
print("resolution =", resolution)
print("archi      = liae")
print("ae_dims    =", ae_dims)
print("e_dims     =", e_dims)
print("d_dims     =", d_dims)
print("d_mask_dims=", d_mask_dims)

# ============================================================
# ENCODER
# ============================================================

print()
print("ENCODER")
print("-" * 100)

encoder_layers = [
    (3, 80),
    (80, 160),
    (160, 320),
    (320, 640),
]

res = resolution

encoder_total = 0

for i, (in_ch, out_ch) in enumerate(encoder_layers):

    out_res = res // 2

    params = (
        5 * 5 * in_ch * out_ch
        + out_ch
    )

    encoder_total += params

    print(
        f"down{i + 1}: "
        f"{res}x{res}x{in_ch} -> "
        f"{out_res}x{out_res}x{out_ch} "
        f"params={params:,}"
    )

    res = out_res

print()
print(
    "Encoder output:",
    f"{res}x{res}x640"
)

encoder_flat = res * res * 640

print(
    "Flattened:",
    encoder_flat
)

print(
    "Encoder total:",
    f"{encoder_total:,}"
)

# ============================================================
# INTER
# ============================================================

print()
print("INTER")
print("-" * 100)

lowest_dense_res = resolution // 16

print(
    "lowest_dense_res:",
    lowest_dense_res
)

dense1_params = (
    encoder_flat * ae_dims
    + ae_dims
)

dense2_output = (
    lowest_dense_res
    * lowest_dense_res
    * ae_dims * 2
)

dense2_params = (
    ae_dims * dense2_output
    + dense2_output
)

print()
print(
    f"Dense1: "
    f"{encoder_flat} -> {ae_dims}"
)

print(
    "Dense1 params:",
    f"{dense1_params:,}"
)

print()
print(
    f"Dense2: "
    f"{ae_dims} -> {dense2_output}"
)

print(
    "Dense2 params:",
    f"{dense2_params:,}"
)

print()
print(
    "Dense2 reshape:",
    f"{lowest_dense_res}x"
    f"{lowest_dense_res}x"
    f"{ae_dims * 2}"
)

# Inter upscale
inter_upscale_in = ae_dims * 2
inter_upscale_out = ae_dims * 2

inter_upscale_params = (
    3 * 3
    * inter_upscale_in
    * (inter_upscale_out * 4)
    + (inter_upscale_out * 4)
)

print()
print(
    "Inter upscale:",
    f"{lowest_dense_res}x"
    f"{lowest_dense_res}x"
    f"{inter_upscale_in}"
    " -> "
    f"{lowest_dense_res * 2}x"
    f"{lowest_dense_res * 2}x"
    f"{inter_upscale_out}"
)

print(
    "Inter upscale params:",
    f"{inter_upscale_params:,}"
)

inter_total = (
    dense1_params
    + dense2_params
    + inter_upscale_params
)

print()
print(
    "Inter total:",
    f"{inter_total:,}"
)

# ============================================================
# DECODER
# ============================================================

print()
print("DECODER")
print("-" * 100)

decoder_layers = [
    ("upscale0", 256, 384),
    ("upscale1", 384, 192),
    ("upscale2", 192, 96),
]

decoder_total = 0

for name, in_ch, out_ch in decoder_layers:

    params = (
        3 * 3
        * in_ch
        * (out_ch * 4)
        + (out_ch * 4)
    )

    decoder_total += params

    print(
        f"{name}: "
        f"{in_ch} -> {out_ch} "
        f"(DepthToSpace x2) "
        f"params={params:,}"
    )

# Residual blocks

print()
print("RESIDUAL BLOCKS")
print("-" * 100)

for name, ch in [
    ("res0", 384),
    ("res1", 192),
    ("res2", 96),
]:

    params = (
        2 * (
            3 * 3 * ch * ch
            + ch
        )
    )

    decoder_total += params

    print(
        f"{name}: "
        f"channels={ch} "
        f"params={params:,}"
    )

# RGB output

out_conv_params = (
    1 * 1 * 96 * 3
    + 3
)

decoder_total += out_conv_params

print()
print(
    "out_conv:",
    "96 -> 3",
    f"params={out_conv_params:,}"
)

# Mask branch

print()
print("MASK BRANCH")
print("-" * 100)

mask_layers = [
    ("upscalem0", 256, 128),
    ("upscalem1", 128, 64),
    ("upscalem2", 64, 32),
]

for name, in_ch, out_ch in mask_layers:

    params = (
        3 * 3
        * in_ch
        * (out_ch * 4)
        + (out_ch * 4)
    )

    decoder_total += params

    print(
        f"{name}: "
        f"{in_ch} -> {out_ch} "
        f"(DepthToSpace x2) "
        f"params={params:,}"
    )

mask_out_params = (
    1 * 1 * 32
    + 1
)

decoder_total += mask_out_params

print()
print(
    "out_convm:",
    "32 -> 1",
    f"params={mask_out_params:,}"
)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print()
print(
    "Encoder:",
    f"{encoder_total:,}"
)

print(
    "Inter:",
    f"{inter_total:,}"
)

print(
    "Inter_AB + Inter_B:",
    f"{inter_total * 2:,}"
)

print(
    "Decoder:",
    f"{decoder_total:,}"
)

grand_total = (
    encoder_total
    + inter_total * 2
    + decoder_total
)

print()
print(
    "CALCULATED TOTAL:",
    f"{grand_total:,}"
)

print()
print("ACTUAL CHECKPOINT TOTAL:")
print("42,773,412")

print()
difference = grand_total - 42773412

print(
    "Difference:",
    f"{difference:,}"
)

print()
print("=" * 100)
print("DONE")
print("=" * 100)
