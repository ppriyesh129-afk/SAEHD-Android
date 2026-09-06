from pathlib import Path

print("=" * 100)
print("DEEPFACELAB SAEHD LIAE-UD EXACT GRAPH")
print("=" * 100)

resolution = 128
ae_dims = 256
e_dims = 64
d_dims = 64
d_mask_dims = 22

print()
print("CONFIG")
print("-" * 100)
print("resolution =", resolution)
print("archi      = liae-ud")
print("ae_dims    =", ae_dims)
print("e_dims     =", e_dims)
print("d_dims     =", d_dims)
print("d_mask_dims=", d_mask_dims)

# ---------------------------------------------------------
# Encoder
# ---------------------------------------------------------

encoder_out_res = resolution // 16
encoder_out_ch = e_dims * 8

print()
print("ENCODER")
print("-" * 100)

print(
    f"Input                         : "
    f"{resolution} x {resolution} x 3"
)

ch = 3

for i in range(4):
    out_ch = e_dims * min(2 ** i, 8)

    in_res = resolution // (2 ** i)
    out_res = resolution // (2 ** (i + 1))

    params = (5 * 5 * ch * out_ch) + out_ch

    print(
        f"down{i + 1} "
        f"{in_res:>3}x{in_res:<3}x{ch:<4} -> "
        f"{out_res:>3}x{out_res:<3}x{out_ch:<4} "
        f"params={params:,}"
    )

    ch = out_ch

encoder_params = sum(
    (5 * 5 * (e_dims * min(2 ** i, 8)) *
     (e_dims * min(2 ** i, 8)))
    for i in range(4)
)

# Correct first layer separately.
encoder_params = (
    5 * 5 * 3 * e_dims + e_dims
)

for i in range(1, 4):
    in_ch = e_dims * min(2 ** (i - 1), 8)
    out_ch = e_dims * min(2 ** i, 8)

    encoder_params += (
        5 * 5 * in_ch * out_ch
        + out_ch
    )

print()
print(
    f"Encoder flattened tensor : "
    f"{encoder_out_res} x {encoder_out_res} x {encoder_out_ch}"
)

print(
    f"Encoder flattened size   : "
    f"{encoder_out_res * encoder_out_res * encoder_out_ch:,}"
)

print(
    f"Encoder parameters       : "
    f"{encoder_params:,}"
)

# ---------------------------------------------------------
# LIAE-UD Inter
# ---------------------------------------------------------

lowest_dense_res = resolution // 32

inter_input = (
    encoder_out_res *
    encoder_out_res *
    encoder_out_ch
)

inter_output = (
    lowest_dense_res *
    lowest_dense_res *
    ae_dims
)

print()
print("INTER")
print("-" * 100)

print(
    f"Encoder input            : "
    f"{inter_input:,}"
)

print(
    f"Dense1                   : "
    f"{inter_input} -> {ae_dims}"
)

dense1_params = (
    inter_input * ae_dims
    + ae_dims
)

print(
    f"Dense1 parameters        : "
    f"{dense1_params:,}"
)

print(
    f"Dense2                   : "
    f"{ae_dims} -> "
    f"{lowest_dense_res}x"
    f"{lowest_dense_res}x"
    f"{ae_dims}"
)

dense2_params = (
    ae_dims * inter_output
    + inter_output
)

print(
    f"Dense2 parameters        : "
    f"{dense2_params:,}"
)

# Upscale in Inter.
# Conv input = ae_dims
# Conv output = ae_dims * 4
inter_upscale_params = (
    3 * 3 * ae_dims * (ae_dims * 4)
    + (ae_dims * 4)
)

print(
    f"Inter Upscale            : "
    f"{lowest_dense_res}x"
    f"{lowest_dense_res}x"
    f"{ae_dims} -> "
    f"{lowest_dense_res * 2}x"
    f"{lowest_dense_res * 2}x"
    f"{ae_dims}"
)

print(
    f"Inter Upscale parameters: "
    f"{inter_upscale_params:,}"
)

inter_total = (
    dense1_params
    + dense2_params
    + inter_upscale_params
)

print(
    f"Inter total              : "
    f"{inter_total:,}"
)

# ---------------------------------------------------------
# LIAE structure
# ---------------------------------------------------------

print()
print("LIAE")
print("-" * 100)

print(
    "Encoder output           : "
    f"{encoder_out_res}x"
    f"{encoder_out_res}x"
    f"{encoder_out_ch}"
)

print(
    "Inter output             : "
    f"{lowest_dense_res * 2}x"
    f"{lowest_dense_res * 2}x"
    f"{ae_dims}"
)

print()
print("LIAE-UD uses two Inter branches:")
print("  Inter_AB")
print("  Inter_B")

print()
print(
    "Decoder input channels   : "
    f"{ae_dims * 2}"
)

# ---------------------------------------------------------
# Decoder
# ---------------------------------------------------------

print()
print("DECODER")
print("-" * 100)

in_ch = ae_dims
d_ch = d_dims
mask_ch = d_mask_dims

print()
print("IMAGE BRANCH")

# upscale0
out_ch = d_ch * 8
params = 3 * 3 * in_ch * (out_ch * 4) + (out_ch * 4)

print(
    f"upscale0                 : "
    f"{in_ch} -> {out_ch}"
    f" (depth-to-space x2)"
)
print(f"parameters               : {params:,}")

# upscale1
in_ch_1 = d_ch * 8
out_ch_1 = d_ch * 4

params = (
    3 * 3 * in_ch_1 * (out_ch_1 * 4)
    + out_ch_1 * 4
)

print(
    f"upscale1                 : "
    f"{in_ch_1} -> {out_ch_1}"
    f" (depth-to-space x2)"
)
print(f"parameters               : {params:,}")

# upscale2
in_ch_2 = d_ch * 4
out_ch_2 = d_ch * 2

params = (
    3 * 3 * in_ch_2 * (out_ch_2 * 4)
    + out_ch_2 * 4
)

print(
    f"upscale2                 : "
    f"{in_ch_2} -> {out_ch_2}"
    f" (depth-to-space x2)"
)
print(f"parameters               : {params:,}")

print()
print("RESIDUAL BLOCKS")

for name, ch in [
    ("res0", d_ch * 8),
    ("res1", d_ch * 4),
    ("res2", d_ch * 2),
]:
    params = (
        2 * (3 * 3 * ch * ch + ch)
    )

    print(
        f"{name:<8} channels={ch:<4} "
        f"parameters={params:,}"
    )

# ---------------------------------------------------------
# Image outputs for -d
# ---------------------------------------------------------

print()
print("LIAE-UD IMAGE OUTPUT")
print("-" * 100)

for i in range(4):
    params = (
        3 * 3 * (d_ch * 2) * 3
        + 3
    )

    print(
        f"out_conv{i + 1}           : "
        f"{d_ch * 2} -> 3, "
        f"3x3, parameters={params:,}"
    )

print()
print(
    "Four RGB outputs are concatenated "
    "and depth-to-space x2 is applied."
)

print(
    "Therefore final image resolution = "
    f"{resolution} x {resolution}"
)

# ---------------------------------------------------------
# MASK BRANCH
# ---------------------------------------------------------

print()
print("MASK BRANCH")
print("-" * 100)

mask_specs = [
    ("upscalem0", in_ch, mask_ch * 8),
    ("upscalem1", mask_ch * 8, mask_ch * 4),
    ("upscalem2", mask_ch * 4, mask_ch * 2),
    ("upscalem3", mask_ch * 2, mask_ch),
]

for name, in_c, out_c in mask_specs:
    params = (
        3 * 3 * in_c * (out_c * 4)
        + out_c * 4
    )

    print(
        f"{name:<10}: "
        f"{in_c} -> {out_c}, "
        f"parameters={params:,}"
    )

mask_out_params = (
    mask_ch * 1 * 1
    + 1
)

print(
    f"out_convm               : "
    f"{mask_ch} -> 1, "
    f"1x1, parameters={mask_out_params:,}"
)

# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print()
print("=" * 100)
print("IMPORTANT")
print("=" * 100)

print()
print(
    "This report describes the exact DeepFaceLab "
    "source architecture for:"
)

print(
    "resolution=128, archi=liae-ud, "
    "ae_dims=256, e_dims=64, "
    "d_dims=64, d_mask_dims=22"
)

print()
print(
    "NEXT STEP:"
)

print(
    "Obtain the matching LIAE-UD checkpoint and "
    "compare every stored tensor against this graph."
)

print()
print("=" * 100)
