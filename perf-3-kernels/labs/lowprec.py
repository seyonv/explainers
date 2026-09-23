"""Low-precision formats simulator (card: tensor-cores-low-precision.html).

numpy implementations of round-to-nearest-even into the FP8 (E4M3, E5M2) and FP4 (E2M1)
grids, OCP MX block scaling (32 elements, E8M0 power-of-two scale) and NVFP4 block
scaling (16 elements, E4M3 scale + FP32 per-tensor scale). Prints:
  1. the format table (max, min normal, min subnormal, binades), computed from the bit layout
  2. the H100/T4/B200 speed and byte ratios and the Llama-3.1-8B decode ceilings
  3. the worked example: one 32-value block of weights quantized four ways
  4. RMS relative error over 1M N(0,1) weights with 0.1% outliers, per format
These are our simulations of the number formats, not model-accuracy results.

python3 labs/lowprec.py   (numpy only, a few seconds)
"""
import numpy as np

# ------------------------------------------------------------------ formats
# (exponent bits, mantissa bits, bias, special-value rule)
#   "ieee": the all-ones exponent is reserved for inf/NaN (FP32, BF16, FP16, E5M2)
#   "e4m3": only S.1111.111 is NaN, no inf, so the top binade is usable up to 1.110
#   "none": every encoding is a number (E2M1)
FORMATS = {
    "FP32": (8, 23, 127, "ieee"),
    "BF16": (8, 7, 127, "ieee"),
    "FP16": (5, 10, 15, "ieee"),
    "E5M2": (5, 2, 15, "ieee"),
    "E4M3": (4, 3, 7, "e4m3"),
    "E2M1": (2, 1, 1, "none"),
}


def fmt_limits(name):
    e, m, bias, rule = FORMATS[name]
    if rule == "ieee":
        emax_field, mant_max = 2**e - 2, 2**m - 1
    elif rule == "e4m3":
        emax_field, mant_max = 2**e - 1, 2**m - 2  # 1111.111 is NaN, 1111.110 is max
    else:
        emax_field, mant_max = 2**e - 1, 2**m - 1
    vmax = 2.0 ** (emax_field - bias) * (1 + mant_max / 2**m)
    min_normal = 2.0 ** (1 - bias)
    min_sub = 2.0 ** (1 - bias - m)
    return vmax, min_normal, min_sub


def quantize(x, name):
    """Round-to-nearest-even onto the format's grid, saturating (SAT mode)."""
    e, m, bias, _ = FORMATS[name]
    vmax, _, _ = fmt_limits(name)
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    emin = 1 - bias
    with np.errstate(divide="ignore"):
        exp = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    exp = np.maximum(exp, emin)            # below min normal: subnormal spacing
    step = 2.0 ** (exp - m)                # spacing of the grid in this binade
    q = np.round(a / step) * step          # np.round = round half to even
    q = np.minimum(q, vmax)                # saturate
    return np.sign(x) * q


def grid(name):
    """All non-negative values of a small format (for printing)."""
    vmax, _, min_sub = fmt_limits(name)
    xs = np.arange(0, vmax + min_sub / 2, min_sub)
    return np.unique(quantize(xs, name))


# ------------------------------------------------------------------ schemes
def per_tensor_fp8(x):
    s = np.abs(x).max() / 448.0            # map the tensor's amax onto E4M3's max
    return quantize(x / s, "E4M3") * s, s


def mx(x, elem, k=32, round_up=False):
    """OCP MX v1.0 section 6.3: X = largest power of two <= amax, divided by the
    largest power of two of the element type; P = quantize(V / X), clamped.
    round_up=True is our variant: the smallest power of two with amax / X <= max, so
    nothing clips (not in the spec)."""
    emax_elem = {"E2M1": 2, "E4M3": 8, "E5M2": 15}[elem]
    b = x.reshape(-1, k)
    amax = np.abs(b).max(axis=1, keepdims=True)
    if round_up:
        X = 2.0 ** np.ceil(np.log2(amax / fmt_limits(elem)[0]))
    else:
        X = 2.0 ** (np.floor(np.log2(amax)) - emax_elem)   # an E8M0 value
    return (quantize(b / X, elem) * X).reshape(x.shape), X.ravel()


def nvfp4(x, k=16):
    """NVFP4: per-tensor FP32 scale so block scales fit E4M3, then per-16-block E4M3 scale.
    This is the common recipe (amax / (6 * 448) global scale); our implementation."""
    s_t = np.abs(x).max() / (6.0 * 448.0)
    b = x.reshape(-1, k)
    amax = np.abs(b).max(axis=1, keepdims=True)
    s_b = quantize(amax / 6.0 / s_t, "E4M3")            # the stored FP8 block scale
    s = np.where(s_b > 0, s_b * s_t, 1.0)
    return (quantize(b / s, "E2M1") * s).reshape(x.shape), s_b.ravel(), s_t


def int4_group(x, g=128):
    b = x.reshape(-1, g)
    s = np.abs(b).max(axis=1, keepdims=True) / 7.0      # symmetric, levels -7..7
    return (np.clip(np.round(b / s), -7, 7) * s).reshape(x.shape), s.ravel()


def rel_rms(q, x):
    return np.sqrt(np.mean((q - x) ** 2)) / np.sqrt(np.mean(x ** 2))


# ------------------------------------------------------------------ 1. formats
print("1. FORMAT TABLE (computed from the bit layout)")
print(f"{'format':6} {'bits':>4} {'bias':>4} {'max':>12} {'min normal':>12} {'min subnormal':>14} {'binades':>8}")
for name, (e, m, bias, _) in FORMATS.items():
    vmax, mn, ms = fmt_limits(name)
    print(f"{name:6} {1 + e + m:>4} {bias:>4} {vmax:>12.6g} {'2^%d' % np.log2(mn):>12} "
          f"{'2^%d' % np.log2(ms):>14} {np.log2(vmax / ms):>8.1f}")
print("E8M0 (MX scale): unsigned exponent, bias 127, 2^-127 .. 2^127, 0xFF = NaN, no zero")
print("E2M1 values:", ", ".join(f"{v:g}" for v in grid("E2M1")))
print("E4M3 values in [1, 2):", ", ".join(f"{v:g}" for v in grid("E4M3") if 1 <= v < 2))
print("E5M2 values in [1, 2):", ", ".join(f"{v:g}" for v in quantize(np.arange(1, 2, 1 / 64), 'E5M2')[::16]))
print("bits per element incl. scale:  MXFP8 (8*32+8)/32 = %.2f   MXFP6 = %.2f   MXFP4 = %.2f   "
      "NVFP4 (4*16+8)/16 = %.2f   INT4 g128 + FP16 scale = %.3f"
      % ((8 * 32 + 8) / 32, (6 * 32 + 8) / 32, (4 * 32 + 8) / 32, (4 * 16 + 8) / 16, (4 * 128 + 16) / 128))

# ------------------------------------------------------------------ 2. hardware
print("\n2. TENSOR-CORE SPEED AND THE DOUBLE LEVER")
T4 = dict(fp32=8.1e12, fp16=65e12, int8=130e12, int4=260e12, bw=320e9)
H100 = dict(fp32=67e12, bf16=989e12, fp8=1979e12, bw=3.35e12)
B200 = dict(bf16=2250e12, fp8=4500e12, fp4=9000e12, bw=8e12)
print(f"T4   fp16 tensor / fp32 = {T4['fp16'] / T4['fp32']:.1f}x   ops:byte fp32 {T4['fp32'] / T4['bw']:.0f}, "
      f"fp16 {T4['fp16'] / T4['bw']:.0f}, int8 {T4['int8'] / T4['bw']:.0f}")
print(f"H100 bf16 tensor / fp32 = {H100['bf16'] / H100['fp32']:.1f}x, fp8 / fp32 = {H100['fp8'] / H100['fp32']:.1f}x   "
      f"ops:byte fp32 {H100['fp32'] / H100['bw']:.0f}, bf16 {H100['bf16'] / H100['bw']:.0f}, fp8 {H100['fp8'] / H100['bw']:.0f}")
print(f"B200 ops:byte bf16 {B200['bf16'] / B200['bw']:.0f}, fp8 {B200['fp8'] / B200['bw']:.0f}, fp4 {B200['fp4'] / B200['bw']:.0f}")
P = 8.03e9
for label, bits in [("BF16", 16), ("FP8", 8), ("NVFP4 (weight-only on H100)", 4.5)]:
    gb = P * bits / 8
    print(f"Llama-3.1-8B {label:28} {gb / 1e9:6.2f} GB -> H100 batch-1 decode ceiling "
          f"{H100['bw'] / gb:5.0f} tok/s ({gb / H100['bw'] * 1e3:.2f} ms/step)")
for label, C in [("BF16", H100["bf16"]), ("FP8", H100["fp8"])]:
    print(f"prefill 1,000 tokens floor at 100% MFU, {label}: 2*8.03e9*1000/{C:.3g} = {2 * P * 1000 / C * 1e3:.1f} ms")
# Illustrative: batch 32, 4,096-token contexts, KV cache stays BF16 (128 KiB/token)
kv = 32 * 4096 * 131072
w16, w8 = P * 2, P * 1
print(f"illustrative decode step, batch 32 x 4,096 ctx: KV {kv / 1e9:.2f} GB + weights "
      f"{w16 / 1e9:.2f} -> {w8 / 1e9:.2f} GB: bytes/step {(kv + w16) / 1e9:.2f} -> {(kv + w8) / 1e9:.2f} GB "
      f"= {(kv + w16) / (kv + w8):.2f}x faster (not 2x)")

# ------------------------------------------------------------------ 3. worked example
print("\n3. WORKED EXAMPLE: 32 weights, N(0,1) x 0.02, one outlier (seed 0)")
rng = np.random.default_rng(0)
w = np.round(rng.normal(0, 0.02, 32), 4)
w[5] = 0.1500                                   # the outlier, in NVFP4 block 1
f8, s8 = per_tensor_fp8(w)
m4, X = mx(w, "E2M1")
n4, sb, st = nvfp4(w)
print(f"FP8 per-tensor scale s = amax/448 = {w.max():.4f}/448 = {s8:.4e}")
print(f"MXFP4 E8M0 scale X = 2^(floor(log2 {np.abs(w).max():.4f}) - 2) = 2^{int(np.log2(X[0]))} = {X[0]:.6g}")
print(f"NVFP4 tensor scale s_t = amax/(6*448) = {st:.4e}; block scales (E4M3) = "
      + ", ".join(f"{v:g}" for v in sb) + " -> effective " + ", ".join(f"{v * st:.4e}" for v in sb))
print(f"{'i':>3} {'weight':>8} {'FP8':>9} {'MXFP4':>9} {'NVFP4':>9}")
for i in range(32):
    print(f"{i:>3} {w[i]:>8.4f} {f8[i]:>9.5f} {m4[i]:>9.5f} {n4[i]:>9.5f}")
for lab, q in [("FP8 per-tensor", f8), ("MXFP4", m4), ("NVFP4", n4)]:
    print(f"{lab:15} rel RMS error: block 1 (0-15, outlier) {rel_rms(q[:16], w[:16]):.3f}  "
          f"block 2 (16-31) {rel_rms(q[16:], w[16:]):.3f}  all 32 {rel_rms(q, w):.3f}  "
          f"zeros {int(np.sum((q == 0) & (w != 0)))}")

# ------------------------------------------------------------------ 4. error sweep
print("\n4. RMS RELATIVE ERROR, 1,048,576 N(0,1) weights, 0.1% outliers x 20 (seed 1)")
rng = np.random.default_rng(1)
x = rng.normal(0, 1, 1 << 20)
idx = rng.choice(x.size, x.size // 1000, replace=False)
x[idx] *= 20.0
rows = [
    ("BF16", 16, quantize(x, "BF16")),
    ("FP8 E4M3 per-tensor", 8, per_tensor_fp8(x)[0]),
    ("MXFP8 (E4M3, k=32)", 8.25, mx(x, "E4M3")[0]),
    ("MXFP8 round-up scale", 8.25, mx(x, "E4M3", round_up=True)[0]),
    ("MXFP4 (E2M1, k=32)", 4.25, mx(x, "E2M1")[0]),
    ("MXFP4 round-up scale", 4.25, mx(x, "E2M1", round_up=True)[0]),
    ("NVFP4 (E2M1, k=16)", 4.5, nvfp4(x)[0]),
    ("INT4 per-group 128", 4.125, int4_group(x)[0]),
]
print(f"{'scheme':22} {'bits/elem':>9} {'rel RMS err':>12} {'SQNR dB':>8}")
for lab, bits, q in rows:
    r = rel_rms(q, x)
    print(f"{lab:22} {bits:>9.3f} {r:>12.5f} {-20 * np.log10(r):>8.1f}")
print(f"which NVFP4 ingredient matters? MXFP4 with k=16: {rel_rms(mx(x, 'E2M1', k=16)[0], x):.5f}   "
      f"NVFP4 with k=32: {rel_rms(nvfp4(x, k=32)[0], x):.5f}")
b = np.abs(x.reshape(-1, 32)).max(axis=1)
for elem, vmax, emax in [("E4M3", 448, 8), ("E2M1", 6, 2)]:
    r = b / 2.0 ** (np.floor(np.log2(b)) - emax)
    print(f"MX spec rule, {elem}: {np.mean(r > vmax):.1%} of blocks have amax/X > {vmax} and clip their largest value")

# Try this:
#   - set the outlier multiplier in part 4 to 1.0 (no outliers) and watch INT4 g128 catch up
#   - use mx(x, "E5M2") for MXFP8 with E5M2 elements: more range, less precision
