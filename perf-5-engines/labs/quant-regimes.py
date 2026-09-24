"""Quantization for serving: which format speeds up which phase, plus a SmoothQuant toy.

Run: python3 labs/quant-regimes.py   (numpy only, CPU, about a second)

Part 1 is the course's step-time model (Scaling Book Part 7, course 2 "step time"):

  decode step = B * KV_seq / BW  +  max( 2 * B * N / C ,  weight_bytes / BW )
                attention: reads     MLP math at the       reading every
                every KV cache       format's tensor-core  weight once
                (BF16, unchanged)    rate C

applied to Llama-3.1-8B on one H100 SXM at 2,048 tokens of context, for three formats:
  BF16    2 bytes/param, BF16 math (989 TFLOPS)
  W4A16   INT4 weights + one FP16 scale per group of 128 = 0.5 + 2/128 bytes/param,
          math still BF16: weights are dequantized to BF16 inside the kernel (GPTQ, AWQ)
  W8A8    INT8 (SmoothQuant) or FP8 weights AND activations, 1 byte/param,
          math at the 8-bit tensor-core rate (1,979 TFLOPS dense FP8/INT8)
Upper bounds: perfect overlap, peak bandwidth, no dequantization cost, no non-GEMM ops.
All weights (8.03B) are quantized, matching the course's 16.06 GB "read every step".

Part 2 is a numpy toy of SmoothQuant on SYNTHETIC data: activations with 3 outlier
channels (~100x the rest), random weights, per-tensor symmetric INT8 on both, and the
error of X @ W before and after smoothing with s_j = max|X_j|^a / max|W_j|^(1-a).
"""
import numpy as np

GB = 1e9
N = 8.03e9                     # parameters
KV_TOK = 131072                # 2 * L * K * H * 2 bytes = 128 KiB per token (BF16 KV)
CTX = 2048
BW, HBM = 3.35e12, 80e9        # H100 SXM
C16, C8 = 989e12, 1979e12      # dense BF16 and FP8/INT8 tensor-core peaks
GROUP = 128

FORMATS = {
    #          bytes per parameter           math rate
    "BF16":  (2.0,                            C16),
    "W4A16": (0.5 + 2 / GROUP,                C16),
    "W8A8":  (1.0,                            C8),
}


def step(B, bpp, C, ctx=CTX):
    kv = B * ctx * KV_TOK / BW
    math = 2 * B * N / C
    w = N * bpp / BW
    return kv + max(math, w), kv, math, w


print("== 1. Llama-3.1-8B on one H100, 2,048-token context: decode step time by format ==")
kv_seq = CTX * KV_TOK
print(f"KV per sequence: {CTX} x {KV_TOK:,} B = {kv_seq / GB:.4f} GB (BF16 in every row: KV quantization is separate)")
bmax = {}
for name, (bpp, C) in FORMATS.items():
    wb = N * bpp
    bmax[name] = int((HBM - wb) // kv_seq)
    bcrit = C * bpp / (2 * BW)
    print(f"{name:6s}: {bpp:.4f} B/param -> weights {wb / GB:5.2f} GB, weight read {wb / BW * 1e3:.2f} ms, "
          f"HBM fits B <= {bmax[name]}, MLP B_crit (math = weight read) = C*bytes/(2*BW) = {bcrit:.1f}")

print(f"\n{'B':>4} | " + " | ".join(f"{n + ' ms':>9} {'tok/s':>7}" for n in FORMATS) +
      f" | {'W4A16 x':>7} | {'W8A8 x':>6} | note")
BS = [1, 2, 4, 8, 16, 32, 64, 76, 128, 238, 256, 268, 282, 295, 384, 512]
rows = {}
for B in BS:
    t = {n: step(B, *FORMATS[n])[0] for n in FORMATS}
    rows[B] = t
    oom = [n for n in FORMATS if B > bmax[n]]
    note = ("does not fit in 80 GB: " + ", ".join(oom)) if oom else ""
    print(f"{B:4d} | " + " | ".join(f"{t[n] * 1e3:9.2f} {B / t[n]:7,.0f}" for n in FORMATS) +
          f" | {t['BF16'] / t['W4A16']:6.2f}x | {t['BF16'] / t['W8A8']:5.2f}x | {note}")

cross = next(B for B in range(1, 513) if step(B, *FORMATS["W8A8"])[0] < step(B, *FORMATS["W4A16"])[0])
print(f"\nW8A8 overtakes W4A16 at B = {cross}: there W4A16's BF16 math (2*B*N/989e12) passes W8A8's 8.03 GB weight read")

print("\n-- the two rows the card uses, term by term --")
for B in (1, 238):
    for n, (bpp, C) in FORMATS.items():
        tot, kv, m, w = step(B, bpp, C)
        bound = "math" if m > w else "weights"
        print(f"B={B:3d} {n:6s}: KV {kv * 1e3:6.2f} + max(math {m * 1e3:5.2f}, weights {w * 1e3:5.2f}) = "
              f"{tot * 1e3:6.2f} ms -> {B / tot:7,.0f} tok/s  ({rows[B]['BF16'] / tot:.2f}x vs BF16; MLP {bound}-bound)")

print("\n-- each format at its own maximum batch (HBM full at 2k context) --")
for n, (bpp, C) in FORMATS.items():
    B = bmax[n]
    tot = step(B, bpp, C)[0]
    print(f"{n:6s}: B = {B}: step {tot * 1e3:.2f} ms, {B / tot:,.0f} tok/s total, {1 / tot:.1f} tok/s per user")

print("\n-- prefill of a 1,000-token prompt (compute-bound): max(2*N*n / C, weights / BW) --")
for n, (bpp, C) in FORMATS.items():
    m, w = 2 * N * 1000 / C, N * bpp / BW
    print(f"{n:6s}: max({m * 1e3:5.2f}, {w * 1e3:4.2f}) = {max(m, w) * 1e3:5.2f} ms "
          f"({2 * N * 1000 / C16 / max(m, w):.2f}x vs BF16)")

# ------------------------------------------------------------------ Part 2
print("\n== 2. SmoothQuant toy (SYNTHETIC data, seed 0): per-tensor INT8 on X and W ==")
rng = np.random.default_rng(0)
T, CI, CO = 256, 512, 512
X = rng.normal(0, 1, (T, CI))
OUT = [7, 200, 431]                                   # 3 outlier input channels
for j in OUT:
    X[:, j] = rng.choice([-1, 1], T) * rng.normal(100, 10, T)   # ~100x, in every token
W = rng.normal(0, 0.02, (CI, CO))                     # flat, easy-to-quantize weights
Y = X @ W


def q8(a):
    """Symmetric per-tensor INT8: step = max|a| / 127."""
    d = np.abs(a).max() / 127
    return np.clip(np.round(a / d), -127, 127) * d, d


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


ax, aw = np.abs(X).max(0), np.abs(W).max(1)           # per input channel j
print(f"X: {T} tokens x {CI} channels N(0,1); channels {OUT} = +-N(100, 10) in every token; "
      f"W: {CI}x{CO} N(0, 0.02^2)")
print(f"max|X| normal channels (median) {np.median(np.delete(ax, OUT)):.2f}, outlier channels {ax[OUT].min():.1f}-{ax[OUT].max():.1f}")
print(f"\n{'smoothing':>13} | {'max|X^|':>8} | {'X step':>8} | {'levels, normal ch':>17} | "
      f"{'X err':>6} | {'X err, normal ch':>16} | {'W err':>6} | {'Y = XW err':>10}")
res = {}
for a in (None, 0.0, 0.25, 0.5, 0.75, 1.0):
    s = np.ones(CI) if a is None else ax ** a / aw ** (1 - a)
    Xs, Ws = X / s, W * s[:, None]                    # X diag(s)^-1 . diag(s) W = X W exactly
    assert np.allclose(Xs @ Ws, Y)
    Xq, dx = q8(Xs)
    Wq, _ = q8(Ws)
    lev = 2 * np.median(np.delete(np.abs(Xs).max(0), OUT)) / dx   # distinct INT8 steps a normal channel spans
    nm = np.delete(np.arange(CI), OUT)
    ex, exn, ew, ey = rel(Xq, Xs), rel(Xq[:, nm], Xs[:, nm]), rel(Wq, Ws), rel(Xq @ Wq, Y)
    res[a] = ey
    lab = "none" if a is None else f"alpha={a}"
    print(f"{lab:>13} | {np.abs(Xs).max():8.3f} | {dx:8.5f} | {lev:17.1f} | {ex:6.4f} | {exn:16.4f} | {ew:6.4f} | {ey:10.4f}")
print(f"\nerror of Y: none {res[None]:.4f} -> alpha 0.5 {res[0.5]:.4f} = {res[None] / res[0.5]:.1f}x smaller")
s5 = ax ** 0.5 / aw ** 0.5
j0 = [j for j in range(CI) if j not in OUT][0]
print(f"alpha=0.5 example factors: outlier channel {OUT[0]}: sqrt({ax[OUT[0]]:.1f} / {aw[OUT[0]]:.4f}) = {s5[OUT[0]]:.1f}; "
      f"normal channel {j0}: sqrt({ax[j0]:.2f} / {aw[j0]:.4f}) = {s5[j0]:.1f}")
print("SmoothQuant paper: alpha = 0.5 for OPT and BLOOM, 0.75 for GLM-130B; ablation sweet spot 0.4-0.6 on OPT-175B.")

# Try this:
# 1. Set CTX = 8192: HBM now fits B = 59 / 70 / 67 (BF16 / W4A16 / W8A8), the KV term grows,
#    and every format's gain at full batch shrinks further.
# 2. Keep the LM head (128256 x 4096) in BF16, as many quantized checkpoints do (our assumption):
#    add 128256*4096*(2 - bpp) bytes to each quantized format's weights. At B = 1 the W4A16
#    speedup drops from 3.70x to 3.15x, and W8A8 from 1.97x to 1.85x.
# 3. Change the outliers to +-N(10, 3): the output error without smoothing falls to 0.037 and
#    alpha = 0.5 only gains 1.8x instead of 3.5x. The bigger the outliers, the more smoothing buys.
