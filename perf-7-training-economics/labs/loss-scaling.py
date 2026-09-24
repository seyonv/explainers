"""Mixed precision and activation checkpointing, on a CPU with numpy.

Card: precision-recompute.html ("Mixed precision, FP8 and activation checkpointing").
Runs in a few seconds:  python3 labs/loss-scaling.py

Part 1  FP16 underflow: what fraction of small gradients turn into zero, with and
        without loss scaling, and where overflow starts.
Part 2  Dynamic loss scaling: halve on overflow (skip the step), grow after clean steps.
Part 3  BF16 (emulated by rounding float32 to its top 16 bits): same gradients, no scaling.
Part 4  Why FP32 master weights survive BF16: a small update added to a weight of 1.0.
Part 5  Activation memory and FLOPs for Llama-3.1-8B at seq 8,192 (Korthikanti's formula,
        applied as-is) with no / selective / full checkpointing, plus Chen's sqrt(n) segments.

The gradient distribution in parts 1-3 is ILLUSTRATIVE: log2|g| ~ Normal(-17.5, 4),
chosen so that about 5% of values sit below 2^-24, the share Micikevicius et al. (2018)
report for weight gradients in their Mandarin speech model (Fig 2b).
"""
import numpy as np

rng = np.random.default_rng(0)

FP16_MAX = float(np.finfo(np.float16).max)            # 65,504
FP16_MIN_SUB = float(np.finfo(np.float16).smallest_subnormal)   # 2^-24
FP16_MIN_NORM = float(np.finfo(np.float16).tiny)       # 2^-14


def to_bf16(x):
    """Round float32 to bfloat16 (keep the top 16 bits, round-to-nearest-even), return float32."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounding = ((b >> 16) & 1) + 0x7FFF
    return ((b + rounding) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def fp16_stats(g, scale):
    """Cast g*scale to FP16 and report zeros (underflow), subnormals and inf (overflow)."""
    with np.errstate(over="ignore"):
        h = (g * scale).astype(np.float16)
    a = np.abs(h.astype(np.float32))
    nz = g != 0
    zero = np.mean((a == 0) & nz)
    sub = np.mean((a > 0) & (a < FP16_MIN_NORM))
    inf = np.mean(~np.isfinite(a))
    return zero, sub, inf


print("=" * 78)
print("Part 1. FP16 underflow of gradients (illustrative distribution)")
print("=" * 78)
N = 1_000_000
log2g = rng.normal(-17.5, 4.0, N)
g = (np.sign(rng.standard_normal(N)) * 2.0 ** log2g).astype(np.float32)
a = np.abs(g)
print(f"FP16 max {FP16_MAX:,.0f} · min normal 2^-14 = {FP16_MIN_NORM:.3g} · "
      f"min subnormal 2^-24 = {FP16_MIN_SUB:.3g}")
print(f"gradients: {N:,} values, median |g| = 2^{np.median(log2g):.1f}, "
      f"max |g| = {a.max():.2f} (2^{np.log2(a.max()):.1f})")
print(f"share below 2^-24 (FP32 view):            {np.mean(a < 2**-24):6.2%}")
print(f"share below 2^-14 (subnormal in FP16):    {np.mean(a < 2**-14):6.2%}")
print()
print(f"{'loss scale':>10} | {'-> zero':>8} | {'subnormal':>9} | {'-> inf':>7} | max|g|*scale")
for s in [1, 8, 128, 1024, 8192, 16384, 32768, 65536]:
    z, sb, inf = fp16_stats(g, s)
    print(f"{s:>10,} | {z:8.2%} | {sb:9.2%} | {inf:7.4%} | {a.max() * s:,.0f}")
best = FP16_MAX / a.max()
print(f"largest safe scale = 65,504 / max|g| = {best:,.0f}  -> largest power of 2 = "
      f"{2 ** int(np.floor(np.log2(best))):,}")
print("unscale before the optimizer: g_fp32 = fp16(g * S) / S  (exact for powers of 2)")
h = (g * 8192).astype(np.float16).astype(np.float32) / 8192
kept = (h != 0)
rel = np.abs(h[kept] - g[kept]) / a[kept]
print(f"after S = 8,192 and unscale: median relative error {np.median(rel):.2e}, "
      f"99th pct {np.percentile(rel, 99):.2e}")

print()
print("=" * 78)
print("Part 2. Dynamic loss scaling (illustrative policy and gradient drift)")
print("=" * 78)
S, clean, skipped, growth_interval = 2.0 ** 16, 0, 0, 200
hist = []
for step in range(2000):
    # gradients shrink as training settles, with occasional spikes (illustrative)
    gmax = 4.0 * (0.5 ** (step / 500)) * (8.0 if rng.random() < 0.01 else 1.0)
    if gmax * S > FP16_MAX:          # overflow detected when unscaling -> skip update
        S /= 2
        skipped += 1
        clean = 0
    else:
        clean += 1
        if clean == growth_interval:
            S *= 2
            clean = 0
    hist.append(S)
print(f"start S = 65,536; halve on overflow; double after {growth_interval} clean steps")
print(f"steps 2,000 · skipped {skipped} ({skipped / 2000:.1%}) · final S = {S:,.0f}")
print("S at steps 0/500/1000/1500/1999:", ", ".join(f"{hist[i]:,.0f}" for i in (0, 500, 1000, 1500, 1999)))

print()
print("=" * 78)
print("Part 3. BF16: FP32's exponent range, fewer mantissa bits")
print("=" * 78)
b = to_bf16(g)
print(f"BF16 zeros: {np.mean((b == 0) & (g != 0)):.4%}   (min normal 2^-126, so nothing underflows)")
relb = np.abs(b - g) / a
print(f"BF16 relative error: median {np.median(relb):.2e}, max {relb.max():.2e} "
      f"(<= 2^-8 = {2**-8:.2e})")
h16 = g.astype(np.float16).astype(np.float32)
norm = a >= FP16_MIN_NORM
rel16 = np.abs(h16[norm] - g[norm]) / a[norm]
print(f"FP16 relative error on its normal range: max {rel16.max():.2e} (<= 2^-11 = {2**-11:.2e})")
print("-> BF16 keeps every gradient (no loss scaling) but rounds each ~8x more coarsely than FP16.")

print()
print("=" * 78)
print("Part 4. Why FP32 master weights: add a 1e-4 update to w = 1.0, 1,000 times")
print("=" * 78)
upd = 1e-4
w32 = np.float32(1.0)
w16 = np.float16(1.0)
wb = to_bf16(np.float32(1.0))
for _ in range(1000):
    w32 = np.float32(w32 + np.float32(upd))
    w16 = np.float16(w16 + np.float16(upd))
    wb = to_bf16(np.float32(wb) + np.float32(upd))
print(f"FP32 master: {float(w32):.4f}   FP16: {float(w16):.4f}   BF16: {float(np.asarray(wb).item()):.4f}")
print(f"spacing just above 1.0: FP16 2^-10 = {2**-10:.2e}, BF16 2^-7 = {2**-7:.2e}; "
      f"update/weight = 1/{1 / upd:,.0f}")
print("An update smaller than half the spacing rounds away. Micikevicius: FP16 loses it once")
print("weight/update > 2048; for BF16 (7 mantissa bits) that ratio is only 256.")

print()
print("=" * 78)
print("Part 5. Activation checkpointing: Llama-3.1-8B, seq 8,192, micro-batch 1")
print("=" * 78)
L, h, a_heads, s, bsz = 32, 4096, 32, 8192, 1
N_params = 8.03e9
GiB = 2 ** 30
sbh = s * bsz * h
five = 5 * a_heads * s / h
none = sbh * L * (34 + five)
sel = sbh * L * 34
full = sbh * L * 2
one_layer = sbh * (34 + five)
print(f"Korthikanti Eq 1 per layer: sbh(34 + 5as/h), sbh = {sbh:,}, 5as/h = {five:.0f}")
print(f"  (GPT-3 in the paper: 5*96*2048/12288 = {5 * 96 * 2048 / 12288:.0f})")
fwd_dense = 2 * N_params * s * bsz
attn_fwd = 4 * s * s * h * L * bsz            # QK^T + attention-over-V, forward (Appendix A)
model = 3 * (fwd_dense + attn_fwd)            # = 6*N*s + 12*L*h*s^2
extra_sel = attn_fwd                          # recompute QK^T and AV in the backward pass
extra_full = fwd_dense + attn_fwd             # one more forward pass
rows = [("none", none, 0.0), ("selective", sel, extra_sel), ("full", full, extra_full)]
print(f"model FLOPs per sequence = 6*N*s + 12*L*h*s^2 = {3 * fwd_dense:.3e} + {3 * attn_fwd:.3e} "
      f"= {model:.3e}")
print(f"{'strategy':>10} | {'activations':>11} | {'GB':>7} | {'vs none':>7} | {'extra FLOPs':>11} | {'overhead':>8}")
for name, mem, ex in rows:
    print(f"{name:>10} | {mem / GiB:8.2f} GiB | {mem / 1e9:7.1f} | {mem / none:7.1%} | {ex:11.3e} | {ex / model:8.1%}")
print(f"full recompute also rebuilds one whole layer during backward: + {one_layer / GiB:.2f} GiB peak "
      f"({(sbh * 34) / GiB:.2f} GiB if that layer uses selective/FlashAttention)")
print(f"selective saves {1 - sel / none:.1%} here (paper: 70% for GPT-3, where 5as/h = 80, not {five:.0f})")
print(f"Korthikanti's own overhead convention, Eq 9: s/6h = {s / (6 * h):.1%} here, "
      f"{2048 / (6 * 12288):.2%} for GPT-3 (the paper's 2.7%)")
print(f"by Appendix A's term count (4Bs^2h per layer), GPT-3's overhead would be "
      f"{(4 * 2048**2 * 12288) / (3 * (24 * 2048 * 12288**2 + 4 * 2048**2 * 12288)):.2%}")

tp = 8
print(f"\nWith TP{tp} + sequence parallel (Llama 3's layout), selective: sbh*34/t*L = {sel / tp / GiB:.2f} GiB per GPU")
state16 = 16 * N_params / 1e9
print(f"Training state 16 B/param = {state16:.1f} GB; ZeRO-3 over 8 GPUs = {state16 / 8:.1f} GB per GPU")
for name, mem, _ in rows:
    tot = state16 / 8 + mem / 1e9
    print(f"  8 GPUs, ZeRO-3, no TP, {name:>9}: {state16 / 8:.1f} + {mem / 1e9:.1f} = {tot:6.1f} GB "
          f"{'fits 80 GB' if tot < 80 else 'does NOT fit 80 GB'}")

print("\nWhole Llama-3.1-8B run (15T tokens), 6ND vs 8ND (Scaling Book: full remat):")
D = 15e12
for k in (6, 8):
    fl = k * N_params * D
    hrs = fl / (989e12 * 0.40) / 3600
    print(f"  {k}ND = {fl:.2e} FLOPs executed -> {hrs / 1e6:.2f}M H100-hours if the GPUs run at 40% of peak; "
          f"MFU (counts 6ND only) = {0.40 * 6 / k:.0%}")

print("\nPartial full-recompute curve (recompute k of 32 layers fully, keep the rest):")
print(f"{'k':>3} | {'activations GiB':>15} | {'extra FLOPs':>11}")
for k in [0, 4, 8, 16, 24, 28, 32]:
    mem = (L - k) * one_layer + k * 2 * sbh
    ex = k / L * extra_full
    print(f"{k:>3} | {mem / GiB:15.1f} | {ex / model:11.1%}")
k29 = (L - 29) * one_layer + 29 * 2 * sbh
print(f"k = 29 gets to {k29 / GiB:.1f} GiB at {29 / L * extra_full / model:.1%}; "
      f"selective (every layer) gets {sel / GiB:.1f} GiB at {extra_sel / model:.1%}")

print("\nChen et al. sqrt(n): n equal layers, checkpoint every g-th, hold one segment while recomputing")
n = 32
print(f"{'g':>3} | {'memory (layer units) = n/g + g':>30}")
for gsz in [1, 2, 4, 6, 8, 16, 32]:
    print(f"{gsz:>3} | {n / gsz + gsz:30.1f}")
print(f"min at g = sqrt({n}) = {n ** 0.5:.1f}; compute cost: one extra forward regardless of g")
print("For a transformer, a layer's input (2sbh) is ~177x smaller than its activations, so")
print("checkpointing every layer (g = 1) already gives near-minimal memory: that is 'full'.")

# Try this:
# 1. Change the gradient distribution to rng.normal(-12, 3, N): how small can the loss scale be?
# 2. Replace to_bf16 with plain truncation (b & 0xFFFF0000) and compare Part 3/4 errors.
# 3. Set s = 2048 and h = 12288, a_heads = 96 in Part 5 to reproduce the paper's GPT-3 70%.
