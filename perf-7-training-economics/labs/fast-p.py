"""KernelBench's fast_p metric, and the speed-of-light check that catches "too fast to be true".

A calculator (no GPU, no measurements). It does three things:
  1. fast_p on a small ILLUSTRATIVE results table (10 made-up tasks shaped like KernelBench's
     levels), at p = 0, 1, 1.5, 2, with the paper's strict "speedup > p" and with ">=" for contrast
         fast_p = (1/N) * sum_i 1(correct_i AND speedup_i > p)        (KernelBench paper, section 3.3)
  2. speed-of-light (SoL) times on H100 (plus L40S and B200) for a matmul and two attention shapes,
         t_SoL = max(FLOPs / dense peak, bytes / HBM bandwidth)
     with the dense peak and with the 2:4-sparse headline peak (the trap)
  3. flags any result faster than the SoL, recomputes fast_p without it, and prices what a
     slower-than-ceiling decode step costs (course 7's $/M formula; our framing)

Specs (dense, per GPU): H100 SXM 989 TF BF16, 3.35 TB/s; L40S 362 TF, 0.864 TB/s; B200 2,250 TF
(dense FP8 4.5 PF / 2), 8 TB/s. Headline (2:4 sparse) BF16 is 2x. Same numbers as choosing-hardware.html.
KernelBench v0.1 says a BF16 4096x4096 matmul on an H100 "should take at least 0.138 ms".

Run: python3 labs/fast-p.py   (stdlib only, < 1 s)
"""

# ---------------------------------------------------------------- 1. fast_p on a toy table
# (task, level, correct?, failure, t_ref ms, t_new ms)   ALL ILLUSTRATIVE, not KernelBench results.
# t_ref = PyTorch Eager reference time, t_new = the generated kernel's time. speedup = t_ref / t_new.
TASKS = [
    ("matmul 4096^2 BF16",        1, True,  "",               0.240, 0.100),
    ("diag(A) @ B",               1, True,  "",               0.900, 0.100),
    ("3D convolution",            1, False, "compile error",  None,  None),
    ("softsign",                  1, True,  "",               0.170, 0.100),
    ("layernorm",                 1, True,  "",               0.060, 0.100),
    ("conv2d + batchnorm + scale",2, False, "wrong values",   None,  None),
    ("matmul + scale + sigmoid",  2, True,  "",               0.150, 0.100),
    ("conv2d + bias + ReLU",      2, True,  "",               0.110, 0.100),
    ("MiniGPT block",             3, True,  "",               0.085, 0.100),
    ("AlexNet",                   3, False, "runtime error",  None,  None),
]


def speedup(t):
    # rounded so an exact 1.50x stays 1.50 (0.150 / 0.100 is 1.4999999999999998 in floating point)
    return round(t[4] / t[5], 6) if t[2] else None


def fast_p(tasks, p, strict=True):
    hits = 0
    for t in tasks:
        s = speedup(t)
        if t[2] and (s > p if strict else s >= p):
            hits += 1
    return hits / len(tasks), hits


print("1. fast_p on an illustrative table of 10 tasks")
print(f"   {'task':28s} {'lvl':>3s} {'correct':>8s} {'speedup':>8s}")
for t in TASKS:
    s = speedup(t)
    print(f"   {t[0]:28s} {t[1]:>3d} {'yes' if t[2] else 'no':>8s} "
          f"{(f'{s:.2f}x' if s else t[3]):>8s}")
print()
PS = [0, 1, 1.5, 2]
print(f"   {'p':>4s} {'fast_p (speedup > p)':>22s} {'with >= instead':>18s}")
for p in PS:
    f, h = fast_p(TASKS, p)
    g, k = fast_p(TASKS, p, strict=False)
    print(f"   {p:>4g} {f'{h}/10 = {f:.1f}':>22s} {f'{k}/10 = {g:.1f}':>18s}")
print("   fast_0 = correctness rate (every correct kernel has speedup > 0).")
print("   The 1.50x task counts for fast_1.5 only under >=; the paper and README use strict >.")
print()

# ---------------------------------------------------------------- 2. speed of light
GPUS = {  # name: (dense BF16 TFLOP/s, headline sparse BF16 TFLOP/s, HBM TB/s)
    "H100 SXM": (989, 1979, 3.35),
    "L40S":     (362, 733, 0.864),
    "B200":     (2250, 4500, 8.0),
}
BF16 = 2  # bytes


def matmul_shape(m, n, k):
    return 2 * m * n * k, BF16 * (m * k + k * n + m * n)


def attn_prefill(b, s, hq, hkv, d, causal=True):
    # QK^T and PV: 2 matmuls of 2*s*s*d FLOPs per head; causal skips about half
    flops = 4 * b * hq * s * s * d * (0.5 if causal else 1.0)
    byts = BF16 * b * s * d * (2 * hq + 2 * hkv)  # read Q, K, V once, write O once (FlashAttention-style)
    return flops, byts


def attn_decode(b, ctx, hq, hkv, d):
    flops = 4 * b * hq * ctx * d                  # one new query per sequence
    byts = BF16 * b * ctx * 2 * hkv * d           # read the whole K and V cache
    return flops, byts


SHAPES = {
    "matmul 4096x4096x4096":                matmul_shape(4096, 4096, 4096),
    "prefill attn, 1 layer, s=8192 causal": attn_prefill(1, 8192, 32, 8, 128),
    "decode attn, 1 layer, B=64, ctx 2048": attn_decode(64, 2048, 32, 8, 128),
}
# Llama-3.1-8B attention shape: 32 query heads, 8 KV heads, head dim 128 (per layer: 4 KiB of KV per token)


def sol_ms(flops, byts, tf, tbs):
    tc, tm = flops / (tf * 1e12), byts / (tbs * 1e12)
    return max(tc, tm) * 1e3, tc * 1e3, tm * 1e3


print("2. Speed of light, t_SoL = max(FLOPs / peak, bytes / bandwidth)")
for name, (fl, by) in SHAPES.items():
    print(f"   {name}: {fl:.4g} FLOPs, {by / 1e6:.1f} MB, intensity {fl / by:.0f} FLOP/byte")
    for g, (dense, sparse, bw) in GPUS.items():
        d, dc, dm = sol_ms(fl, by, dense, bw)
        s, sc, sm = sol_ms(fl, by, sparse, bw)
        bound = "compute" if dc > dm else "memory"
        print(f"      {g:9s} dense {d:7.4f} ms (compute {dc:.4f}, memory {dm:.4f}; {bound}-bound)"
              f" | sparse headline {s:7.4f} ms")
fl, by = SHAPES["matmul 4096x4096x4096"]
print(f"   Check: 2 * 4096^3 / 989e12 = {2 * 4096**3 / 989e12 * 1e3:.4f} ms "
      f"(v0.1 prints 0.138 ms; 0.1390 rounds to 0.139)")
print()

# ---------------------------------------------------------------- 3. flag too-fast results
print("3. Flag results faster than the speed of light (H100, BF16)")
H100 = GPUS["H100 SXM"]
sol_dense = sol_ms(fl, by, H100[0], H100[2])[0]
sol_sparse = sol_ms(fl, by, H100[1], H100[2])[0]
t_claim = TASKS[0][5]
print(f"   claimed matmul 4096^2 time   {t_claim:.3f} ms  (speedup {speedup(TASKS[0]):.2f}x)")
print(f"   SoL with dense 989 TF        {sol_dense:.4f} ms  -> "
      f"{'IMPOSSIBLE: flag it' if t_claim < sol_dense else 'possible'}"
      f" ({sol_dense / t_claim:.2f}x of peak needed)")
print(f"   SoL with sparse 1,979 TF     {sol_sparse:.4f} ms  -> "
      f"{'IMPOSSIBLE: flag it' if t_claim < sol_sparse else 'passes: the trap lets it through'}")
cleaned = [t if i != 0 else (t[0], t[1], False, "too fast", None, None) for i, t in enumerate(TASKS)]
print(f"   {'p':>4s} {'before':>8s} {'after flagging':>15s}")
for p in PS:
    print(f"   {p:>4g} {fast_p(TASKS, p)[0]:>8.1f} {fast_p(cleaned, p)[0]:>15.1f}")
print()

# ---------------------------------------------------------------- 4. why kernels move $/token
print("4. Kernel quality -> $ per million tokens (Llama-3.1-8B, H100, B = 64, 2k ctx; our model)")
PRICE = 3.99          # $/GPU-hr, Lambda 8x
B = 64
KV_SEQ = 128 * 1024 * 2048          # bytes of KV per sequence at 2k context (128 KiB/token)
W, N, C, BW = 16.06e9, 8.03e9, 989e12, 3.35e12
# course 2's step formula: step = B*KV_seq/BW + max(2*B*N/C, W/BW)   -> 9.92 ms (course 7 _facts)
STEP_CEIL = B * KV_SEQ / BW + max(2 * B * N / C, W / BW)
for frac in (1.0, 0.9, 0.8, 0.7):
    step = STEP_CEIL / frac
    tps = B / step
    usd = PRICE / (tps * 3600) * 1e6
    print(f"   kernels at {frac:.0%} of the bandwidth ceiling: step {step * 1e3:5.2f} ms, "
          f"{tps:6.0f} tok/s, ${usd:.3f}/M")

# Try this:
#   - Change TASKS[0]'s t_new to 0.150 ms: it clears both checks, so the SoL check alone
#     can't tell an honest 1.6x from a subtle cheat. KernelBench v0.1 also checks for torch/cuBLAS calls.
#   - Set causal=False in the prefill shape: FLOPs double and the dense/sparse gap doubles with it.
#   - Change decode B to 1: the decode-attention SoL is memory-bound either way, so the
#     dense-vs-sparse choice doesn't matter there. It only bites on compute-bound shapes.
