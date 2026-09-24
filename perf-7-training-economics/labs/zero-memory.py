"""Data parallelism, ZeRO and FSDP (course 7, card 3).

Run: python3 labs/zero-memory.py   (numpy only, CPU, under a second)

Nothing here is measured. It is a calculator plus a tiny simulator:
  (1) ZeRO Table 1 (Rajbhandari et al. 2020): model-state memory per GPU for the 7.5B model,
      recomputed from the formulas and compared with the paper's printed values.
  (2) Llama-3.1-8B on 8 H100s: per-GPU memory for DP / ZeRO-1 / ZeRO-2 / ZeRO-3
      (16 bytes/param, ZeRO's convention; and 20 bytes with FP32 gradient accumulation).
  (3) Communication per training step: bytes each GPU sends and the time at
      NVLink (450 GB/s per direction) and InfiniBand (50 GB/s per GPU), ring algorithm.
  (4) When DP / FSDP stays compute-bound: Scaling Book's B/X > C/W for TPU v5p and H100.
  (5) Global batch = micro-batch x gradient accumulation x DP.
  (6) A numpy check that DP and ZeRO-1/2/3 produce identical weights, with elements
      communicated and optimizer floats stored per rank.
  (7) Why FSDP is terrible for decode serving: gather the weights for every token.
"""
import numpy as np

GB = 1e9
PSI_8B = 8.03e9                 # Llama-3.1-8B parameters (course facts)
H100_C = 989e12                 # dense BF16 FLOP/s (course facts)
H100_HBM = 3.35e12              # B/s
NVLINK = 450e9                  # B/s per GPU, one direction (course 6, collectives card)
IB = 50e9                       # B/s per GPU: one 400 Gb/s NIC (course 6, collectives card)
K = 12                          # mixed-precision Adam: fp32 master 4 + m 4 + v 4 (ZeRO section 3.1)


def zero_mem(psi, nd, stage, k=K, fp32_grad_acc=False):
    """Model-state bytes per GPU. Stage 0 = plain DP. Forms from ZeRO Figure 1 / Playbook."""
    g = 2 + (4 if fp32_grad_acc else 0)            # gradient bytes per param
    if stage == 0:
        return (2 + g + k) * psi
    if stage == 1:
        return 2 * psi + g * psi + k * psi / nd
    if stage == 2:
        return 2 * psi + (g + k) * psi / nd
    return (2 + g + k) * psi / nd


# ------------------------------------------------------------------ (1) ZeRO Table 1
print("(1) ZeRO Table 1, 7.5B model, K = 12 (GB per device, model states only)")
paper = {1: (120, 120, 120), 4: (52.5, 41.3, 30), 16: (35.6, 21.6, 7.5),
         64: (31.4, 16.6, 1.88), 256: (30.4, 15.4, 0.47), 1024: (30.1, 15.1, 0.12)}
print(f"    {'N_d':>5} | {'Pos':>7} {'Pos+g':>7} {'Pos+g+p':>8} | paper")
for nd, p in paper.items():
    ours = [zero_mem(7.5e9, nd, s) / GB for s in (1, 2, 3)]
    print(f"    {nd:>5} | {ours[0]:7.2f} {ours[1]:7.2f} {ours[2]:8.3f} | {p[0]} / {p[1]} / {p[2]}")
print(f"    plain DP, any N_d: 16 x 7.5e9 = {zero_mem(7.5e9, 64, 0) / GB:.0f} GB")
print(f"    Figure 1 example (N_d = 64): 120 -> {zero_mem(7.5e9, 64, 1) / GB:.1f} -> "
      f"{zero_mem(7.5e9, 64, 2) / GB:.1f} -> {zero_mem(7.5e9, 64, 3) / GB:.2f} GB")

# ------------------------------------------------------------------ (2) Llama-3.1-8B on 8 H100s
print("\n(2) Llama-3.1-8B (8.03B params) on N_d = 8 H100s, model states per GPU (no activations)")
names = ["DP (replica)", "ZeRO-1", "ZeRO-2", "ZeRO-3 / FSDP"]
for fp32 in (False, True):
    tag = "20 B/param (FP32 grad accumulation, Playbook)" if fp32 else "16 B/param (ZeRO convention)"
    print(f"    {tag}")
    for s, name in enumerate(names):
        m = zero_mem(PSI_8B, 8, s, fp32_grad_acc=fp32) / GB
        print(f"      {name:14s} {m:7.1f} GB   {'fits' if m < 80 else 'DOES NOT FIT'} in 80 GB, "
              f"{80 - m:5.1f} GB left for activations")
print("    16 B/param terms: params 2Y = %.2f GB, grads 2Y = %.2f GB, optimizer 12Y = %.2f GB"
      % (2 * PSI_8B / GB, 2 * PSI_8B / GB, 12 * PSI_8B / GB))
print("    ZeRO-3 as the DP degree grows (16 B/param):",
      ", ".join(f"N_d={nd}: {zero_mem(PSI_8B, nd, 3) / GB:.2f} GB" for nd in (1, 2, 4, 8, 16, 64)))

# ------------------------------------------------------------------ (3) communication per step
print("\n(3) Communication per step, Llama-3.1-8B, BF16 grads and params (2 bytes), ring algorithm")
n = 8
S = 2 * PSI_8B                               # bytes of one full BF16 copy = 16.06 GB
f = (n - 1) / n
plan = {"DP": (["all-reduce grads (= reduce-scatter + all-gather)"], 2),
        "ZeRO-1": (["reduce-scatter grads", "all-gather updated params"], 2),
        "ZeRO-2": (["reduce-scatter grads (on the fly)", "all-gather updated params"], 2),
        "ZeRO-3": (["all-gather params (forward)", "all-gather params (backward)",
                    "reduce-scatter grads"], 3)}
for name, (ops, psi_units) in plan.items():
    sent = psi_units * f * S
    print(f"    {name:7s} {psi_units}Y elements -> {psi_units} x 7/8 x {S / GB:.2f} GB = {sent / GB:5.1f} GB sent per GPU"
          f" | NVLink {sent / NVLINK * 1e3:6.1f} ms | IB {sent / IB * 1e3:6.0f} ms   ({'; '.join(ops)})")
seq = 4096
for toks in (seq, 2 * seq):
    t_fb = 6 * PSI_8B * toks / H100_C
    t_b = 4 * PSI_8B * toks / H100_C
    print(f"    compute for {toks} tokens per GPU: 6*N*tokens / 989 TF = {t_fb * 1e3:.0f} ms at 100% "
          f"({t_fb / 0.4 * 1e3:.0f} ms at 40% MFU); backward alone {t_b * 1e3:.0f} ms at 100%")

# ------------------------------------------------------------------ (4) compute-bound condition
print("\n(4) DP / FSDP stay compute-bound when tokens per chip B/X > C / W  (Scaling Book Part 5)")
rows = [("TPU v5p, 1 ICI axis (book)", 4.59e14, 1.8e11),
        ("H100 in a node, NVLink (book: 2200)", 990e12, 450e9),
        ("H100 across nodes, 400 GB/s node egress (book: 2475)", 990e12, 400e9),
        ("ours: H100 NVLink, 989 TF", H100_C, NVLINK),
        ("ours: H100 flat ring over IB, 50 GB/s per GPU", H100_C, IB)]
for label, C, W in rows:
    print(f"    {label:55s} C/W = {C:.3g} / {W:.3g} = {C / W:8,.0f} tokens per chip")
print(f"    with the ring's (n-1)/n for n = 8 over NVLink: {H100_C / NVLINK * 7 / 8:,.0f} tokens per GPU")
b_star = H100_C / NVLINK * f
print(f"    check on the 8B: backward at b* = {b_star:,.0f} tokens -> {4 * PSI_8B * b_star / H100_C * 1e3:.1f} ms;"
      f" DP all-reduce -> {2 * f * S / NVLINK * 1e3:.1f} ms (equal, so b* is the break-even)")

# ------------------------------------------------------------------ (5) global batch
print("\n(5) Global batch = micro-batch x grad accumulation x DP  (Playbook)")
gbs_tokens = 4 * 1024 * 1024
gbs = gbs_tokens // seq
print(f"    Playbook: 4M tokens / 4k sequence = {gbs} sequences")
for dp in (128, 512, 8):
    print(f"      mbs 2, DP {dp:3d}: grad_acc = {gbs} / (2 x {dp}) = {gbs // (2 * dp)}")

# ------------------------------------------------------------------ (6) numpy: DP == ZeRO
print("\n(6) Numpy check: 4 ranks, linear model, Adam, 3 steps. DP vs ZeRO-1/2/3 end with identical weights")
rng = np.random.default_rng(0)
R, P, M = 4, 16, 8                           # ranks, params, samples per rank
X = rng.standard_normal((R, M, P))
w_true = rng.standard_normal(P)
Y = X @ w_true
w0 = rng.standard_normal(P)
lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8


def grad(w, r):
    return 2 * X[r].T @ (X[r] @ w - Y[r]) / M


def adam(w, g, m, v, t):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    return w - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps), m, v


def dp(steps=3):
    ws = [w0.copy() for _ in range(R)]; ms = [np.zeros(P)] * R; vs = [np.zeros(P)] * R
    sent = 0
    for t in range(1, steps + 1):
        g = np.mean([grad(ws[r], r) for r in range(R)], axis=0)     # all-reduce
        sent += 2 * (R - 1) / R * P
        for r in range(R):
            ws[r], ms[r], vs[r] = adam(ws[r], g, ms[r], vs[r], t)
    return ws, sent / steps, 3 * P                                  # master, m, v per rank


def zero(stage, steps=3):
    sh = P // R
    shards = [w0[r * sh:(r + 1) * sh].copy() for r in range(R)]     # fp32 master shard per rank
    ms = [np.zeros(sh) for _ in range(R)]; vs = [np.zeros(sh) for _ in range(R)]
    full = w0.copy()                                                # replicated copy (stages 1, 2)
    sent = 0
    for t in range(1, steps + 1):
        if stage == 3:                                              # gather before forward and backward
            full = np.concatenate(shards); sent += 2 * (R - 1) / R * P
        gs = [grad(full, r) for r in range(R)]
        g_mean = np.mean(gs, axis=0)                                # reduce-scatter: rank r keeps slice r
        sent += (R - 1) / R * P
        for r in range(R):
            shards[r], ms[r], vs[r] = adam(shards[r], g_mean[r * sh:(r + 1) * sh], ms[r], vs[r], t)
        if stage in (1, 2):                                         # all-gather updated params
            full = np.concatenate(shards); sent += (R - 1) / R * P
    return [np.concatenate(shards)] * R, sent / steps, 3 * sh


# floats each rank keeps between steps: (bf16 params, bf16 grads, fp32 master + m + v)
resident = {0: (P, P, 3 * P), 1: (P, P, 3 * P // R), 2: (P, P // R, 3 * P // R), 3: (P // R, P // R, 3 * P // R)}
ref, s_dp, o_dp = dp()
print(f"    P = {P} params. Resident floats per rank are (params, grads, optimizer).")
print(f"    DP     : {s_dp:5.1f} elements sent per rank per step (2 x P x 3/4 = {2 * 0.75 * P:.0f}), resident {resident[0]}")
for st in (1, 2, 3):
    ws, s, o = zero(st)
    diff = max(np.abs(ws[r] - ref[r]).max() for r in range(R))
    print(f"    ZeRO-{st} : {s:5.1f} elements sent per rank per step, resident {resident[st]}, "
          f"max |w - w_DP| = {diff:.1e}")

# ------------------------------------------------------------------ (7) FSDP for decode
print("\n(7) FSDP for decode: every token must gather the weights it doesn't hold")
for name, wbytes in (("Llama-3.1-8B", 2 * PSI_8B), ("Llama-3.1-70B", 141.1e9)):
    gather = f * wbytes / NVLINK
    hbm = wbytes / H100_HBM
    print(f"    {name:14s} all-gather 7/8 x {wbytes / GB:.1f} GB over NVLink = {gather * 1e3:5.1f} ms per token"
          f"   vs reading all of it from one HBM {hbm * 1e3:.2f} ms, a TP8 shard {hbm / 8 * 1e3:.2f} ms")

# Try this:
# 1. zero_mem(PSI_8B, 8, s, k=4) for s in 0..3: plain SGD with momentum-ish K = 4 shrinks the gap between stages.
# 2. Set n = 64 in (3): bytes per GPU go from 1.75x to 1.97x of 16 GB, so DP time barely grows with more GPUs.
# 3. In (4), use W = 300e9 (H800 NVLink): C/W = 3,297 tokens per GPU, the Scaling Book's DeepSeek-V3 number.
