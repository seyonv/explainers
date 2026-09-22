"""Speculative sampling lab: Scaling Book Part 7, Appendix D (speculative sampling)
and Appendix A (how real is the batch > 240 rule?).

Run: python3 labs/speculative-sampling.py   (numpy, CPU, a few seconds)

Parts
  (a) Speculative sampling on a toy 8-token vocabulary. The draft proposes K tokens from q,
      each is accepted with probability min(1, p/q), the first rejected one is resampled from
      norm(max(0, p - q)), and a bonus token comes from p if all K pass (Leviathan et al. 2023,
      arXiv 2211.17192, Section 2.3 and Algorithm 1). We check that the emitted tokens follow
      the target p (total-variation distance ~ 0 over 200k chains) and measure the acceptance rate.
  (b) Expected tokens per verify step and wall-time speedup for Llama-3.1-8B + a hypothetical
      1.0B draft on one H100 at batch 1 (our numbers, not the book's).
  (c) Appendix A: recompute the book's ~365 us minimum layer time, print a flat-then-linear
      step-time table, and (our inference) show why speculation fades once B*(K+1) passes B_crit.
"""
import numpy as np

rng = np.random.default_rng(0)

# ------------------------------------------------------------------ (a) toy speculative sampling
print("== (a) Speculative sampling on a toy vocabulary (8 tokens) ==")
p = np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.07, 0.04, 0.02])   # target (big model)
q = np.array([0.40, 0.15, 0.20, 0.05, 0.10, 0.04, 0.04, 0.02])   # draft (small model)
V = len(p)
K = 4          # draft lookahead
L = 16         # tokens generated per chain
N = 200_000    # independent chains

alpha_theory = np.minimum(p, q).sum()
resid = np.maximum(p - q, 0)
resid /= resid.sum()
print(f"alpha = sum_x min(p, q) = {alpha_theory:.3f}   TV(p, q) = {0.5 * np.abs(p - q).sum():.3f}")

# The toy "model" ignores context, so every position has the same p and q.
# That keeps the check exact: every output position must be distributed as p.
out = np.full((N, L + K + 1), -1, dtype=np.int64)
filled = np.zeros(N, dtype=np.int64)
proposed = accepted = steps = 0
while True:
    act = np.nonzero(filled < L)[0]
    if act.size == 0:
        break
    n = act.size
    x = rng.choice(V, size=(n, K), p=q)                        # K draft tokens per chain
    ok = rng.random((n, K)) < np.minimum(1.0, p[x] / q[x])     # accept with min(1, p/q)
    first_bad = np.where(ok.all(1), K, np.argmin(ok, 1))       # accepted prefix length
    proposed += (first_bad + (first_bad < K)).sum()        # drafts the target actually judged
    accepted += first_bad.sum()
    steps += n
    # the extra token: resample from the residual if a draft was rejected, else bonus from p
    extra = np.where(first_bad < K,
                     rng.choice(V, size=n, p=resid),
                     rng.choice(V, size=n, p=p))
    toks = np.concatenate([x, extra[:, None]], 1)
    toks = np.where(np.arange(K + 1)[None, :] < first_bad[:, None], toks,
                    np.where(np.arange(K + 1)[None, :] == first_bad[:, None], extra[:, None], -1))
    for j in range(K + 1):
        m = toks[:, j] >= 0
        out[act[m], filled[act[m]] + j] = toks[m, j]
    filled[act] += first_bad + 1

emitted = out[:, :L]
tvs = []
for pos in range(L):
    freq = np.bincount(emitted[:, pos], minlength=V) / N
    tvs.append(0.5 * np.abs(freq - p).sum())
freq_all = np.bincount(emitted.ravel(), minlength=V) / emitted.size
naive = rng.choice(V, size=N, p=q)                              # "keep the draft, don't verify"
tv_naive = 0.5 * np.abs(np.bincount(naive, minlength=V) / N - p).sum()
noise = 0.5 * np.sqrt(2 * V / (np.pi * N))                      # rough E[TV] of pure sampling noise
print(f"chains: {N:,}  x  {L} tokens   verify steps: {steps:,}")
print("target p        :", " ".join(f"{v:.3f}" for v in p))
print("emitted (pos 1-16):", " ".join(f"{v:.3f}" for v in freq_all))
print(f"TV(emitted, p): max over positions = {max(tvs):.4f}, mean = {np.mean(tvs):.4f}"
      f"   (sampling noise alone ~ {noise:.4f})")
print(f"TV(unverified draft, p) = {tv_naive:.3f}   <- what 'just trust the draft' gets")
acc_rate = accepted / proposed
tok_per_step = filled.sum() / steps
print(f"acceptance rate: measured {acc_rate:.3f}  vs alpha {alpha_theory:.3f}")
E_toy = (1 - alpha_theory ** (K + 1)) / (1 - alpha_theory)
print(f"tokens per verify step: measured {tok_per_step:.3f}"
      f"  vs formula (1-a^(K+1))/(1-a) = {E_toy:.3f}")

# ------------------------------------------------------------------ (b) expected speedup table
print("\n== (b) Llama-3.1-8B target + hypothetical 1.0B draft, one H100, batch 1 ==")
BW, C = 3.35e12, 989e12
P_t, P_d = 8.03e9, 1.0e9                     # params; the draft size is illustrative
t_target = 2 * P_t / BW                      # bf16 weight read per step
t_draft = 2 * P_d / BW
c = t_draft / t_target
print(f"target step = 2*8.03e9 B / 3.35e12 B/s = {t_target * 1e3:.2f} ms")
print(f"draft step  = 2*1.0e9  B / 3.35e12 B/s = {t_draft * 1e3:.3f} ms   c = {c:.3f}")
print(f"verify K+1 = 5 tokens: math 2*8.03e9*5/9.89e14 = {2 * P_t * 5 / C * 1e6:.0f} us  << "
      f"{t_target * 1e3:.2f} ms weight read, so the verify step still costs ~1 step")
print(f"{'alpha':>5} {'K':>3} {'E tokens/step':>14} {'time/step ms':>13} {'speedup':>8} {'tok/s':>7}")
for a in (0.6, 0.7, 0.8):
    for k in (2, 4, 6):
        E = (1 - a ** (k + 1)) / (1 - a)
        t = k * t_draft + t_target
        print(f"{a:5.1f} {k:3d} {E:14.2f} {t * 1e3:13.2f} {E / (k * c + 1):8.2f}x {E / t:7.0f}")
print(f"baseline: 1 token per {t_target * 1e3:.2f} ms = {1 / t_target:.0f} tok/s")

# ------------------------------------------------------------------ (c) Appendix A
print("\n== (c) Appendix A: the batch > 240 rule, measured shape ==")
W5, C5 = 8.2e11, 1.97e14                     # TPU v5e HBM B/s, bf16 FLOP/s
D, F = 8192, 32768
P_exact = 2 * D * F
print(f"layer params = 2 matmuls * 8192 * 32768 = {P_exact / 1e6:.0f}M   (book: 'about 600M')")
for label, P in (("book's 600M", 600e6), ("exact 537M", P_exact)):
    t = 2 * P / 4 / W5
    print(f"  {label:12s}: 2 bytes * {P / 1e6:.0f}M / 4 chips = {2 * P / 4 / 1e6:.0f} MB per chip"
          f" / 8.2e11 B/s = {t * 1e6:.1f} us")
P = 600e6
t_w = 2 * P / 4 / W5
print(f"{'batch':>6} {'weight read us':>15} {'math us':>9} {'step us':>9} {'tokens/us':>10}")
for B in (1, 16, 64, 128, 240, 256, 512, 1024, 2048):
    t_m = 2 * B * P / 4 / C5
    step = max(t_w, t_m)
    print(f"{B:6d} {t_w * 1e6:15.0f} {t_m * 1e6:9.0f} {step * 1e6:9.0f} {B / (step * 1e6):10.3f}")
print(f"crossing: B = C / W = 1.97e14 / 8.2e11 = {C5 / W5:.0f}")

print("\n-- our inference: speculation spends the flat part (Llama-3.1-8B, H100, alpha 0.8, K 4) --")
print("   weights-only roofline: step = max(weight read, 2*P*tokens/C); KV reads ignored")
a, k = 0.8, 4
E = (1 - a ** (k + 1)) / (1 - a)
print(f"{'batch':>6} {'normal ms':>10} {'verify tokens':>14} {'verify ms':>10} {'draft ms':>9} {'speedup':>8}")
for B in (1, 16, 32, 64, 128, 256):
    t_norm = max(2 * P_t / BW, 2 * P_t * B / C)
    tv = max(2 * P_t / BW, 2 * P_t * B * (k + 1) / C)
    td = k * max(2 * P_d / BW, 2 * P_d * B / C)
    print(f"{B:6d} {t_norm * 1e3:10.2f} {B * (k + 1):14d} {tv * 1e3:10.2f} {td * 1e3:9.2f}"
          f" {E * t_norm / (tv + td):7.2f}x")

# Try this:
# 1. Make the draft worse: q = np.full(8, 1/8) (uniform). alpha drops from 0.85 to 0.725 and tokens/step fall,
#    but TV(emitted, p) stays at sampling noise: verification keeps the output exact.
# 2. In (b), set P_d = 3e9 (a 3B draft): c rises to ~0.37 and K = 6 turns into a slowdown at alpha 0.6.
# 3. In (c), extend the batch list past 240 (e.g. 4096) and watch tokens/us stop rising at ~0.66.
