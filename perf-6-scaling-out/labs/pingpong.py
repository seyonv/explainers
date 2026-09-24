"""Ping-pong pipeline simulator for MegaScale-Infer (arXiv 2504.02263, sections 4.1, 4.2, 7.3).

One decode iteration of a disaggregated MoE layer stack. The global batch is split into
m micro-batches. Each micro-batch, for each of L MoE layers, goes
    attention node (T_a) -> A->E network (T_c) -> expert node (T_e) -> E->A network (T_c)
and then on to the next layer. Every one of the four resources handles one micro-batch at
a time, first come first served (the two network directions are separate links).

Prints:
  1. the paper's minimum micro-batch count m >= 2(1 + T_c/T_f)
  2. simulated busy fraction of the attention and expert nodes vs m and T_c/T_f,
     and a check of the simulator against Eq 5 (T_total) and Eq 4 (per-micro-batch bounds)
  3. what breaks when T_a != T_e (constraint 1) or T_c > T_f (constraint 2)
  4. the m = 3 timeline drawn on the card
  5. Mixtral 8x22B numbers: 196,608 B per pair, Eq 6 lower bound, TBT budget, Eq 8 memory
  6. M2N percentages turned into speed-up factors

Time units are "T_f = 1" (illustrative), except in section 5.
stdlib + numpy, runs in well under a second:  python3 labs/pingpong.py
"""
import math
import numpy as np


def simulate(m, L, Ta, Te, Tc, trace=False):
    """Return (T_total, attention busy fraction, expert busy fraction, per-mb finish times, events)."""
    free = {"A": 0.0, "AE": 0.0, "E": 0.0, "EA": 0.0}
    dur = {"A": Ta, "AE": Tc, "E": Te, "EA": Tc}
    ready = np.zeros(m)          # when each micro-batch is ready for its next stage
    events = []
    for layer in range(L):
        for res in ("A", "AE", "E", "EA"):
            for j in range(m):   # FIFO keeps micro-batches in order at every resource
                start = max(ready[j], free[res])
                end = start + dur[res]
                free[res] = end
                ready[j] = end
                if trace:
                    events.append((res, layer, j, start, end))
    total = ready.max()
    return total, m * L * Ta / total, m * L * Te / total, ready.copy(), events


def eq5(m, L, Ta, Te, Tc):
    Tf = max(Ta, Te)
    return (Ta + Te + 2 * Tc) + Tf * (m * L - 1)


print("=" * 72)
print("1. Minimum micro-batches, constraint 3:  m*T_f >= 2*(T_f + T_c)  =>  m >= 2(1 + T_c/T_f)")
print("=" * 72)
print(f"{'T_c/T_f':>8} {'2(1+r)':>8} {'smallest integer m':>20}")
for r in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.95):
    bound = 2 * (1 + r)
    print(f"{r:>8.2f} {bound:>8.2f} {math.ceil(bound - 1e-12):>20d}")
print("-> T_c < T_f/2 needs m = 3; slower comm (up to T_c < T_f) needs m = 4. N_m = 4 in Alg 1.")

print()
print("=" * 72)
print("2. Simulated busy fraction, balanced T_a = T_e = 1, L = 56 layers (Mixtral 8x22B)")
print("=" * 72)
L = 56
rs = (0.2, 0.4, 0.6, 0.8)
print(f"{'m':>3} " + " ".join(f"{'Tc/Tf='+str(r):>12}" for r in rs) + "   steady-state min(1, m/(2(1+r)))")
util = {}
for m in range(1, 6):
    row = []
    for r in rs:
        tot, ua, ue, _, _ = simulate(m, L, 1.0, 1.0, r)
        util[(m, r)] = ua
        row.append(f"{ua*100:>11.1f}%")
    ss = " ".join(f"{min(1, m/(2*(1+r)))*100:5.1f}%" for r in rs)
    print(f"{m:>3} " + " ".join(row) + "   " + ss)
print("(attention and expert nodes are identical here because T_a = T_e)")
r = 0.2
print(f"throughput ratios (fixed micro-batch size, so throughput ~ busy fraction) at T_c/T_f = {r}: m 1->2 = {util[(2, r)]/util[(1, r)]:.2f}x, "
      f"m 2->3 = {util[(3, r)]/util[(2, r)]:.2f}x, m 3->4 = {util[(4, r)]/util[(3, r)]:.2f}x")
r = 0.4
print(f"throughput ratios (fixed micro-batch size, so throughput ~ busy fraction) at T_c/T_f = {r}: m 1->2 = {util[(2, r)]/util[(1, r)]:.2f}x, "
      f"m 2->3 = {util[(3, r)]/util[(2, r)]:.2f}x, m 3->4 = {util[(4, r)]/util[(3, r)]:.2f}x")

print("\nCheck against Eq 5, T_total = (T_a+T_e+2T_c) + T_f(mL-1), when constraint 3 holds:")
for m, r in ((3, 0.2), (3, 0.4), (4, 0.6), (4, 0.8)):
    tot, *_ = simulate(m, L, 1.0, 1.0, r)
    print(f"  m={m} Tc/Tf={r}: simulated {tot:.2f}  Eq5 {eq5(m, L, 1, 1, r):.2f}")
m, r = 2, 0.4
tot, *_ = simulate(m, L, 1.0, 1.0, r)
print(f"  m={m} Tc/Tf={r} (constraint 3 violated): simulated {tot:.2f}  Eq5 {eq5(m, L, 1, 1, r):.2f}"
      "  <- Eq 5 assumes no bubbles")

print("\nEq 4 bounds on one micro-batch's iteration latency, m=3, Tc/Tf=0.4, L=56:")
tot, _, _, fin, _ = simulate(3, L, 1.0, 1.0, 0.4)
lo, hi = (1 + 1 + 0.8) + 3 * 1 * (L - 1), 3 * 1 * L
print(f"  lower {lo:.1f} <= each micro-batch (finish - its first start): "
      f"{[round(float(f) - j, 1) for j, f in enumerate(fin)]} <= upper {hi:.1f}")

print()
print("=" * 72)
print("3. Breaking constraints 1 and 2 (m = 3, L = 56)")
print("=" * 72)
for Ta, Te, Tc, label in ((1.0, 1.0, 0.4, "balanced, fast comm"),
                          (1.0, 0.5, 0.2, "experts too fast (T_e = T_a/2)"),
                          (0.5, 1.0, 0.2, "attention too fast (T_a = T_e/2)"),
                          (1.0, 1.0, 1.3, "T_c > T_f (constraint 2 broken)")):
    tot, ua, ue, _, _ = simulate(3, L, Ta, Te, Tc)
    print(f"  {label:<36} attention busy {ua*100:5.1f}%  experts busy {ue*100:5.1f}%")
tot, ua, ue, _, _ = simulate(8, L, 1.0, 1.0, 1.3)
print(f"  {'T_c > T_f, even with m = 8':<36} attention busy {ua*100:5.1f}%  experts busy {ue*100:5.1f}%"
      "  <- the link, not compute, is the bottleneck")

print()
print("=" * 72)
print("4. Timeline for the card: m = 3, T_a = T_e = 1, T_c = 0.4, L = 2")
print("=" * 72)
tot, ua, ue, _, ev = simulate(3, 2, 1.0, 1.0, 0.4, trace=True)
for res, layer, j, s, e in ev:
    print(f"  {res:>2} layer {layer+1} mb{j+1}: {s:4.1f} -> {e:4.1f}")
print(f"  T_total = {tot:.1f} (Eq 5: {eq5(3, 2, 1, 1, 0.4):.1f}); busy A {ua*100:.1f}%, E {ue*100:.1f}%")
tot1, ua1, _, _, _ = simulate(1, 2, 1.0, 1.0, 0.4)
print(f"  same 2 layers with m = 1: T_total = {tot1:.1f}, busy {ua1*100:.1f}%"
      "  (short L = 2 exaggerates fill and drain; see section 2 for L = 56)")

print()
print("=" * 72)
print("5. Mixtral 8x22B numbers (paper Table 4 + HF config: 48 query heads, 8 KV heads -> g = 6)")
print("=" * 72)
h, Lm, E, K, g = 6144, 56, 8, 2, 6
b_a, tp_a = 128, 2
pair = b_a * K / E * h * 2 / tp_a
print(f"  bytes per attention GPU -> expert GPU = 128 x 2/8 x 6144 x 2 B / TP2 = {pair:,.0f} B  (paper: 196,608)")
send = b_a * h * K / tp_a * 2          # Eq 6 numerator in elements, x2 bytes for bf16 (our reading)
W = 200e9 / 8                          # one 200 Gb/s NIC per GPU (paper section 7.3) = 25 GB/s
print(f"  one attention GPU sends b_a*h*K/tp_a = {send/2:,.0f} values = {send:,.0f} B per micro-batch per layer")
print(f"  Eq 6 attention side with Util = 1: {send:,.0f} B / 25 GB/s = {send/W*1e6:.1f} us  (a lower bound)")
print(f"  one 196,608 B message alone at 25 GB/s = {pair/W*1e6:.2f} us; 256 KB = {262144/W*1e6:.2f} us")
for m in (3, 4):
    print(f"  TBT 150 ms, m = {m}, Eq 4 upper bound m*T_f*L <= SLO -> T_f <= {150e3/(m*Lm):.0f} us per layer")
print(f"  -> if T_f sat at that 893 us ceiling, T_c/T_f >= {send/W/ (150e-3/(3*Lm)):.3f}: deep in the 'fast comm' (m = 3) regime")
print(f"  Step-3 (50 ms TPOT, 3 stages, 61 layers): 50/3 = {50/3:.2f} ms per stage; paper uses 16.6 ms"
      f" -> 16.6 ms / 61 = {16.6e3/61:.0f} us per layer")
s_len = 571 + 159                      # median input + output (illustrative sequence length)
kv_tok = 4 * h * Lm / g                # Eq 8: 4*m*b_a*s*h*L/g bytes, i.e. K and V, 2 B each, h/g wide
P_a = Lm * (h * h + 2 * h * (h // g) + h * h)   # Q, K, V, O projections only (our count)
kv = 3 * b_a * s_len * kv_tok
print(f"  Eq 8 KV per token = 4*h*L/g = {kv_tok:,.0f} B = {kv_tok/1024:.0f} KiB")
print(f"  m=3, b_a=128, s=571+159={s_len}: KV = {kv/1e9:.1f} GB; attention weights 2*P_a = "
      f"{2*P_a/1e9:.2f} GB (P_a = {P_a/1e9:.2f}B, QKVO only); total {(kv+2*P_a)/1e9:.1f} GB "
      f"< tp_a*C_a = 2 x 80 = 160 GB")
b_max = (2 * 80e9 - 2 * P_a) / (3 * s_len * kv_tok)
print(f"  largest b_a that fits at tp_a = 2, m = 3: {b_max:.0f}")

print()
print("=" * 72)
print("6. M2N vs NCCL (section 7.3): 'x% lower latency' as a speed-up factor = 1/(1 - x)")
print("=" * 72)
for label, pct in (("256 KB median", 68.2), ("256 KB P99", 92.9), ("best median", 80.8),
                   ("best P99", 96.2), ("M,N sweep P99 low", 54.7), ("M,N sweep P99 high", 96.9)):
    print(f"  {label:<20} {pct:5.1f}% lower -> {1/(1-pct/100):5.1f}x faster")

# Try this:
# 1. Set rs = (0.45, 0.5, 0.55) and watch m = 3 go from 100% busy to just under it at T_c/T_f = 0.5.
# 2. In section 3, sweep Te from 0.3 to 1.5 at Ta = 1 (like the paper's DBRX DP sweep):
#    whichever side is faster idles, and throughput per GPU peaks at Te = Ta.
# 3. Raise L to 200: fill and drain matter less, and the simulated busy fraction approaches
#    the steady-state column.
