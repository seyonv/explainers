"""Worked problems lab: Scaling Book Part 7, "Worked Problems" Q1-Q7.

Run: python3 labs/worked-problems.py   (stdlib only, CPU, well under a second)

Prints the book's answers for its invented model (Q1, Q2, Q2 with K=1, Q3, Q5,
Q7 crossover), our own worked solutions for Q4 and Q6 (the book gives none),
and the same questions redone for Llama-3.1-8B on one H100 (our recomputation).
"""
import math

# ---- The book's invented model (Part 7, Worked Problems) ----
L, D, F, N, K, H, V = 64, 4096, 16384, 32, 8, 256, 32128

# ---- Hardware ----
V5E = dict(bf16=1.97e14, int8=3.94e14, hbm_bw=8.2e11, hbm=16e9, ici=4.5e10)  # book Parts 1, 7
H100 = dict(bf16=9.89e14, fp8=1.979e15, hbm_bw=3.35e12, hbm=80e9)             # book Part 12
CHIPS = 16          # TPU v5e 4x4
CTX = 128e3         # "128k sequences" as the book computes it
BCRIT_V5E, BCRIT_H100 = 240, 295


def params(L, D, F, N, K, H, V, experts=1):
    """Book's Q1 formula, shared input/output embedding."""
    return L * D * (3 * experts * F + 2 * H * (N + K)) + D * V


def kv_per_token(L, K, H, bytes_per=1):
    return 2 * L * K * H * bytes_per


def gb(x):
    return x / 1e9


print("== Q1: parameters and KV cache per token (int8) ==")
P = params(L, D, F, N, K, H, V)
mlp, attn, emb = L * D * 3 * F, L * 2 * D * H * (N + K), D * V
print(f"MLP  L*D*3F        = {mlp:,}")
print(f"attn L*2*D*H*(N+K) = {attn:,}")
print(f"emb  D*V           = {emb:,}")
print(f"total              = {P:,}  = {P / 1e9:.2f}e9   (book: 18.4e9)")
kv = kv_per_token(L, K, H)
print(f"KV/token 2*L*K*H   = {kv:,} B = {kv / 1e3:.0f} kB   (book: 262kB)")

print("\n== Q2: largest batch on v5e 4x4, int8, 128k context ==")
per_seq = kv * CTX
hbm_total = CHIPS * V5E["hbm"]
free = hbm_total - P
print(f"KV per sequence = {kv:,} * {CTX:.0f} = {gb(per_seq):.1f} GB")
print(f"HBM = 16 * 16e9 = {gb(hbm_total):.0f} GB; free after weights = {gb(free):.1f} GB")
print(f"batch = {gb(free):.1f} / {gb(per_seq):.1f} = {free / per_seq:.2f} -> {math.floor(free / per_seq)}   (book: 7)")
kv1 = kv_per_token(L, 1, H)
per_seq1 = kv1 * CTX
print(f"K=1: KV/token = {kv1:,} B, per seq = {gb(per_seq1):.2f} GB, "
      f"batch = {free / per_seq1:.2f} -> {math.floor(free / per_seq1)}   (book: 'about 56')")

print("\n== Q3: time to load all int8 params from HBM, 4x4 ==")
t_load = P / (V5E["hbm_bw"] * CHIPS)
print(f"{P / 1e9:.2f}e9 / (8.2e11 * 16) = {t_load * 1e3:.2f} ms   (book: 18e9 -> 1.4 ms)")

print("\n== Q4 (our solution, not the book's): sharding on v5e 4x4, int8 ==")
tp_1axis = F / 2200
tp_2axis = 2 * F / 2200
print(f"prefill model-parallel ICI bound F/2200: 1 axis = {tp_1axis:.1f}-way, 2 axes = {tp_2axis:.1f}-way")
per_chip_w_4 = P / 4
print(f"prefill plan: 4-way model x 4-way sequence; weights/chip = {gb(per_chip_w_4):.2f} GB")
beta = 8  # book: W_hbm / W_ici 'usually around 8' for v5e
for B in [7, 32]:
    print(f"decode ICI bound Y < F/(B*beta), B={B}: {F / (B * beta):.0f}-way")
print(f"decode plan: 16-way model; weights/chip = {gb(P / 16):.2f} GB; "
      f"KV sharded K={K} heads x {16 // K} batch")
for B, S in [(7, CTX), (32, 8192)]:
    kv_bytes = B * S * kv
    t = (P + kv_bytes) / (CHIPS * V5E["hbm_bw"])
    t_math = 2 * P * B / (CHIPS * V5E["int8"])
    print(f"  B={B:3d}, S={S:>8.0f}: KV = {gb(kv_bytes):6.1f} GB; "
          f"T_hbm = ({gb(P):.1f} + {gb(kv_bytes):.1f}) GB / {CHIPS * V5E['hbm_bw'] / 1e12:.2f} TB/s = {t * 1e3:.1f} ms; "
          f"T_math = {t_math * 1e6:.1f} us")
print(f"  latency-bound comms when Y > B*D/45000: B=7 -> {7 * D / 45000:.2f}  (so yes, latency-bound)")
colls, hops, t_hop = 6, 4, 1e-6   # illustrative: collectives per layer, hops per collective, 1 us/hop
t_comm = L * colls * hops * t_hop
print(f"  rough comms (illustrative): {L} layers * {colls} collectives * {hops} hops * 1 us = {t_comm * 1e3:.2f} ms")
print(f"  full-HBM ceiling: 16e9 / 8.2e11 = {V5E['hbm'] / V5E['hbm_bw'] * 1e3:.1f} ms per step")

print("\n== Q5: MoE, E=16, k=2 ==")
E, k = 16, 2
Ptot = params(L, D, F, N, K, H, V, experts=E)
Pact = params(L, D, F, N, K, H, V, experts=k)
print(f"total  = {Ptot / 1e9:.1f}e9  ({Ptot / P:.1f}x dense)   (book: 212e9, 'about 12x')")
print(f"active = {Pact / 1e9:.2f}e9  ({Pact / P:.2f}x dense)   (book: 31.2e9)")
print(f"B_crit = 240 * E/k = 240 * {E // k} = {BCRIT_V5E * E // k}   (book: 1920)")
print(f"KV/token unchanged = {kv:,} B")
print(f"FLOPs = 2 * {Pact / 1e9:.1f}e9 * T = {2 * Pact / 1e9:.1f}e9 * T")
print(f"(Part 4 Q8, DeepSeek int8 weights: 120 * 256/8 = {120 * 256 // 8})")

print("\n== Q6 (our solution, not the book's): expert sharding [E_Z, D_X, F_Y] on v5e 8x16 ==")
Y, Z = 8, 16
n6 = Y * Z
exp_p = L * 3 * E * D * F
attn_p = L * 2 * D * H * (N + K)
print(f"expert params = {exp_p / 1e9:.1f}e9, attention = {attn_p / 1e9:.2f}e9, embedding = {emb / 1e9:.2f}e9")
a_full = Ptot / n6
print(f"(a) everything sharded {n6}-way: {gb(a_full):.2f} GB/chip -> load {a_full / V5E['hbm_bw'] * 1e3:.2f} ms, "
      f"free {gb(V5E['hbm'] - a_full):.2f} GB/chip = {n6 * (V5E['hbm'] - a_full) / kv / 1e6:.1f}M tokens of KV")
b_chip = exp_p / n6 + (attn_p + emb) / Y
print(f"(b) experts {n6}-way, attention+embedding {Y}-way only: {gb(b_chip):.2f} GB/chip -> "
      f"load {b_chip / V5E['hbm_bw'] * 1e3:.2f} ms, free {gb(V5E['hbm'] - b_chip):.2f} GB/chip")
min_chips = Ptot / V5E["hbm"]
print(f"smallest slice: {gb(Ptot):.0f} GB / 16 GB = {min_chips:.1f} chips -> 16 (4x4); "
      f"per chip {gb(Ptot / 16):.2f} GB, left for KV {gb(16 * V5E['hbm'] - Ptot):.0f} GB total "
      f"= {(16 * V5E['hbm'] - Ptot) / kv:,.0f} tokens")

print("\n== Q7: 2D weight-stationary vs 1D model parallel ==")
print("T_2D = 2BD/(X W) + 4BF/(YZ W); with F=4D the optimum is X = sqrt(N/8), YZ = sqrt(8N)")
for n in [16, 64, 72, 128, 256]:
    X = math.sqrt(n / 8)
    YZ = n / X
    t2d = 2 / X + 4 * (F / D) / YZ          # in units of BD/W
    print(f"  N={n:4d}: X={X:5.2f}, YZ={YZ:6.2f}, T_2D = {t2d:.3f} BD/W  "
          f"(sqrt(128/N) = {math.sqrt(128 / n):.3f})  vs 1D 4/3 = {4 / 3:.3f}")
print(f"crossover N > 128 * (3/4)^2 = {128 * 0.75 ** 2:.0f}   (book: 72)")
print(f"general  N > 32 * (F/D) * (3/4)^2: F/D=4 -> {32 * 4 * 0.75 ** 2:.0f}; "
      f"Llama-3.1-8B F/D=3.5 -> {32 * 14336 / 4096 * 0.75 ** 2:.0f}")

print("\n== Same questions, Llama-3.1-8B on one H100 (our recomputation) ==")
lL, lD, lF, lN, lK, lH, lV = 32, 4096, 14336, 32, 8, 128, 128256
lP = lL * lD * 3 * lF + lL * (2 * lD * lH * lN + 2 * lD * lH * lK) + 2 * lD * lV  # untied embeddings
print(f"Q1 params = {lP / 1e9:.2f}e9 (untied embeddings: 2*D*V)")
for label, bpp in [("bf16", 2), ("int8/fp8", 1)]:
    lkv = kv_per_token(lL, lK, lH, bpp)
    w = lP * bpp
    free_h = H100["hbm"] - w
    seq = lkv * CTX
    lkv1 = kv_per_token(lL, 1, lH, bpp)
    print(f"  {label:8s}: weights {gb(w):.2f} GB, KV/token {lkv:,} B, per 128k seq {gb(seq):.2f} GB, "
          f"Q2 batch = {gb(free_h):.1f}/{gb(seq):.2f} = {free_h / seq:.2f} -> {math.floor(free_h / seq)}; "
          f"K=1 -> {math.floor(free_h / (lkv1 * CTX))}; "
          f"Q3 load = {w / H100['hbm_bw'] * 1e3:.2f} ms")
lPtot = lL * lD * 3 * E * lF + lL * 2 * lD * lH * (lN + lK) + 2 * lD * lV
lPact = lL * lD * 3 * k * lF + lL * 2 * lD * lH * (lN + lK) + 2 * lD * lV
print(f"Q5-style MoE (E=16, k=2): total {lPtot / 1e9:.1f}e9, active {lPact / 1e9:.1f}e9, "
      f"B_crit = 295 * 8 = {BCRIT_H100 * E // k} (bf16); FLOPs = {2 * lPact / 1e9:.1f}e9 * T")

# Try this:
# 1. Set K = 1 (MQA) at the top and rerun: Q2's batch jumps to 56, and Q4's decode step shrinks.
# 2. Change CTX to 8192: how many 8k-token users fit on the 4x4? (and on the H100?)
# 3. Set E, k = 64, 4 in the Q5 block: total params and B_crit (= 240*E/k) grow 4x and 2x; active params grow only with k.
