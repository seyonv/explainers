"""Modelling LLaMA 2-13B on 8x TPU v5e: Scaling Book Part 7,
"Modeling throughput and latency for LLaMA 2-13B".

Run: python3 labs/modeling-13b.py   (stdlib only, CPU, well under a second)

The book's general step-time formula (decode, one new token per sequence):

  step = B * KV_seq / W_hbm  +  max( 2 * B * P / C ,  P_bytes / W_hbm )
         attention: always          MLP/linears: memory-bound until
         memory-bound               B passes B_crit

  throughput = B / step          (tokens/s, B sequences each emit 1 token)

B = sequences in the batch (= tokens per decode step), KV_seq = KV cache bytes
of one sequence, P = parameter count, P_bytes = 2P in bf16, C = total FLOP/s,
W_hbm = total HBM bytes/s.
"""

GB, GiB = 1e9, 2**30
BATCHES = [1, 8, 16, 32, 64, 240]

# ---- The book's two tables, verbatim ------------------------------------
BOOK = {
    "MHA (K=40)": dict(
        kv=[6.7, 53.6, 107.2, 214.4, 428.8, 1608],
        tot=[32.7, 79.6, 133.2, 240.4, 454.8, 1634],
        step=[4.98, 12.13, 20.30, 36.65, 69.33, 249.09],
        thr=[200.61, 659.30, 787.99, 873.21, 923.13, 963.53]),
    "GQA 1:5 (K=8)": dict(
        kv=[1.34, 10.72, 21.44, 42.88, 85.76, 321.6],
        tot=[27.34, 36.72, 47.44, 68.88, 111.76, 347.6],
        step=[4.17, 5.60, 7.23, 10.50, 17.04, 52.99],
        thr=[239.94, 1429.19, 2212.48, 3047.62, 3756.62, 4529.34]),
}

# ---- LLaMA 2-13B shapes (book table) ------------------------------------
L, D, F, N, H, V = 40, 5120, 13824, 40, 128, 32000
T = 8192  # context length the book uses for the KV cache


def llama_params(L, D, F, N, K, H, V):
    ffw = 3 * D * F * L
    attn = (2 * D * N * H + 2 * D * K * H) * L
    vocab = 2 * V * D
    return ffw + attn + vocab


def kv_bytes(L, K, H, T, bytes_per=2):
    return 2 * bytes_per * H * K * L * T


def step_time(B, kv_seq, p_bytes, p_count, bw, flops):
    attn = B * kv_seq / bw
    mlp = max(2 * B * p_count / flops, p_bytes / bw)
    return attn + mlp


def pct(a, b):
    return (a / b - 1) * 100


P13 = llama_params(L, D, F, N, 40, H, V)   # the book keeps params fixed for GQA
print("== LLaMA 2-13B, first principles ==")
print(f"params = 3DFL + (2DNH + 2DKH)L + 2VD = {P13:,}  -> bf16 {2 * P13 / GB:.2f} GB ({2 * P13 / GiB:.2f} GiB)")
for K in (40, 8):
    kv = kv_bytes(L, K, H, T)
    print(f"KV per {T}-token sequence, K={K}: 2*2*H*K*L*T = {kv:,} B = {kv / GB:.3f} GB = {kv / GiB:.3f} GiB")

# ---- 8x TPU v5e --------------------------------------------------------
CHIPS = 8
BW_CHIP, FLOPS_CHIP = 8.2e11, 1.97e14            # book Parts 1, 7
BW, FLOPS = CHIPS * BW_CHIP, CHIPS * FLOPS_CHIP  # 6.56e12 B/s, 1.576e15 FLOP/s
print(f"\n8x v5e: W_hbm = 8 x 8.2e11 = {BW:.3e} B/s (book: '6.5TiB/s (0.82TiB/s each)'),"
      f" C = {FLOPS:.3e} FLOP/s (book: 1600 TF/s)")

# (A) The book's own inputs: params 26 'GiB', KV 6.7 / 1.34 'GiB', W = 6.56e12,
#     with every 'GiB' treated as 1e9 bytes. This reproduces the book exactly.
# (B) Exact bytes from first principles, same W and C.
# (C) Literal reading of the labels: memory in GiB (2^30), bandwidth 6.5 TiB/s (2^40).
for name, K in (("MHA (K=40)", 40), ("GQA 1:5 (K=8)", 8)):
    book = BOOK[name]
    kv_exact = kv_bytes(L, K, H, T)
    kv_book = book["kv"][0] * GB
    print(f"\n== {name}: book vs our recomputation ==")
    print(f"{'B':>4} | {'book KV':>7} {'book tot':>8} | {'book ms':>7} {'(A) ms':>7} {'(B) ms':>7} {'(C) ms':>7} | "
          f"{'book tok/s':>10} {'(A) tok/s':>9} {'(B) tok/s':>9} | {'B vs book':>9} {'C vs book':>9}")
    for i, B in enumerate(BATCHES):
        sA = step_time(B, kv_book, 26 * GB, 13e9, BW, 1.6e15)
        sB = step_time(B, kv_exact, 2 * P13, P13, BW, FLOPS)
        sC = (book["tot"][i] * GiB) / (6.5 * 2**40)
        print(f"{B:4d} | {book['kv'][i]:7.2f} {book['tot'][i]:8.2f} | {book['step'][i]:7.2f} {sA * 1e3:7.2f} "
              f"{sB * 1e3:7.2f} {sC * 1e3:7.2f} | {book['thr'][i]:10.2f} {B / sA:9.2f} {B / sB:9.2f} | "
              f"{pct(sB * 1e3, book['step'][i]):+8.2f}% {pct(sC * 1e3, book['step'][i]):+8.2f}%")
    worstA = max(abs(pct(step_time(B, kv_book, 26 * GB, 13e9, BW, 1.6e15) * 1e3, book["step"][i]))
                 for i, B in enumerate(BATCHES))
    print(f"(A) worst |diff| vs book step time: {worstA:.2f}%  (book step times are rounded to 0.01 ms)")
    print("per chip (tok/s / 8):", [round(t / 8, 1) for t in book["thr"]])
    print("per user (1000 / step ms):", [round(1000 / s_, 1) for s_ in book["step"]])
    # asymptote as B -> infinity (attention term dominates): W / KV_seq
    print(f"throughput ceiling as B -> inf: W_hbm / KV_seq = {BW / kv_book:.0f} tok/s "
          f"({BW_CHIP / kv_book:.1f} tok/s per chip, independent of chip count)")
    print(f"batch at which KV bytes = param bytes (half the ceiling): 26 / {book['kv'][0]} = {26 / book['kv'][0]:.1f}")

# ---- Which rows fit in HBM? ---------------------------------------------
print("\n== Memory limit: which batches fit? ==")
for label, hbm in (("128 GB  (8 x 16e9)", 128 * GB), ("128 GiB (8 x 16 GiB)", 128 * GiB)):
    for name in BOOK:
        kv1 = BOOK[name]["kv"][0] * GB
        bmax = int((hbm - 26 * GB) // kv1)
        B = min(bmax, 10**6)
        s = step_time(B, kv1, 26 * GB, 13e9, BW, 1.6e15)
        print(f"HBM {label}: {name:14s} max batch = {bmax:3d}  -> {B / s:7.1f} tok/s at that batch (book inputs)")
b16 = BOOK["MHA (K=40)"]["tot"][2]
print(f"batch 16 MHA total = {b16} (x1e9 B) = {b16 * GB / GiB:.1f} GiB: over 128 GB, under 128 GiB")
g, m = BOOK["GQA 1:5 (K=8)"]["thr"][4], BOOK["MHA (K=40)"]["thr"][2]
print(f"headline: GQA batch 64 / MHA batch 16 = {g} / {m} = {g / m:.2f}x throughput")
print(f"          per chip: {m / 8:.1f} -> {g / 8:.1f} tok/s/chip")
for i, B in enumerate(BATCHES):
    a, b = BOOK["MHA (K=40)"]["step"][i], BOOK["GQA 1:5 (K=8)"]["step"][i]
    print(f"latency at B={B:3d}: MHA {a:6.2f} ms  GQA {b:6.2f} ms  ({a / b:.2f}x faster per step)")

for T_alt in (1024, 4096, 32768):
    kv_alt = 6.7 * GB * T_alt / 8192
    print(f"MHA at {T_alt:5d}-token context: KV/seq {kv_alt / GB:.2f} GB, max batch in 128 GB = {int((128 * GB - 26 * GB) // kv_alt)}")

# ---- More chips only, MHA: does per-chip throughput improve? ------------
print("\n== MHA with more chips (our recomputation, book inputs, 16e9 B HBM per chip) ==")
kv1 = 6.7 * GB
for chips in (8, 16, 32, 64):
    hbm, bw = chips * 16 * GB, chips * BW_CHIP
    bmax = int((hbm - 26 * GB) // kv1)
    s = step_time(bmax, kv1, 26 * GB, 13e9, bw, chips * FLOPS_CHIP)
    print(f"{chips:3d} chips: max batch {bmax:4d}, step {s * 1e3:6.2f} ms, {bmax / s:8.1f} tok/s total, "
          f"{bmax / s / chips:6.1f} tok/s per chip")

# ---- Same exercise, our recomputation: Llama-3.1-8B on one H100 --------
print("\n== Llama-3.1-8B on 1x H100 SXM, 8k context (our recomputation, not in the book) ==")
L8, D8, F8, N8, H8, V8 = 32, 4096, 14336, 32, 128, 128256
P8 = llama_params(L8, D8, F8, N8, 8, H8, V8)
HBM_H, BW_H, C_H = 80 * GB, 3.35e12, 9.89e14
T8 = 8192
print(f"params {P8 / 1e9:.2f}B -> bf16 {2 * P8 / GB:.2f} GB; free HBM {(HBM_H - 2 * P8) / GB:.1f} GB")
for name, K in (("MHA hypothetical (K=32)", 32), ("real GQA-8 (K=8)", 8)):
    kv = kv_bytes(L8, K, H8, T8)
    bmax = int((HBM_H - 2 * P8) // kv)
    print(f"-- {name}: KV/seq {kv / GB:.2f} GB, max batch {bmax}")
    for B in [1, 8, 14, 16, 32, 59, 64]:
        s = step_time(B, kv, 2 * P8, P8, BW_H, C_H)
        tot = (2 * P8 + B * kv) / GB
        fit = "fits" if B <= bmax else "OOM"
        print(f"   B={B:3d}: mem {tot:7.1f} GB {fit:4s}  step {s * 1e3:6.2f} ms  {B / s:8.1f} tok/s")
    print(f"   ceiling W/KV = {BW_H / kv:.0f} tok/s")

# ---- SVG helper: throughput-vs-batch points (log2 x axis) ---------------
print("\n== SVG points (book values) ==")
for name in BOOK:
    print(name, [(B, t) for B, t in zip(BATCHES, BOOK[name]["thr"])])

# Try this:
# 1. int8 KV cache: give kv_bytes() bytes_per=1 and divide the BOOK "kv" inputs by 2.
#    MHA's KV halves to 3.36 GB/seq, ~30 sequences fit in 128 GB instead of 15,
#    and the throughput ceiling W/KV doubles to ~1,950 tok/s.
# 2. 32k context: set T = 32768 and T8 = 32768. MHA 13B then needs 26.8 GB per sequence,
#    so only 3 sequences fit on 8 chips; GQA-8 Llama-3.1-8B on one H100 drops from 59 to 14.
