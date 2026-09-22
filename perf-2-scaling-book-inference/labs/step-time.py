"""Step-time lab: Scaling Book Part 7, "Theoretical estimates for LLM latency and throughput"

Run: python3 labs/step-time.py   (stdlib only, CPU, well under a second)

The book's general decode step time (a lower bound: perfect overlap, peak bandwidth):

  step = B * KV / BW                         attention: always bandwidth-bound
       + max(2 * B * N_params / FLOPs,       MLP (all the weights): compute ...
             param_bytes / BW)               ... or the weight read, whichever is longer

  with KV = KV-cache bytes per sequence (context * bytes per token).
  Small-batch special case (book): min step = (B * KV + param_bytes) / BW
  Max throughput (book):           tokens/s = B * BW / (B * KV + param_bytes)
"""

GB = 1e9


def step_time(B, kv_seq, p_bytes, n_params, bw, flops):
    """Returns (total, attention/KV read, MLP compute, weight read) in seconds."""
    t_kv = B * kv_seq / bw
    t_math = 2 * B * n_params / flops
    t_w = p_bytes / bw
    return t_kv + max(t_math, t_w), t_kv, t_math, t_w


def dominant(t_kv, t_math, t_w):
    mlp = "MLP math" if t_math > t_w else "weight read"
    # compare the two terms that actually add up: KV read vs the MLP max()
    return "KV read" if t_kv > max(t_math, t_w) else mlp


# ---------------------------------------------------------------- 1. Pop Quiz
print("== 1. The book's Pop Quiz: 30B dense, int8 weights, bf16 FLOPs, TPU v5e 4x4 ==")
chips = 16
bw_v5e, fl_v5e, hbm_v5e = 8.2e11, 1.97e14, 16e9
P_q, N_q = 30e9, 30e9          # int8: 1 byte per parameter
kv_q = 100e3 * 8192            # 100 kB/token * 8192 context
print(f"params: {P_q / GB:.0f} GB   KV per sequence: 100e3 * 8192 = {kv_q / 1e6:.0f} MB")
print(f"total bandwidth: {chips} * 8.2e11 = {chips * bw_v5e:.3g} B/s   total FLOPs: {chips} * 1.97e14 = {chips * fl_v5e:.3g}/s")
for B in (4, 256):
    small = (B * kv_q + P_q) / (chips * bw_v5e)
    tot, tkv, tm, tw = step_time(B, kv_q, P_q, N_q, chips * bw_v5e, chips * fl_v5e)
    print(f"B={B:3d}: min-step formula (B*KV + P)/BW = {small * 1e3:6.2f} ms | general: KV {tkv * 1e3:6.2f} ms"
          f" + max(math {tm * 1e3:5.2f}, weights {tw * 1e3:5.2f}) = {tot * 1e3:6.2f} ms"
          f" -> {B / tot:,.0f} tok/s total, {B / tot / chips:,.0f} tok/s/chip")
print(f"memory check at B=256: {256 * kv_q / GB:.1f} GB KV + {P_q / GB:.0f} GB weights = "
      f"{(256 * kv_q + P_q) / GB:.1f} GB of {chips * hbm_v5e / GB:.0f} GB HBM")
print("(book: 2.5 ms and 'roughly ... 21ms')")

# ------------------------------------------------ 2. Llama-3.1-8B on one H100
print("\n== 2. Our recomputation: Llama-3.1-8B BF16 on one H100 SXM ==")
N = 8.03e9
P = 16.06e9
KV_TOK = 131072                 # 2 * L * K * H * 2 bytes = 128 KiB
H100 = dict(bw=3.35e12, flops=9.89e14, hbm=80e9)
free = H100["hbm"] - P
print(f"weights {P / GB:.2f} GB, KV {KV_TOK:,} B/token, free HBM {free / GB:.2f} GB")
ROWS = {}
for ctx in (2048, 8192):
    kv = ctx * KV_TOK
    bmax = int(free // kv)
    print(f"\n-- context {ctx:,}: KV per sequence {kv / 1e6:.1f} MB; HBM holds at most B = {bmax} --")
    print(f"{'B':>4} | {'KV GB':>6} | {'KV ms':>6} | {'W ms':>5} | {'math ms':>7} | {'step ms':>7} | "
          f"{'tok/s/user':>10} | {'tok/s total':>11} | dominant")
    rows = []
    for B in sorted({1, 8, 32, 64, 128, 256, bmax}):
        if B > bmax:
            print(f"{B:4d} | {B * kv / GB:6.1f} | out of memory: needs {(B * kv + P) / GB:.1f} GB > 80 GB")
            continue
        tot, tkv, tm, tw = step_time(B, kv, P, N, H100["bw"], H100["flops"])
        rows.append((B, tot, tkv, tm, tw))
        print(f"{B:4d} | {B * kv / GB:6.2f} | {tkv * 1e3:6.2f} | {tw * 1e3:5.2f} | {tm * 1e3:7.3f} | "
              f"{tot * 1e3:7.2f} | {1 / tot:10.1f} | {B / tot:11,.0f} | {dominant(tkv, tm, tw)}")
    ROWS[ctx] = rows
    print(f"   KV read = weight read at B = P / KV = {P / kv:.1f};  "
          f"throughput ceiling as B -> inf (attention-bound): BW / KV = {H100['bw'] / kv:,.0f} tok/s")
print(f"\nB where MLP math = weight read (the MLP's B_crit): P / (2 N / FLOPs * BW) = "
      f"{P / (2 * N / H100['flops'] * H100['bw']):.1f}")

# ------------------------------------------------ 3. Pareto points
print("\n== 3. Pareto points (per-token latency = step time, throughput per chip) ==")
for ctx in (2048, 8192):
    pts = ", ".join(f"B={B}: ({tot * 1e3:.2f} ms, {B / tot:,.0f})" for B, tot, *_ in ROWS[ctx])
    print(f"H100 ctx {ctx}: {pts}")
pq = []
for B in (1, 4, 16, 32, 64, 128, 240, 256):
    tot = step_time(B, kv_q, P_q, N_q, chips * bw_v5e, chips * fl_v5e)[0]
    pq.append(f"B={B}: ({tot * 1e3:.2f} ms, {B / tot / chips:,.0f})")
print("Pop Quiz setup (v5e 4x4, per chip):", ", ".join(pq))

# ------------------------------------------------ 4. The alternatives
print("\n== 4. The simpler rules, at 2k context ==")
naive = H100["bw"] / P
print(f"'tok/s = BW / params': {H100['bw']:.3g} / {P:.4g} = {naive:.1f} tok/s at any batch")
for B, tot, *_ in ROWS[2048]:
    if B in (1, 128):
        print(f"   formula at B={B}: {B / tot:,.0f} tok/s total ({B / tot / naive:.1f}x the naive rule)")
t_flops = 2 * N / H100["flops"]
tot1 = ROWS[2048][0][1]
print(f"'FLOPs / peak' at B=1: 2 * 8.03e9 / 9.89e14 = {t_flops * 1e6:.1f} us per token "
      f"vs weights-only {P / H100['bw'] * 1e3:.2f} ms ({P / H100['bw'] / t_flops:.0f}x optimistic); "
      f"vs 2k-context step {tot1 * 1e3:.2f} ms ({tot1 / t_flops:.0f}x)")

# ------------------------------------------------ 5. Other GPUs
print("\n== 5. Same model, 2k context, other GPUs (book Part 12 specs) ==")
GPUS = {"H100": (3.35e12, 9.89e14, 80e9), "H200": (4.8e12, 9.89e14, 141e9), "B200": (8.0e12, 2.25e15, 192e9)}
kv = 2048 * KV_TOK
for name, (bw, fl, hbm) in GPUS.items():
    bmax = int((hbm - P) // kv)
    t1 = step_time(1, kv, P, N, bw, fl)[0]
    tm, tkv, tmm, tw = step_time(bmax, kv, P, N, bw, fl)
    print(f"{name}: B=1 {t1 * 1e3:.2f} ms ({1 / t1:.0f} tok/s) | max B {bmax}: {tm * 1e3:.2f} ms, "
          f"{1 / tm:.1f} tok/s/user, {bmax / tm:,.0f} tok/s total | ceiling BW/KV {bw / kv:,.0f}")

# ------------------------------------------------ 6. perf-1 serving-metrics consistency
print("\n== 6. Cross-check with perf-1 serving-metrics (~1,100 tokens of context = 144.2 MB) ==")
kv = 1100 * KV_TOK
for B in (1, 120, 443):
    mem_only = (B * kv + P) / H100["bw"]
    tot, *_ = step_time(B, kv, P, N, H100["bw"], H100["flops"])
    print(f"B={B:3d}: min-step (memory only) {mem_only * 1e3:6.2f} ms -> {B / mem_only:7,.0f} tok/s | "
          f"general {tot * 1e3:6.2f} ms -> {B / tot:7,.0f} tok/s")

# ------------------------------------------------ 7. SVG coordinates
print("\n== 7. SVG points: x = step ms (0..30 -> 70..770), y = tok/s total (0..14000 -> 290..40) ==")
for ctx in (2048, 8192):
    for B, tot, *_ in ROWS[ctx]:
        x = 70 + tot * 1e3 / 30 * 700
        y = 290 - B / tot / 14000 * 250
        print(f"ctx {ctx} B={B:3d}: ({x:.1f}, {y:.1f})")

# Try this:
# 1. FP8 weights: set P = 8.03e9 (1 byte/param). The weight read halves (4.79 -> 2.40 ms), B=1 at 2k goes
#    4.87 -> 2.48 ms, and more HBM is free for KV, but the BW/KV throughput ceiling doesn't move.
# 2. GQA -> MHA: set KV_TOK = 4 * 131072 (K = 32 KV heads instead of 8). At 8k context only 14 sequences
#    fit and KV dominates from B = 4.
# 3. H200: set H100 = dict(bw=4.8e12, flops=9.89e14, hbm=141e9). Everything memory-bound gets 1.43x
#    faster, and the extra HBM lets the batch run past B_crit, where the MLP turns compute-bound.
