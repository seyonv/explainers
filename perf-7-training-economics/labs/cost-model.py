"""$ per million tokens and the batch dial.

Run: python3 labs/cost-model.py
Stdlib only, CPU, well under a second.

Prints every number on cost-per-token.html:
  1. The batch dial: course 2's step formula -> tok/s -> $/M output tokens (the _facts.md table)
  2. Per-user speed vs $/M, and the "cheaper for slower" ratios (vs the Scaling Book's ~100x for 2x)
  3. Input tokens: the prefill floor (compute-bound) -> $/M input
  4. Utilization: an idle GPU still bills
  5. Levers as multipliers: FP8 weights, FP8 KV, shorter context, cheaper GPU-hours
  6. The market (OpenRouter, 2026-09-24) vs our ceiling
  7. DeepSeek Day 6 as a P&L (their numbers; derived $/M are our arithmetic)

Everything here is a ceiling ("our model"): 100% of HBM bandwidth and peak FLOPs,
output tokens only unless stated, the GPU busy 100% of the hour.
"""

# ---- hardware and price (_facts.md; prices seen 2026-09-24)
BW = 3.35e12          # H100 SXM HBM bytes/s
C_BF16 = 989e12       # dense BF16 FLOP/s
C_FP8 = 1979e12       # dense FP8 FLOP/s
HBM = 80e9            # bytes
PRICE = 3.99          # $/GPU-hour, Lambda 8x H100 on-demand (the course default)

# ---- Llama-3.1-8B
N = 8.03e9            # params
KV_TOK = 128 * 1024   # bytes of BF16 KV cache per token (course 1)
CTX = 2048            # tokens of context per sequence (the running example)

BATCHES = [1, 8, 16, 32, 64, 128, 238]


def step_s(B, ctx=CTX, wbytes=2, kvbytes=2, flops=C_BF16):
    """Course 2: step = B*KV_seq/BW + max(2*B*N/C, W/BW)."""
    W = N * wbytes
    kv_seq = ctx * KV_TOK * kvbytes / 2
    return B * kv_seq / BW + max(2 * B * N / flops, W / BW)


def usd_per_m(tok_s, price=PRICE, util=1.0):
    """$/M = GPU $/hr / (tok/s * 3600 * utilization) * 1e6."""
    return price / (tok_s * 3600 * util) * 1e6


def b_max(ctx=CTX, wbytes=2, kvbytes=2):
    """Largest batch whose KV cache fits in HBM next to the weights (no activation headroom)."""
    return int((HBM - N * wbytes) / (ctx * KV_TOK * kvbytes / 2))


def rule(t):
    print()
    print("=" * 76)
    print(t)
    print("=" * 76)


rule(f"1. The batch dial: Llama-3.1-8B BF16, one H100, {CTX} context, ${PRICE}/GPU-hr")
print(f"weights W = {N*2/1e9:.2f} GB -> W/BW = {N*2/BW*1e3:.3f} ms;"
      f" KV per sequence = {CTX*KV_TOK/1e6:.1f} MB -> {CTX*KV_TOK/BW*1e3:.4f} ms each")
print(f"batch that fills 80 GB with KV: ({HBM/1e9:.0f} - {N*2/1e9:.2f}) GB / {CTX*KV_TOK/1e6:.1f} MB"
      f" = {b_max()}")
print(f"{'B':>5}{'step ms':>9}{'KV ms':>8}{'weights|FLOPs ms':>18}{'tok/s':>9}{'per user':>10}{'$/M out':>9}")
rows = {}
for B in BATCHES:
    s = step_s(B)
    kv = B * CTX * KV_TOK / BW
    body = max(2 * B * N / C_BF16, N * 2 / BW)
    tps = B / s
    rows[B] = (s, tps, usd_per_m(tps))
    print(f"{B:>5}{s*1e3:>9.2f}{kv*1e3:>8.2f}{body*1e3:>18.2f}{tps:>9,.0f}{1/s:>10.0f}{usd_per_m(tps):>9.2f}")
s1, t1, c1 = rows[1]
BTOP = BATCHES[-1]   # the biggest batch that fits (238 at 2k context)
s_top, t_top, c_top = rows[BTOP]
print(f"worked: {PRICE} / ({t_top:,.0f} x 3600) x 1e6 = {c_top:.4f} $/M  (B={BTOP})")
print(f"worked: {PRICE} / ({t1:,.0f} x 3600) x 1e6 = {c1:.4f} $/M  (B=1)")

rule("2. Cheaper for slower: cost ratio vs per-token latency ratio (vs B=1)")
for B in BATCHES[1:]:
    s, t, c = rows[B]
    print(f"B={B:>3}: {c1/c:>5.1f}x cheaper per token for {s/s1:.2f}x the per-token latency"
          f"  ({1/s1:.0f} -> {1/s:.0f} tok/s per user)")
print("Scaling Book Part 8 (LLaMA 3-70B int8, 16 TPU v5e): ~100x cheaper for 2x latency;"
      " its printed code gives 77x for 1.57x (course 2)")
w_tokens_8b = N * 2 / KV_TOK
w_tokens_70b = 70e9 / 160e3
print(f"why ours is flatter (our inference): 8B weights = KV of {w_tokens_8b/1e3:,.0f}k tokens;"
      f" 70B int8 weights = KV of {w_tokens_70b/1e3:,.0f}k tokens (160 kB/token)")

rule("3. Input tokens: prefill is compute-bound")
pre = 2 * N * 1000 / C_BF16
pre_tps = 1000 / pre
print(f"prefill floor, 1,000 tokens: 2 x {N:.2e} x 1000 / {C_BF16:.3g} = {pre*1e3:.2f} ms"
      f" -> {pre_tps:,.0f} tok/s")
print(f"$/M input at 100% of peak = {usd_per_m(pre_tps):.4f}; at 50% of peak = {usd_per_m(pre_tps, util=0.5):.4f}")
print(f"output at B={BTOP} / input ceiling = {c_top/usd_per_m(pre_tps):.1f}x;"
      f" at B=1 = {c1/usd_per_m(pre_tps):.0f}x")

rule("4. Utilization: the GPU bills for every hour, busy or not")
for u in (1.0, 0.75, 0.5, 0.25):
    print(f"busy {u:>4.0%} of the day: B={BTOP} -> ${usd_per_m(t_top, util=u):.2f}/M;"
          f" B=64 -> ${usd_per_m(rows[64][1], util=u):.2f}/M")

rule("5. Levers as multipliers on the biggest-batch row (all 'our model')")
base = c_top


def lever(name, B, **kw):
    ctx = kw.pop("ctx", CTX)
    price = kw.pop("price", PRICE)
    s = step_s(B, ctx=ctx, **kw)
    c = usd_per_m(B / s, price=price)
    print(f"{name:<44} B={B:>3}  step {s*1e3:6.2f} ms  {B/s:>7,.0f} tok/s  ${c:.3f}/M  ({base/c:.2f}x cheaper)")
    return c


lever("BF16 baseline", BTOP)
lever("FP8 weights (KV BF16)", b_max(wbytes=1), wbytes=1, flops=C_FP8)
fp8 = lever("FP8 weights + FP8 KV", b_max(wbytes=1, kvbytes=1), wbytes=1, kvbytes=1, flops=C_FP8)
lever("BF16, 1k context instead of 2k", b_max(ctx=1024), ctx=1024)
lever("FP8 weights + KV at RunPod $3.49", b_max(wbytes=1, kvbytes=1), wbytes=1, kvbytes=1, flops=C_FP8,
      price=3.49)
lever("FP8 weights + KV at AWS spot $2.60", b_max(wbytes=1, kvbytes=1), wbytes=1, kvbytes=1, flops=C_FP8,
      price=2.60)
print(f"FP8 at B=1: step {step_s(1, wbytes=1, flops=C_FP8)*1e3:.2f} ms,"
      f" ${usd_per_m(1/step_s(1, wbytes=1, flops=C_FP8)):.2f}/M (vs ${c1:.2f})")

rule("6. The market vs our ceiling (OpenRouter endpoints, 2026-09-24, $/M in / out)")
market = [("DeepInfra FP8", 0.02, 0.04), ("Groq", 0.05, 0.08), ("CoreWeave BF16", 0.22, 0.22)]
for name, i, o in market:
    print(f"{name:<16} in ${i:.2f}  out ${o:.2f}  -> out = {o/c_top:.2f}x our BF16 B={BTOP} ceiling (${c_top:.2f});"
          f" {o/fp8:.2f}x our FP8 ceiling (${fp8:.3f})")

rule("7. DeepSeek Day 6: one day of V3/R1 serving (their numbers)")
nodes, gpus, lease = 226.75, 8, 2.0
cost = nodes * gpus * lease * 24
rev = 562_027
print(f"cost = {nodes} nodes x {gpus} H800 x ${lease}/hr x 24 = ${cost:,.0f}/day")
print(f"revenue at R1 prices (theoretical) = ${rev:,}/day")
print(f"profit / cost = {(rev-cost)/cost:.1%}  (their '545% cost profit margin');"
      f" profit / revenue = {(rev-cost)/rev:.1%}")
chk = 342e3 * 0.14 + 266e3 * 0.55 + 168e3 * 2.19   # millions of tokens x $/M
print(f"revenue check: 342B x $0.14 + 266B x $0.55 + 168B x $2.19 per M = ${chk:,.0f}")
node_hr = gpus * lease
print(f"decode node: ${node_hr:.0f}/hr / (14.8k tok/s x 3600) = ${usd_per_m(14.8e3, price=node_hr):.3f}/M output")
print(f"prefill node: ${node_hr:.0f}/hr / (73.7k tok/s x 3600) = ${usd_per_m(73.7e3, price=node_hr):.3f}/M input")
print(f"all-in: ${cost:,.0f} / 168B output tokens = ${cost/168e3:.3f}/M output")
print(f"per H800 GPU: decode {14.8e3/8:,.0f} out tok/s; our 8B H100 B={BTOP} ceiling {t_top:,.0f}")

# Try this:
# 1. Set PRICE = 2.0 (DeepSeek's assumed H800 lease) and re-read section 1: $0.06/M at B=238.
# 2. Set CTX = 8192 and BATCHES = [1, 8, 16, 32, 59]: the KV cache fills 80 GB at B=59,
#    and the cheapest row costs about 4x more than at 2k.
# 3. Set C_BF16 = 989e12 * 0.5 (half of peak FLOPs, closer to practice): input $/M doubles,
#    decode rows up to B=128 don't move (still weight-read bound), and only B=238 gets
#    slower (23.86 -> 26.80 ms), because 2*B*N/C now beats W/BW above B ~ 148.
