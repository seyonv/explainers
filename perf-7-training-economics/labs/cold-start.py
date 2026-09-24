"""Autoscaling, cold starts and scale to zero (course 7, card 9).

Run: python3 labs/cold-start.py   (stdlib + numpy, CPU, under a second)

Nothing here is measured. It is a calculator:
  (1) Weight-load time = bytes / bandwidth for each storage tier (Llama-3.1-8B, Llama-70B class).
  (2) Time to first token after a scale-up, summed over Kiely's four cold-start parts, for three
      setups. Every component time except the weight load is ILLUSTRATIVE (no source gives them).
  (3) ServerlessLLM's published numbers, and the ratios they imply.
  (4) The cost side: an ILLUSTRATIVE bursty day of traffic, served four ways at $3.99/GPU-hr
      (Lambda 8x H100, course default): provision for the peak, autoscale with min 2, min 1,
      and scale to zero. GPU-hours, $/day, effective $/M tokens, and how many cold starts users see.
  (5) What a burst costs while a new replica is still starting.
  (6) One replica kept warm for light traffic: $/M tokens vs a pay-per-token API.
"""
import numpy as np

GB = 1e9
PRICE = 3.99                    # $/GPU-hr, Lambda H100 SXM 8x (course facts)
W_8B = 16.06e9                  # Llama-3.1-8B BF16 weights, bytes (course facts)
H100_C = 989e12                 # dense BF16 FLOP/s
CAP = 6450                      # output tok/s per H100 replica at batch 64, 2k context (course facts, "our model")

# ------------------------------------------------------------------ (1) weight-load math
TIERS = [  # name, bytes/s, where the number comes from
    ("Hugging Face / S3 over 1 Gbps", 1e9 / 8, "1 Gbps link (ServerlessLLM testbed i, KServe run)"),
    ("Registry / S3 over 10 Gbps", 10e9 / 8, "10 Gbps link (ServerlessLLM testbed ii)"),
    ("In-region object store, 5 GB/s", 5e9, "ServerlessLLM: 'fast commodity network capable of 5GB/s'"),
    ("Local NVMe RAID 0, 12 GB/s", 12e9, "ServerlessLLM loader benchmark disk"),
    ("Host DRAM -> GPU, PCIe 5", 512e9 / 8, "ServerlessLLM: 512 GB/s over 8 GPUs = 64 GB/s each"),
]
print("(1) Weight-load time = bytes / bandwidth   (a floor: real loaders add deserialization, page faults)")
print(f"    {'tier':34s} {'GB/s':>7s} {'16 GB (8B)':>11s} {'8.03 GB FP8':>12s} {'140 GB (70B)':>13s}")
for name, bw, _ in TIERS:
    t16, t8, t140 = 16e9 / bw, 8.03e9 / bw, 140e9 / bw
    if "PCIe" in name:
        t140 = 140e9 / 512e9     # 70B is split over 8 GPUs, each on its own PCIe link
    print(f"    {name:34s} {bw / GB:7.3f} {t16:10.2f}s {t8:11.2f}s {t140:12.2f}s")
print(f"    exact 8B weights: {W_8B / GB:.2f} GB at 1 Gbps = {W_8B / (1e9 / 8):.1f} s")
print("    70B on PCIe assumes 8 GPUs load in parallel (512 GB/s aggregate).")
print(f"    check vs ServerlessLLM: 130 GB at 5 GB/s = {130e9 / 5e9:.0f} s (paper: 'a minimum of 26 seconds')")
print(f"    check vs KServe: OPT-6.7B ~{6.7e9 * 2 / GB:.1f} GB at 1 Gbps = {6.7e9 * 2 / (1e9 / 8):.0f} s (paper measured 114 s)")

# ------------------------------------------------------------------ (2) TTFT after a scale-up
# ILLUSTRATIVE component times. Only the weight load is computed from bytes / bandwidth.
IMAGE = 10e9                    # illustrative inference image size (Kiely: "often many gigabytes")
PREFILL = 2 * 8.03e9 * 1000 / H100_C      # a 1,000-token prompt at 100% of dense BF16
SETUPS = [
    # name, GPU procurement s, image bandwidth, weight bandwidth, engine startup s
    ("Naive: new cloud node, pull all", 300, 10e9 / 8, 1e9 / 8, 180),
    ("Tuned: warm node pool, cached image", 0, None, 5e9, 30),
    ("Best: weights in host DRAM", 0, None, 512e9 / 8, 10),
]
print("\n(2) Time to first token after scale-up, Llama-3.1-8B BF16 on one H100 (ILLUSTRATIVE parts)")
print(f"    {'setup':38s} {'GPU':>6s} {'image':>7s} {'weights':>8s} {'engine':>7s} {'prefill':>8s} {'total':>8s}")
for name, gpu, ibw, wbw, eng in SETUPS:
    img = IMAGE / ibw if ibw else 0.0
    wts = 16e9 / wbw
    tot = gpu + img + wts + eng + PREFILL
    print(f"    {name:38s} {gpu:5.0f}s {img:6.1f}s {wts:7.2f}s {eng:6.0f}s {PREFILL * 1e3:6.1f}ms {tot:7.1f}s")
print("    illustrative: GPU 300 s (node boot), image 10 GB, engine 180 s uncached compile / 30 s / 10 s.")
print("    Once weights are fast, engine startup is the largest part; the prefill is noise.")

# ------------------------------------------------------------------ (3) ServerlessLLM ratios
print("\n(3) ServerlessLLM (OSDI '24), published numbers and ratios")
for what, a, b in [("OPT-6.7B GSM8K start, Ray Serve vs SLLM", 12.1, 0.8),
                   ("OPT-6.7B GSM8K start, Ray Serve+cache vs SLLM", 8.2, 0.8),
                   ("OPT-30B start, Ray Serve vs SLLM", 213, 7.5),
                   ("OPT-30B start, Ray Serve+cache vs SLLM", 199.2, 7.5)]:
    print(f"    {what:46s} {a:6.1f} s / {b:4.1f} s = {a / b:5.1f}x")
print(f"    100 Gbps what-if: Ray Serve 3.8 s, 'still 4.7 times slower' -> SLLM ~ {3.8 / 4.7:.2f} s")
print(f"    LLaMA-2-70B into 8 GPUs with PyTorch: 84 s; at 8.2x faster that would be ~{84 / 8.2:.0f} s (our division)")
tok_mig = 2000 * 4                 # 2,000 token ids at 4 bytes each
kv_mig = 2000 * 128 * 1024         # Llama-3.1-8B KV at 128 KiB/token
print(f"    live migration, 2,000-token request on the 8B: tokens {tok_mig / 1e3:.0f} KB vs KV {kv_mig / GB:.2f} GB "
      f"({kv_mig / tok_mig:,.0f}x more bytes)")

# ------------------------------------------------------------------ (4) the cost of a bursty day
# ILLUSTRATIVE hourly demand, output tokens/s, hour 0 = midnight.
demand = np.array([0, 0, 0, 0, 0, 300, 1500, 5000, 12000, 18000, 21000, 24000,
                   25000, 23000, 20000, 18000, 16000, 13000, 10000, 7000, 4000, 2000, 800, 0], float)
tokens = demand.sum() * 3600
need = np.ceil(demand / CAP).astype(int)           # replicas to keep every hour under capacity
print(f"\n(4) An ILLUSTRATIVE bursty day: peak {demand.max():,.0f} tok/s, mean {demand.mean():,.0f} tok/s, "
      f"{tokens / 1e9:.2f}B output tokens")
print(f"    replica = one H100 at {CAP:,} tok/s (batch 64, 'our model'), ${PRICE}/GPU-hr")
print(f"    replicas needed per hour: {need.tolist()}")


def policy(min_rep):
    reps = np.maximum(need, min_rep)
    starts = int(np.sum(np.maximum(np.diff(np.concatenate([[reps[-1]], reps])), 0)))
    # hours where traffic arrives while zero replicas were running the hour before -> a user waits
    prev = np.roll(reps, 1)
    user_cold = int(np.sum((prev == 0) & (demand > 0)))
    return reps, starts, user_cold


rows = [("Provision for the peak", np.full(24, need.max()), 0, 0)]
for m in (2, 1, 0):
    reps, starts, uc = policy(m)
    rows.append((f"Autoscale, min replicas {m}" if m else "Scale to zero", reps, starts, uc))
print(f"    {'policy':28s} {'GPU-h':>6s} {'$/day':>8s} {'$/M out':>8s} {'util':>6s} {'starts':>7s} {'users hit cold':>15s}")
for name, reps, starts, uc in rows:
    gh = reps.sum()
    cost = gh * PRICE
    util = tokens / (gh * 3600 * CAP)
    print(f"    {name:28s} {gh:6d} {cost:8.2f} {cost / (tokens / 1e6):8.3f} {util:6.1%} {starts:7d} {uc:15d}")
print(f"    ceiling at 100% busy (facts table, batch 64): ${PRICE / (CAP * 3600 / 1e6):.3f}/M")
print("    'starts' = replica cold starts per day (hour granularity). 'users hit cold' = hours where the first")
print("    request arrives with zero replicas running, so it waits a full cold start.")

# ------------------------------------------------------------------ (5) a burst while a replica starts
print("\n(5) A burst: demand jumps to 18,000 tok/s while 2 replicas are running (12,900 tok/s)")
excess = 18000 - 2 * CAP
for name, gpu, ibw, wbw, eng in SETUPS:
    t = gpu + (IMAGE / ibw if ibw else 0) + 16e9 / wbw + eng
    backlog = excess * t
    drain = backlog / (3 * CAP - 18000)            # spare capacity once the 3rd replica is up
    print(f"    {name:38s} cold start {t:6.1f} s -> backlog {backlog / 1e6:5.2f}M tokens, "
          f"drained {drain:6.0f} s after the new replica is up (spare {3 * CAP - 18000:,} tok/s)")
print("    With one spare warm replica (+$95.76/day) the burst is absorbed with no queue.")

# ------------------------------------------------------------------ (6) one warm replica, light traffic
print("\n(6) One replica kept warm all day: effective $/M output tokens vs average load")
for avg in (50, 200, 1000, 3000, CAP):
    print(f"    {avg:5d} tok/s average -> ${PRICE / (avg * 3600 / 1e6):7.3f}/M   (${PRICE * 24:.2f}/day)")
print("    compare: OpenRouter Llama-3.1-8B $0.04-0.08/M output (DeepInfra FP8, Groq), 2026-09-24")
print(f"    even 100% busy, this BF16 ceiling is {PRICE / (CAP * 3600 / 1e6) / 0.08:.1f}x the $0.08/M API price; "
      f"at 50 tok/s it is {PRICE / (50 * 3600 / 1e6) / 0.08:,.0f}x")

# Try this:
# 1. Set CAP = 13400 (FP8 weights roughly double the decode ceiling) and rerun (4): fewer replicas, lower $/M.
# 2. Change SETUPS' engine startup from 180 to 600 (a cold TensorRT-LLM build, "several minutes") and watch (5).
# 3. Flatten the day (demand = np.full(24, demand.mean())) and see scale to zero stop saving anything.
