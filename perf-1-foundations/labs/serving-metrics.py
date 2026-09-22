# Serving metrics: TTFT, TPOT, end-to-end latency, throughput, percentiles, goodput.
#
# Part 1: the floors for one request on the running example
#         (Llama-3.1-8B BF16, one H100 SXM, 1,000-token prompt, 200 output tokens, batch 1).
# Part 2: the provider view. Per-user vs total tokens/s as the decode batch grows,
#         using the Scaling Book's general step-time formula
#           step = B * context * KV bytes / W + max(2 * B * N / C, param bytes / W)
#         and where a TPOT SLO cuts.
# Part 3: percentiles and SLO attainment on a SYNTHETIC list of request latencies (seeded).
#
# Everything here is a floor (100% of peak bandwidth / FLOPs). Real engines are slower.
# Run: python3 labs/serving-metrics.py   (stdlib + numpy, well under a second)
import numpy as np

# ---- running example (perf-1-foundations/_facts.md) ----
N_PARAMS = 8.03e9           # parameters
PARAM_BYTES = 16.06e9       # BF16 weights
KV_PER_TOKEN = 131072       # bytes: 2 (K,V) * 32 layers * 8 KV heads * 128 dims * 2 bytes
BW = 3.35e12                # H100 SXM HBM3, bytes/s
FLOPS = 989e12              # H100 SXM dense BF16
HBM = 80e9                  # bytes
PROMPT, OUT = 1000, 200

SLO_TPOT_MS = 10.0          # illustrative TPOT target (= 100 tokens/s per user)
AVG_CTX = PROMPT + OUT // 2  # average tokens in a sequence's KV cache during decode

def kv_s(batch, ctx):
    """Attention: read every sequence's KV cache (always memory-bound)."""
    return batch * ctx * KV_PER_TOKEN / BW

def compute_s(batch):
    """Matmul FLOPs for one decode step (2 FLOPs per param per token), attention FLOPs ignored."""
    return 2 * N_PARAMS * batch / FLOPS

def step_s(batch, ctx):
    """General step time: KV reads + the slower of weight math and weight reads."""
    return kv_s(batch, ctx) + max(compute_s(batch), PARAM_BYTES / BW)

print("== Part 1: one request, batch 1, floors ==")
ttft = 2 * N_PARAMS * PROMPT / FLOPS
weights_read = PARAM_BYTES / BW
print(f"TTFT floor (prefill compute)  2*{N_PARAMS:.3g}*{PROMPT}/{FLOPS:.3g} = {ttft*1e3:.1f} ms")
print(f"  (prefill memory floor, weights read once: {weights_read*1e3:.2f} ms -> compute is the floor)")
tpot_weights = PARAM_BYTES / BW
print(f"TPOT floor, weights only       {PARAM_BYTES/1e9:.2f} GB / {BW/1e12:.2f} TB/s = {tpot_weights*1e3:.2f} ms")
decode_steps = [step_s(1, PROMPT + i) for i in range(1, OUT)]   # tokens 2..200
decode = sum(decode_steps)
tpot = decode / (OUT - 1)
e2e = ttft + decode
print(f"TPOT floor incl. KV (avg over {OUT-1} steps, ctx {PROMPT+1}..{PROMPT+OUT-1}) = {tpot*1e3:.2f} ms")
print(f"E2E floor = TTFT + {OUT-1} x TPOT = {ttft*1e3:.1f} + {OUT-1} x {tpot*1e3:.2f} = {e2e*1e3:.0f} ms")
print(f"  TTFT share of E2E: {ttft/e2e*100:.1f}%   decode share: {decode/e2e*100:.1f}%")
print(f"  per-user speed: {1/tpot:.0f} tok/s")

print("\n== Part 2: provider view, decode batch B, avg context", AVG_CTX, "tokens ==")
kv_seq = AVG_CTX * KV_PER_TOKEN
b_mem = int((HBM - PARAM_BYTES) // kv_seq)
b_slo = int((SLO_TPOT_MS / 1e3 * BW - PARAM_BYTES) // kv_seq)   # valid while B < B_crit
b_crit = PARAM_BYTES * FLOPS / (2 * N_PARAMS * BW)
print(f"KV per sequence: {AVG_CTX} x {KV_PER_TOKEN} B = {kv_seq/1e6:.1f} MB")
print(f"memory limit: ({HBM/1e9:.0f} - {PARAM_BYTES/1e9:.2f}) GB / {kv_seq/1e6:.1f} MB = {b_mem} sequences")
print(f"TPOT <= {SLO_TPOT_MS:g} ms needs B <= ({SLO_TPOT_MS:g}e-3*{BW:.3g} - {PARAM_BYTES:.4g}) / {kv_seq:.4g} = {b_slo}")
print(f"B_crit (weight math = weight reads): {PARAM_BYTES/1e9:.2f}e9 x {FLOPS:.3g} / (2 x {N_PARAMS:.3g} x {BW:.3g}) = {b_crit:.0f}")
print(f"{'B':>5} {'KV GB':>7} {'KV ms':>7} {'math ms':>8} {'wts ms':>7} {'step ms':>8} {'tok/s/user':>11} {'tok/s total':>12} {'meets SLO':>10} {'goodput tok/s':>14}")
for b in sorted({1, 8, 64, b_slo, 128, 256, b_mem}):
    t = step_s(b, AVG_CTX)
    ok = t * 1e3 <= SLO_TPOT_MS
    print(f"{b:>5} {b*kv_seq/1e9:>7.2f} {kv_s(b, AVG_CTX)*1e3:>7.2f} {compute_s(b)*1e3:>8.2f} {PARAM_BYTES/BW*1e3:>7.2f} {t*1e3:>8.2f} "
          f"{1/t:>11.1f} {b/t:>12.0f} {'yes' if ok else 'NO':>10} {b/t if ok else 0:>14.0f}")
print(f"longest step, B={b_mem} (HBM full): {step_s(b_mem, AVG_CTX)*1e3:.2f} ms "
      f"-> a TPOT SLO looser than that never binds on one H100 (memory runs out first)")

print("\n== Part 3: percentiles and SLO attainment, SYNTHETIC requests (seed 0) ==")
rng = np.random.default_rng(0)
n = 1000
queue = rng.exponential(0.030, n)                        # typical queueing, mean 30 ms
stall = rng.random(n) < 0.03                             # 3% get stuck behind a burst
queue[stall] += rng.uniform(0.3, 1.5, stall.sum())
ttft_s = ttft + queue
tpot_s = 7.5e-3 * rng.lognormal(0.0, 0.2, n)             # busy server: ~7.5 ms/token, jittery
e2e_s = ttft_s + (OUT - 1) * tpot_s
SLO_TTFT = 0.25
p = lambda a, q: np.percentile(a, q)
print(f"{'metric':<10} {'mean':>8} {'P50':>8} {'P90':>8} {'P95':>8} {'P99':>8} {'max':>8}")
for name, a, u in [("TTFT ms", ttft_s, 1e3), ("TPOT ms", tpot_s, 1e3), ("E2E s", e2e_s, 1)]:
    print(f"{name:<10} {a.mean()*u:>8.2f} " + " ".join(f"{p(a, q)*u:>8.2f}" for q in (50, 90, 95, 99)) + f" {a.max()*u:>8.2f}")
meet = (ttft_s <= SLO_TTFT) & (tpot_s * 1e3 <= SLO_TPOT_MS)
print(f"SLO: TTFT <= {SLO_TTFT*1e3:.0f} ms AND TPOT <= {SLO_TPOT_MS:g} ms")
print(f"  miss TTFT: {(ttft_s > SLO_TTFT).sum()}   miss TPOT: {(tpot_s*1e3 > SLO_TPOT_MS).sum()}   "
      f"meet both: {meet.sum()} / {n} = {meet.mean()*100:.1f}% SLO attainment")
print(f"  if the server delivers X tok/s in total, goodput counts only tokens from the {meet.mean()*100:.1f}% that met the SLO")

print("\n== Reader's measured spread (llm-latency/_facts.md, 20 identical requests) ==")
print(f"p50 4.66 s, p95 6.54 s, max 7.14 s -> p95/p50 = {6.54/4.66:.2f}x, max/p50 = {7.14/4.66:.2f}x")

# Try this:
# 1. Set SLO_TPOT_MS = 50 (Etalon's medium target): the SLO never binds, memory limits B instead.
#    (25 ms binds only at B ~ 421, above B_crit, where b_slo's memory-only formula no longer applies.)
# 2. Set PROMPT = 8000: KV per sequence grows 7x, the SLO batch limit collapses and TTFT grows 8x.
# 3. Raise the stall rate from 0.03 to 0.10: P50 barely moves, P99 and SLO attainment get much worse.
