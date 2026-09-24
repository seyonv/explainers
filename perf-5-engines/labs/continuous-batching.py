"""Continuous batching lab: Orca (Yu et al., OSDI '22) sections 1-3.

Run: python3 labs/continuous-batching.py   (stdlib + numpy, CPU, a few seconds)

Simulates one Llama-3.1-8B (BF16) server on one H100 SXM under two schedulers with the
same maximum batch size B:
  (a) request-level (static) batching, FasterTransformer + Triton style: when the engine
      is idle, take up to B waiting requests, run the prefill, then keep decoding the whole
      batch until its LONGEST request finishes. Finished rows keep computing (wasted slots)
      and nobody gets an answer until the batch ends.
  (b) iteration-level scheduling (Orca / continuous batching) with selective batching:
      before every iteration, admit waiting requests into free slots (their prefill rides
      in the same iteration), run ONE iteration, return finished requests at once.

Iteration time is the step-time formula from course 2 (ideal peaks, so every time is a
floor, not a measurement), with every slot charged a fixed 2,048-token context:
  t_iter = n_slots * CTX * kv_per_token / HBM  +  max( 2 * params * n_tokens / FLOPs , weights / HBM )
  n_slots  = decode rows in the iteration (static: all rows of the batch, finished or not)
  n_tokens = decode rows + new prompt tokens being prefilled this iteration

Requests are SYNTHETIC (seeded): Poisson arrivals, every prompt 512 tokens, output lengths
uniform on 1..1024 tokens. Not a real trace.
"""
import numpy as np

# ---- constants from _facts.md (Llama-3.1-8B BF16 on one H100 SXM) ----
PARAMS = 8.03e9
W_BYTES = 16.06e9
KV_TOK = 131072              # 2 * L * K * H * 2 bytes = 128 KiB per token
FLOPS = 989e12
HBM = 3.35e12
CTX = 2048                   # fixed context charged to every decode slot (upper bound here)

B_MAX = 64                   # max batch size, same for both schedulers
PROMPT = 512                 # synthetic: every prompt 512 tokens (so static batches need no padding)
OUT_MAX = 1024               # synthetic: output lengths uniform on 1..OUT_MAX
RATES = (5.0, 8.0)           # requests/s, Poisson: one load static batching can keep up with, one it can't
N_REQ = 3000
SEED = 0


def t_iter(n_slots, n_tokens):
    kv = n_slots * CTX * KV_TOK / HBM
    return kv + max(2 * PARAMS * n_tokens / FLOPS, W_BYTES / HBM)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def static_sim(arr, out):
    """Request-level batching. Returns per-request TTFT, E2E, plus slot and token counts."""
    n = len(arr)
    ttft, e2e = np.zeros(n), np.zeros(n)
    t, i = 0.0, 0
    used = wasted = 0              # decode slot-iterations: useful vs spent on finished rows
    iters = rows = 0
    while i < n:
        t = max(t, arr[i])
        j = i
        while j < n and j - i < B_MAX and arr[j] <= t:
            j += 1
        batch = range(i, j)
        b = j - i
        t += t_iter(0, b * PROMPT)                 # initiation phase: all prompts, first token each
        iters += 1; rows += b
        for r in batch:
            ttft[r] = t - arr[r]                   # first token exists now (if the engine could stream)
        longest = max(out[r] for r in batch)
        for k in range(1, longest):                # increment phase: tokens 2..longest
            alive = sum(1 for r in batch if out[r] > k)
            used += alive; wasted += b - alive
            t += t_iter(b, b)                      # every row computes, finished or not
            iters += 1; rows += b
        for r in batch:
            e2e[r] = t - arr[r]                    # answers return only when the batch ends
        i = j
    return ttft, e2e, used, wasted, t, iters, rows


def iteration_sim(arr, out):
    """Iteration-level scheduling with selective batching (Orca S1 + S2)."""
    n = len(arr)
    ttft, e2e = np.zeros(n), np.zeros(n)
    t, nxt, done = 0.0, 0, 0
    running = {}                   # id -> tokens still to generate
    queue = []
    used = iters = rows = 0
    while done < n:
        while nxt < n and arr[nxt] <= t:
            queue.append(nxt); nxt += 1
        if not running and not queue:
            t = arr[nxt]; continue
        new = []
        while queue and len(running) + len(new) < B_MAX:   # FCFS into free slots
            new.append(queue.pop(0))
        n_dec = len(running)
        t += t_iter(n_dec, n_dec + len(new) * PROMPT)      # decodes and prefills in ONE iteration
        iters += 1; rows += n_dec + len(new)
        used += n_dec
        for r in list(running):
            running[r] -= 1
            if running[r] == 0:
                del running[r]; e2e[r] = t - arr[r]; done += 1
        for r in new:
            ttft[r] = t - arr[r]
            if out[r] == 1:
                e2e[r] = t - arr[r]; done += 1
            else:
                running[r] = out[r] - 1
    return ttft, e2e, used, 0, t, iters, rows


def report(name, res, out):
    ttft, e2e, used, wasted, t_end, iters, rows = res
    toks = int(out.sum())
    slot_total = used + wasted
    print(f"{name}")
    print(f"  TTFT            mean {ttft.mean()*1e3:8.0f} ms   P99 {np.percentile(ttft, 99)*1e3:8.0f} ms")
    print(f"  answer returned mean {e2e.mean():8.2f} s    P99 {np.percentile(e2e, 99):8.2f} s")
    print(f"  total throughput     {toks / t_end:8.0f} tok/s  ({toks:,} tokens in {t_end:.1f} s)")
    print(f"  mean rows per iteration {rows / iters:6.1f}   iterations {iters:,}")
    print(f"  decode slots wasted on finished requests {100 * wasted / max(slot_total, 1):5.1f}%"
          f"  ({wasted:,} of {slot_total:,})")
    return dict(ttft_mean=ttft.mean(), ttft_p99=np.percentile(ttft, 99), thr=toks / t_end)


# ---------------------------------------------------------------------------
section("1. Why the batch must stay full: step time at a fixed 2,048-token context")
for b in (1, 8, 32, 64, 238):
    s = t_iter(b, b)
    print(f"  B = {b:3d}: step {s*1e3:6.2f} ms -> {b/s:7.0f} tok/s total, {1/s:5.0f} tok/s per user")
print(f"  At B = 1 each step reads {W_BYTES/1e9:.2f} GB of weights to make ONE token.")
print(f"  Weights alone: {W_BYTES/1e9:.2f} GB / {HBM/1e12:.2f} TB/s = {W_BYTES/HBM*1e3:.2f} ms per step")
print(f"  Prefill of one {PROMPT}-token prompt alone: max(2*8.03e9*{PROMPT}/989e12, weights/BW) = "
      f"{t_iter(0, PROMPT)*1e3:.2f} ms")

# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
gaps = rng.exponential(1.0, N_REQ)           # unit-rate Poisson gaps, rescaled per load below
out = rng.integers(1, OUT_MAX + 1, N_REQ)

for RATE in RATES:
    arr = np.cumsum(gaps) / RATE
    section(f"2. Poisson load: {N_REQ} synthetic requests at {RATE:g} req/s, B_max = {B_MAX}, "
            f"prompt {PROMPT}, output U[1,{OUT_MAX}] (mean {out.mean():.0f})")
    print(f"  offered load: {RATE:g} req/s x {out.mean():.0f} tokens = {RATE*out.mean():,.0f} tok/s\n")
    a = report("(a) request-level (static) batching", static_sim(arr, out), out)
    b = report("(b) iteration-level scheduling + selective batching", iteration_sim(arr, out), out)
    print(f"\n  TTFT mean  {a['ttft_mean']*1e3:,.0f} -> {b['ttft_mean']*1e3:,.0f} ms "
          f"({a['ttft_mean']/b['ttft_mean']:.0f}x lower); P99 {a['ttft_p99']*1e3:,.0f} -> {b['ttft_p99']*1e3:,.0f} ms")

# ---------------------------------------------------------------------------
section(f"3. Saturated: all {N_REQ} requests waiting at t = 0 (peak throughput, B_max = {B_MAX})")
arr0 = np.zeros(N_REQ)
a0 = static_sim(arr0, out)
b0 = iteration_sim(arr0, out)
ta, tb = out.sum() / a0[4], out.sum() / b0[4]
print(f"  (a) static         {ta:7.0f} tok/s   rows/iter {a0[6]/a0[5]:5.1f}   wasted slots "
      f"{100*a0[3]/(a0[2]+a0[3]):4.1f}%")
print(f"  (b) iteration-level{tb:7.0f} tok/s   rows/iter {b0[6]/b0[5]:5.1f}   wasted slots 0.0%")
print(f"  ratio {tb/ta:.2f}x")

# ---------------------------------------------------------------------------
section("4. Selective batching, Fig 5 of the paper: 7 tokens from 4 requests")
reqs = {"x1": ("increment", 1, 3), "x2": ("increment", 1, 1),
        "x3": ("initiation", 2, 0), "x4": ("initiation", 3, 0)}
total = sum(v[1] for v in reqs.values())
print(f"  flattened input to QKV Linear / Attn Out Linear / MLP: [{total}, H]  (no batch dimension)")
for k, (ph, n_new, n_old) in reqs.items():
    print(f"  Attention for {k}: {ph:10s} query [{n_new}, H] against K/V of {n_old + n_new} tokens"
          f" ({n_old} from the K/V manager)")

# Try this:
#   RATES = (6.0,)     -> static batching is near its limit (~6.1 req/s): watch its P99 TTFT explode
#   OUT_MAX = 128      -> Orca's own eval range (max_gen U[1,128]); shorter tails, less static waste
#   B_MAX = 8          -> small batches: less waste per batch, but far lower peak throughput for both
