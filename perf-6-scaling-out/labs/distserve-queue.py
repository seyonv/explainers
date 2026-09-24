"""DistServe I (sections 1-3): goodput, the Fig 1 arithmetic, and the prefill queueing model.

Card: perf-6-scaling-out/distserve-interference.html
Runs on a laptop CPU: stdlib + numpy, about 5-15 s.   python3 labs/distserve-queue.py

Six parts:
  A. DistServe Fig 1 arithmetic (13B model, one A100; the paper's numbers).
  B. DistServe Eq 1-3: average TTFT of a prefill-only instance modelled as an M/D/1 queue
       Eq 1  one GPU:                    D + R D^2 / (2 (1 - R D))
       Eq 2  2-way inter-op (2 stages):  D + R D^2 / (4 (2 - R D))
       Eq 3  2-way intra-op, speedup K:  D/K + R D^2 / (2 K (K - R D))
     with D = 1 (time in units of one prefill), K = 1.5, and the crossover rate where
     inter-op starts to win, found by bisection.
  C. A Monte Carlo check of Eq 1-3: simulate the three queues request by request
     (Poisson arrivals, deterministic service) and compare the mean TTFT with the formulas.
  D. The running example, OUR model (not a measurement): Llama-3.1-8B BF16 on H100s,
     1,000-token prompts, 100-token outputs, Poisson arrivals. A step-time model:
         prefill = 2 * params * prompt tokens / 989 TFLOP/s          (the 16.2 ms floor)
         decode step = (16.06 GB weights + batch KV) / 3.35 TB/s      (4.8 ms at batch 1)
     Colocated = one GPU, prefill-first continuous batching (vLLM-style of the paper's era):
     whenever a prompt is waiting it runs alone and every running decode waits for it.
     Disaggregated = prefill-only GPUs (FCFS, one prompt at a time: the M/D/1 queue of B)
     and decode-only GPUs (continuous batching, no prefills ever). KV transfer is ignored here
     (the placement card covers it). SLOs are ours: P90 TTFT <= 100 ms, P90 TPOT <= 10 ms.
     Goodput = highest rate on a 0.5 req/s grid where >= 90% of requests meet BOTH SLOs.
  E. Chunked prefill's O(N^2) KV re-reads (DistServe 2.3), in bytes and ms for the running example.
  F. Part D again with a looser 20 ms TPOT SLO.
"""
import numpy as np

# ------------------------------------------------------------------ A. Fig 1
print("A. DistServe Fig 1 (13B, one A100-80GB, input 512 / output 64, paper's numbers)")
colo, pre_only, dec_only = 1.6, 5.6, 10.0
print(f"   colocated goodput per GPU            {colo:.1f} rps")
print(f"   2 prefill GPUs can take              2 x {pre_only} = {2 * pre_only:.1f} rps (>= {dec_only:.0f})")
print(f"   1 decode GPU can take                {dec_only:.0f} rps  -> system = min(11.2, 10) = 10 rps")
per_gpu = dec_only / 3
print(f"   per GPU                              10 / 3 = {per_gpu:.2f} rps")
print(f"   gain                                 {per_gpu:.3f} / {colo} = {per_gpu / colo:.2f}x  (paper: 2.1x)")
print()


# ------------------------------------------------------------------ B. Eq 1-3
def eq1(x, D=1.0):            # x = R*D
    return D + x * D / (2 * (1 - x))


def eq2(x, D=1.0):
    return D + x * D / (4 * (2 - x))


def eq3(x, K, D=1.0):
    return D / K + x * D / (2 * K * (K - x))


def crossover(K):
    lo, hi = 1e-9, min(2.0, K) - 1e-9
    f = lambda x: eq3(x, K) - eq2(x)
    if f(lo) > 0:
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return lo


K = 1.5
print("B. Average TTFT from Eq 1-3, D = 1, K = 1.5 (units of one single-GPU prefill)")
print("   RD     Eq1 one GPU   Eq2 inter-op   Eq3 intra-op   winner")
for x in (0.1, 0.25, 0.5, 0.75, 0.9, 0.94, 1.0, 1.2, 1.4):
    e1 = f"{eq1(x):11.3f}" if x < 1 else "   unstable"
    e2, e3 = eq2(x), eq3(x, K)
    w = "intra" if e3 < e2 else "inter"
    print(f"   {x:4.2f}  {e1}   {e2:12.3f}   {e3:12.3f}   {w}")
x0 = crossover(K)
print(f"   crossover RD for K = 1.5: {x0:.4f}  (TTFT there {eq2(x0):.3f} D)")
for k in (1.5, 1.6, 1.7, 1.8, 1.9):
    print(f"   K = {k}: crossover RD = {crossover(k):.3f}")
print("   capacity (queue stays finite): one GPU RD < 1, inter-op RD < 2, intra-op RD < K")
print("   worked RD = 0.5: Eq2 = 1 + 0.5/(4*1.5) = %.3f   Eq3 = 1/1.5 + 0.5/(2*1.5*1.0) = %.3f"
      % (eq2(0.5), eq3(0.5, K)))
print("   worked RD = 0.9: Eq2 = 1 + 0.9/(4*1.1) = %.3f   Eq3 = 1/1.5 + 0.9/(2*1.5*0.6) = %.3f"
      % (eq2(0.9), eq3(0.9, K)))
print("   worked RD = 1.2: Eq2 = 1 + 1.2/(4*0.8) = %.3f   Eq3 = 1/1.5 + 1.2/(2*1.5*0.3) = %.3f"
      % (eq2(1.2), eq3(1.2, K)))
print()


# ------------------------------------------------------------------ C. Monte Carlo check
def fcfs_waits(arrivals, service):
    """Lindley recursion: FCFS single server, deterministic service time."""
    w = np.empty_like(arrivals)
    free = 0.0
    for i, a in enumerate(arrivals):
        start = a if a > free else free
        w[i] = start - a
        free = start + service
    return w


rng = np.random.default_rng(0)
N = 200_000
gaps = rng.exponential(1.0, N)
print("C. Monte Carlo check (200,000 Poisson arrivals, deterministic service, D = 1, K = 1.5)")
print("   RD    one GPU sim/Eq1     inter-op sim/Eq2     intra-op sim/Eq3")
for x in (0.5, 0.9):
    arr = np.cumsum(gaps / x)                   # rate R = x when D = 1
    one = (fcfs_waits(arr, 1.0) + 1.0).mean()
    # 2 pipeline stages of D/2 each: stage 2 never waits (same deterministic service),
    # so TTFT = wait at stage 1 + D.
    inter = (fcfs_waits(arr, 0.5) + 1.0).mean()
    intra = (fcfs_waits(arr, 1.0 / K) + 1.0 / K).mean()
    print(f"   {x:3.1f}   {one:6.3f} / {eq1(x):6.3f}     {inter:6.3f} / {eq2(x):6.3f}      {intra:6.3f} / {eq3(x, K):6.3f}")
print()

# ------------------------------------------------------------------ D. running example
PARAMS, W_BYTES, C, BW, KVB = 8.03e9, 16.06e9, 989e12, 3.35e12, 131072
PROMPT, OUT = 1000, 100
TTFT_SLO, TPOT_SLO = 0.100, 0.010
PREFILL = 2 * PARAMS * PROMPT / C


def decode_step(ctx_sum, batch):
    return max(2 * PARAMS * batch / C, W_BYTES / BW) + ctx_sum * KVB / BW


print("D. Running example, OUR model: Llama-3.1-8B BF16, H100, 1,000-token prompt, 100-token output")
print(f"   prefill of 1,000 tokens     = 2 x 8.03e9 x 1000 / 989e12 = {PREFILL * 1e3:.2f} ms")
print(f"   decode step, batch 1        = 16.06e9 / 3.35e12 = {W_BYTES / BW * 1e3:.2f} ms (+ KV reads)")
print(f"   one prefill = {PREFILL / (W_BYTES / BW):.1f} decode steps of stall for every running request")
print(f"   Eq 1 with D = {PREFILL * 1e3:.2f} ms: capacity 1/D = {1 / PREFILL:.1f} req/s;"
      f" K = 1.5 crossover at R = {x0 / PREFILL:.1f} req/s (K is illustrative)")

base_gaps = np.random.default_rng(1).exponential(1.0, 1500)


def sim_colocated(rate):
    arr = np.cumsum(base_gaps / rate)
    n = len(arr)
    first, last, t, i = np.zeros(n), np.zeros(n), 0.0, 0
    waiting, running = [], []          # running: [req, tokens_done]
    done = 0
    while done < n:
        while i < n and arr[i] <= t:
            waiting.append(i); i += 1
        if waiting:                                    # prefill-first
            r = waiting.pop(0)
            t += PREFILL
            first[r] = t
            running.append([r, 1])
        elif running:
            ctx = sum(PROMPT + k for _, k in running)
            t += decode_step(ctx, len(running))
            keep = []
            for item in running:
                item[1] += 1
                if item[1] == OUT:
                    last[item[0]] = t; done += 1
                else:
                    keep.append(item)
            running = keep
        else:
            t = arr[i]
    ttft = first - arr
    tpot = (last - first) / (OUT - 1)
    return ttft, tpot


def sim_prefill_only(rate):
    arr = np.cumsum(base_gaps / rate)
    ttft = fcfs_waits(arr, PREFILL) + PREFILL
    return ttft, np.zeros_like(ttft)


def sim_decode_only(rate):
    # requests arrive already prefilled (first token made elsewhere), KV transfer ignored
    arr = np.cumsum(base_gaps / rate)
    n = len(arr)
    last, t, i, running, done = np.zeros(n), 0.0, 0, [], 0
    while done < n:
        while i < n and arr[i] <= t:
            running.append([i, 1]); i += 1
        if running:
            ctx = sum(PROMPT + k for _, k in running)
            t += decode_step(ctx, len(running))
            keep = []
            for item in running:
                item[1] += 1
                if item[1] == OUT:
                    last[item[0]] = t; done += 1
                else:
                    keep.append(item)
            running = keep
        else:
            t = arr[i]
    tpot = (last - arr) / (OUT - 1)
    return np.zeros_like(tpot), tpot


def attainment(ttft, tpot):
    return np.mean((ttft <= TTFT_SLO) & (tpot <= TPOT_SLO))


def goodput(sim, top=256.0):
    """Bisection on a 0.5 req/s grid (assumes attainment falls as the rate rises)."""
    lo, hi = 0, int(top * 2)           # in half-req/s units; lo passes, hi fails
    while hi - lo > 1:
        mid = (lo + hi) // 2
        tt, tp = sim(mid / 2)
        if attainment(tt, tp) >= 0.9:
            lo = mid
        else:
            hi = mid
    return lo / 2


print("   SLOs (ours): TTFT <= 100 ms and TPOT <= 10 ms for >= 90% of requests")
print("   rate    colocated: P90 TTFT  P90 TPOT  attain   GPU time spent in prefill = R x 16.24 ms")
for r in (5, 10, 15, 20, 25, 30):
    tt, tp = sim_colocated(r)
    print(f"   {r:4.0f}    {np.percentile(tt, 90) * 1e3:14.1f} ms {np.percentile(tp, 90) * 1e3:6.2f} ms  {attainment(tt, tp) * 100:5.1f}%"
          f"   {r * PREFILL * 1e3:5.0f} ms per s = {r * PREFILL * 100:4.1f}%")
g_col = goodput(sim_colocated)
g_pre = goodput(sim_prefill_only)
g_dec = goodput(sim_decode_only)
print(f"   goodput colocated (1 GPU)    {g_col:.1f} req/s per GPU")
print(f"   goodput prefill-only GPU     {g_pre:.1f} req/s")
print(f"   goodput decode-only GPU      {g_dec:.1f} req/s")
tt, tp = sim_decode_only(g_dec)
print(f"     decode-only at {g_dec:.1f} req/s: P90 TPOT {np.percentile(tp, 90) * 1e3:.2f} ms,"
      f" mean batch (Little's law) {g_dec * np.mean(tp * (OUT - 1)):.0f}")
best = None
for p in range(1, 6):
    for d in range(1, 6):
        sys_rate = min(p * g_pre, d * g_dec)
        per = sys_rate / (p + d)
        if best is None or per > best[0] + 1e-9:
            best = (per, p, d, sys_rate)
per, p, d, sys_rate = best
print(f"   best split up to 5P5D: {p}P{d}D -> min({p} x {g_pre:.1f}, {d} x {g_dec:.1f}) = {sys_rate:.1f} req/s"
      f" over {p + d} GPUs = {per:.1f} req/s per GPU")
print(f"   gain vs colocated: {per:.1f} / {g_col:.1f} = {per / g_col:.2f}x   (our model, not a measurement)")

print()

# ------------------------------------------------------------------ E. chunked prefill's KV re-reads
print("E. Chunked prefill re-reads earlier chunks' KV (DistServe 2.3: N + (N-1) + ... + 1 chunk loads)")
for prompt, chunk in ((1000, 250), (32768, 512)):
    n = -(-prompt // chunk)
    loads = n * (n + 1) // 2
    kv_bytes = loads * chunk * KVB
    compute = 2 * PARAMS * prompt / C
    print(f"   {prompt:,}-token prompt in {n} chunks of {chunk}: {loads} chunk loads vs {n} unchunked;"
          f" {kv_bytes / 1e9:.2f} GB = {kv_bytes / BW * 1e3:.2f} ms of HBM reads"
          f" vs {compute * 1e3:.0f} ms of linear-layer prefill math ({kv_bytes / BW / compute * 100:.1f}%)")
print()

# ------------------------------------------------------------------ F. sensitivity to the TPOT SLO
print("F. Same model with a looser TPOT SLO of 20 ms")
TPOT_SLO = 0.020
g_col2, g_dec2 = goodput(sim_colocated), goodput(sim_decode_only)
best2 = max(((min(p * g_pre, d * g_dec2) / (p + d), p, d) for p in range(1, 6) for d in range(1, 6)))
print(f"   colocated {g_col2:.1f} req/s per GPU; decode-only {g_dec2:.1f}; best {best2[1]}P{best2[2]}D"
      f" {best2[0]:.1f} per GPU -> gain {best2[0] / g_col2:.2f}x")

# Try this:
#   1. Set K = 1.9 in part B: the crossover moves past RD = 1.6, so intra-op wins almost everywhere.
#   2. Set TPOT_SLO = 0.020 (a looser 20 ms): colocation's goodput jumps and the gain shrinks,
#      which is DistServe's point that the win depends on how tight the SLOs are.
#   3. Set PROMPT = 4000: one prefill now stalls decodes for ~65 ms, and colocation collapses.
