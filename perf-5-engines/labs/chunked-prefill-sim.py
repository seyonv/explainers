"""Chunked prefill and stall-free batching: a discrete-time scheduler simulator.

Card: perf-5-engines/sarathi-chunked-prefill.html (Sarathi-Serve II).
Runs on a laptop CPU: stdlib + numpy, about 10-25 s.  python3 labs/chunked-prefill-sim.py

What it models (all numbers are OUR model, not a measurement):
  * Llama-3.1-8B BF16 on one H100 SXM, with the course's step-time model:
        step = KV bytes / BW + max( FLOPs / C , weight bytes / BW )
    FLOPs = 2 * params * tokens in the step + attention-score FLOPs,
    KV bytes = every token's context read by attention + the new KV written.
    No kernel-launch or CPU overhead (FIXED_MS = 0), so the model flatters chunking.
  * Synthetic Poisson arrivals (seeded). Prompt and output lengths are lognormal fits
    to Sarathi-Serve Table 2 (openchat_sharegpt4): prompt median 1730 / P90 5696,
    output median 415 / P90 834. Requests with prompt + output > 8192 are dropped,
    as the paper does for this dataset.
  * Five schedulers:
      vllm-v0       prefill-prioritised: whole prompts in prefill-only iterations
                    (up to 8192 prompt tokens), otherwise one decode iteration
      orca-hybrid   every iteration = all running decodes + whole new prompts
                    (up to 8192 prompt tokens): hybrid, but no chunking
      chunked-only  prompts cut into tau-token chunks, but chunks never share an
                    iteration with decodes; prefill and decode iterations alternate
                    (our reading of Sarathi-Serve Table 4's "chunked-prefills-only")
      sarathi-512   stall-free batching (Sarathi-Serve Algorithm 3), token budget 512:
                    decodes first, then the ongoing prompt's next chunk, then new prompts
      sarathi-2048  same, budget 2048
  * Admission: a request is admitted only if its full prompt + output KV fits in the
    63.9 GB left after weights (a reservation the real engines don't need; it keeps the
    sim simple) and fewer than 256 sequences are running.
  * Metrics, as in Sarathi-Serve 5: median TTFT, P99 TBT (every gap between two tokens
    of one request), throughput, and capacity = the highest Poisson rate whose P99 TBT
    meets the SLO while the median scheduling delay stays <= 2 s (the paper's rule).
    The 50 ms and 100 ms SLOs are ours.
"""
import numpy as np

# ---- the running example: Llama-3.1-8B BF16 on one H100 SXM (course facts) ----
PARAMS = 8.03e9
W_BYTES = 16.06e9
C = 989e12            # dense BF16 FLOP/s
BW = 3.35e12          # HBM bytes/s
KVB = 131072          # KV bytes per token (2*L*K*H*2)
L, NH, HD = 32, 32, 128
KV_TOKENS = int((80e9 - W_BYTES) / KVB)   # 487,823 tokens of KV fit
MAX_SEQS = 256
MAX_PREFILL = 8192    # whole-prompt schedulers: prompt tokens per iteration
FIXED_MS = 0.0        # per-iteration CPU/launch overhead (none in the model)

# ---- workload: lognormal fit to Sarathi-Serve Table 2, openchat_sharegpt4 ----
Z90 = 1.2815515655446004
P_MED, P_P90 = 1730, 5696
O_MED, O_P90 = 415, 834
P_SIG = np.log(P_P90 / P_MED) / Z90
O_SIG = np.log(O_P90 / O_MED) / Z90
N_REQ = 500
SEED = 0


def step_ms(n_tokens, kv_tokens, attn_flops):
    """One iteration of the whole batch, in ms."""
    compute = (2 * PARAMS * n_tokens + attn_flops) / C
    return 1e3 * (kv_tokens * KVB / BW + max(compute, W_BYTES / BW)) + FIXED_MS


def attn_flops(new, past):
    """QK^T and PV for `new` causal tokens after `past` cached tokens."""
    return 4 * L * NH * HD * new * (past + (new + 1) / 2)


def workload(rate, n=N_REQ, seed=SEED):
    rng = np.random.default_rng(seed)
    prompts, outs = [], []
    while len(prompts) < n:
        p = int(round(rng.lognormal(np.log(P_MED), P_SIG)))
        o = int(round(rng.lognormal(np.log(O_MED), O_SIG)))
        if p >= 1 and o >= 1 and p + o <= 8192:
            prompts.append(p)
            outs.append(o)
    gaps = np.random.default_rng(seed + 1).exponential(1.0 / rate, n)
    return np.cumsum(gaps) * 1e3, np.array(prompts), np.array(outs)  # arrivals in ms


def simulate(policy, rate, tau=None, seed=SEED):
    arr, P, O = workload(rate, seed=seed)
    n = len(arr)
    first_sched = np.full(n, np.nan)
    ttft = np.full(n, np.nan)
    done_prompt = np.zeros(n, dtype=np.int64)   # prompt tokens already prefilled
    last_tok = np.zeros(n)                      # time of the latest token
    emitted = np.zeros(n, dtype=np.int64)       # output tokens produced
    gaps, gap_end = [], []
    waiting = list(range(n))                    # FCFS, not yet arrived or queued
    prefilling = []                             # admitted, prompt not finished
    decoding = []                               # prompt finished, generating
    kv_used = 0
    t = 0.0
    wi = 0                                      # next index in `waiting`
    last_was_prefill = False
    out_total = 0

    def admissible(i):
        return (arr[i] <= t and kv_used + P[i] + O[i] <= KV_TOKENS
                and len(prefilling) + len(decoding) < MAX_SEQS)

    while wi < n or prefilling or decoding:
        if not prefilling and not decoding and wi < n and arr[wi] > t:
            t = arr[wi]                          # idle until the next arrival
        chunks = []                              # (request, tokens this step)
        dec = []
        if policy in ("vllm-v0", "orca-hybrid"):
            budget = MAX_PREFILL
            while wi < n and admissible(wi) and P[wi] <= budget:
                i = waiting[wi]; wi += 1
                kv_used += P[i] + O[i]; first_sched[i] = t
                chunks.append((i, P[i])); budget -= P[i]
            if policy == "orca-hybrid" or not chunks:
                dec = list(decoding)
        elif policy == "chunked-only":
            want_prefill = bool(prefilling) or (wi < n and admissible(wi))
            if want_prefill and (not decoding or not last_was_prefill):
                budget = tau
                for i in prefilling:
                    c = min(P[i] - done_prompt[i], budget)
                    if c > 0:
                        chunks.append((i, c)); budget -= c
                while budget > 0 and wi < n and admissible(wi):
                    i = waiting[wi]; wi += 1
                    kv_used += P[i] + O[i]; first_sched[i] = t
                    prefilling.append(i)
                    c = min(P[i], budget); chunks.append((i, c)); budget -= c
            else:
                dec = list(decoding)
        else:  # stall-free batching, Algorithm 3
            dec = list(decoding)
            budget = tau - len(dec)                         # lines 6-8
            for i in prefilling:                             # lines 9-12
                c = min(P[i] - done_prompt[i], max(budget, 0))
                if c > 0:
                    chunks.append((i, c)); budget -= c
            while budget > 0 and wi < n and admissible(wi):  # lines 13-20
                i = waiting[wi]; wi += 1
                kv_used += P[i] + O[i]; first_sched[i] = t
                prefilling.append(i)
                c = min(P[i], budget); chunks.append((i, c)); budget -= c
        if not chunks and not dec:
            # nothing runnable: jump to the next arrival (or wait for memory)
            if wi < n:
                t = max(t, arr[wi]) + (0.0 if arr[wi] > t else 1.0)
                continue
            break
        # ---- cost of this iteration ----
        idx = np.array(dec, dtype=np.int64)
        ntok = len(dec) + sum(c for _, c in chunks)
        ctx = (P[idx] + emitted[idx]).astype(np.float64)
        kv = ctx.sum() + len(dec)
        fl = attn_flops(1, ctx).sum()
        for i, c in chunks:
            kv += done_prompt[i] + c           # read earlier chunks' KV, write new
            fl += attn_flops(c, done_prompt[i])
        t += step_ms(ntok, kv, fl)
        last_was_prefill = bool(chunks) and not dec
        # ---- outputs of this iteration ----
        if dec:
            gaps.append(t - last_tok[idx]); gap_end.append(np.full(len(dec), t))
            last_tok[idx] = t; emitted[idx] += 1; out_total += len(dec)
        for i, c in chunks:
            if policy in ("vllm-v0", "orca-hybrid"):
                decoding.append(i)
            done_prompt[i] += c
            if done_prompt[i] == P[i]:                   # prompt done: first token
                if i in prefilling:
                    prefilling.remove(i); decoding.append(i)
                ttft[i] = t - arr[i]; last_tok[i] = t
                emitted[i] = 1; out_total += 1
        if decoding:
            d = np.array(decoding, dtype=np.int64)
            fin = emitted[d] >= O[d]
            if fin.any():
                kv_used -= int((P[d[fin]] + O[d[fin]]).sum())
                decoding = d[~fin].tolist()
    gaps = np.concatenate(gaps); gap_end = np.concatenate(gap_end)
    lo, hi = arr[n // 10], arr[-1]                          # steady-state window
    win = (gap_end >= lo) & (gap_end <= hi)
    keep = np.arange(n) >= n // 10
    return dict(
        p50_ttft=float(np.median(ttft[keep])),
        p99_tbt=float(np.percentile(gaps[win], 99)),
        p50_tbt=float(np.median(gaps[win])),
        sched_delay=float(np.median(first_sched[keep] - arr[keep])),
        tput=out_total / (t / 1e3),
        mean_out=float(O.mean()),
    )


_memo = {}


def run(policy, rate, tau):
    key = (policy, round(rate, 6), tau)
    if key not in _memo:
        _memo[key] = simulate(policy, rate, tau)
    return _memo[key]


def capacity(policy, tau, slo_ms, lo=1.0, hi=32.0, iters=7):
    """Highest Poisson rate (req/s) meeting P99 TBT <= slo and median sched delay <= 2 s.
    Bisection on a log scale: 32^(1/128) = 2.7% resolution."""
    def ok(r):
        m = run(policy, r, tau)
        return m["p99_tbt"] <= slo_ms and m["sched_delay"] <= 2000
    if not ok(lo):
        return 0.0
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


POLICIES = [("vllm-v0", None), ("orca-hybrid", None), ("chunked-only", 512),
            ("sarathi-512", 512), ("sarathi-2048", 2048)]

if __name__ == "__main__":
    print("== Model: Llama-3.1-8B BF16, one H100 SXM (course step-time model) ==")
    print(f"KV capacity {KV_TOKENS:,} tokens; decode-only floor {step_ms(1, 0, 0):.2f} ms/step")
    for n in (256, 512, 1024, 1730, 2048, 5696):
        print(f"  prefill-only step of {n:>5} tokens: {step_ms(n, n, attn_flops(n, 0)):6.1f} ms")
    arr, P, O = workload(1.0)
    print(f"\n== Workload (synthetic, lognormal fit, seed {SEED}, {N_REQ} requests) ==")
    print(f"prompt sigma {P_SIG:.3f}, output sigma {O_SIG:.3f}")
    print(f"prompt median {np.median(P):.0f}  P90 {np.percentile(P, 90):.0f}  mean {P.mean():.0f}")
    print(f"output median {np.median(O):.0f}  P90 {np.percentile(O, 90):.0f}  mean {O.mean():.0f}")

    # One decode batch of 64 at 2k context, plus a prefill chunk (cf. Fig 9)
    B, ctx = 64, 2048
    kvb, flb = B * (ctx + 1), attn_flops(1, ctx) * B
    base = step_ms(B, kvb, flb)
    print(f"\n== One iteration: {B} decodes at {ctx} context, plus prefill tokens ==")
    print(f"decode-only: {base:.2f} ms")
    for c in (256, 448, 512, 1024, 2048, 5696):
        s = step_ms(B + c, kvb + c, flb + attn_flops(c, 0))
        print(f"  + {c:>5} prefill tokens: {s:6.2f} ms  ({s / base:4.2f}x decode-only)")

    # One-time profiling, Sarathi-Serve 4.3: the largest budget whose iteration meets the SLO
    for slo in (50, 100):
        best = max(c for c in range(0, 8193, 16)
                   if step_ms(B + c, kvb + c, flb + attn_flops(c, 0)) <= slo)
        print(f"  largest prefill chunk that keeps this iteration <= {slo} ms: "
              f"{best} tokens (budget {B + best} incl. the {B} decodes)")

    # Cost of chunking one prompt on its own (cf. Sarathi-Serve Fig 14)
    print("\n== Chunking overhead in our model: one prompt alone, no decodes ==")
    for n in (1730, 5696):
        whole = step_ms(n, n, attn_flops(n, 0))
        row = f"  {n:>5}-token prompt: whole {whole:6.1f} ms"
        for c in (256, 512, 2048):
            tot, done = 0.0, 0
            while done < n:
                k = min(c, n - done)
                tot += step_ms(k, done + k, attn_flops(k, done))
                done += k
            row += f" | chunks of {c}: {tot:6.1f} ms (+{100 * (tot / whole - 1):4.1f}%)"
        print(row)
    n, c = 5696, 512
    starts = list(range(0, n, c))
    extra = sum(starts)                      # chunk k re-reads the KV of all earlier chunks
    last = n - starts[-1]
    print(f"  {n} in chunks of {c}: {len(starts)} chunks (last one {last} tokens);"
          f" earlier-chunk KV re-read = {extra:,} tokens = {extra * KVB / 1e9:.2f} GB"
          f" = {1e3 * extra * KVB / BW:.2f} ms")
    lc = 1e3 * (2 * PARAMS * last + attn_flops(last, starts[-1])) / C
    print(f"  the {last}-token tail chunk: compute {lc:.2f} ms but pays the "
          f"{1e3 * W_BYTES / BW:.2f} ms weight read")

    RATE = 6.0
    print(f"\n== All schedulers at {RATE} req/s (our pick) ==")
    print(f"{'policy':<14}{'P50 TTFT ms':>12}{'P50 TBT ms':>11}{'P99 TBT ms':>11}"
          f"{'sched delay':>12}{'out tok/s':>10}")
    for pol, tau in POLICIES:
        m = simulate(pol, RATE, tau)
        print(f"{pol:<14}{m['p50_ttft']:12.0f}{m['p50_tbt']:11.1f}{m['p99_tbt']:11.1f}"
              f"{m['sched_delay']:12.0f}{m['tput']:10.0f}")

    print("\n== Capacity (req/s) under a P99 TBT SLO (ours) + median sched delay <= 2 s ==")
    print(f"{'policy':<14}{'no TBT SLO':>11}{'100 ms':>9}{'50 ms':>9}{'tok/s @50ms':>13}")
    caps = {}
    for pol, tau in POLICIES:
        c0 = capacity(pol, tau, 1e9)
        c100 = capacity(pol, tau, 100)
        c50 = capacity(pol, tau, 50)
        caps[pol] = (c0, c100, c50)
        print(f"{pol:<14}{c0:11.2f}{c100:9.2f}{c50:9.2f}{c50 * O.mean():13.0f}")
    for slo, k in ((100, 1), (50, 2)):
        v = caps["vllm-v0"][k]
        best = max(caps["sarathi-512"][k], caps["sarathi-2048"][k])
        r = f"{best / v:.1f}x" if v > 0 else "inf (vLLM-v0 has none)"
        print(f"stall-free (best budget) vs vllm-v0 at {slo} ms: {r}")

# Try this:
# 1. FIXED_MS = 2.0 (a CPU/launch cost per iteration): small budgets now pay for their extra
#    iterations. No-SLO capacity: sarathi-512 12.75 -> 10.83 req/s, sarathi-2048 14.99 -> 13.82.
# 2. RATE = 2.0: stalls are rare at light load (vllm-v0 P99 TBT ~13 ms) and the schedulers look
#    alike; the difference only appears once prefills keep landing on a big decode batch.
# 3. P_MED, P_P90 = 7059, 12985 (arxiv_summarization) and raise the 8192 cap to 16384:
#    longer prompts make whole-prompt schedulers stall far longer.
