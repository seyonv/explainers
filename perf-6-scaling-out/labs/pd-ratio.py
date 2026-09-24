"""pd-ratio lab: DistServe's placement search (Algorithm 1) in miniature, plus KV-transfer cost.

Run: python3 labs/pd-ratio.py        (stdlib + numpy, CPU, ~15 s)

Everything here is OUR model, not DistServe's code or numbers:
  * Llama-3.1-8B BF16 on H100 SXM, step times from course 2's ideal floors
    (989 TFLOPS, 3.35 TB/s; no attention FLOPs in prefill, perfect overlap).
  * TP all-reduce cost: 2 per layer (our inference from Megatron), ring bandwidth
    term over NVLink 450 GB/s per direction + 2(n-1) ring hops of an ILLUSTRATIVE 1 us each.
  * Workloads are synthetic (lognormal lengths); only the MEANS come from DistServe
    Fig 7 (ShareGPT 755.5/200.3, HumanEval 171.3/98.2, LongBench 1738.3/90.7 in/out).
  * SLOs are ILLUSTRATIVE (per workload, shaped like DistServe Table 1), 90% attainment.

Part 1  KV transfer per request: OPT-66B (the paper's check), Llama-3.1-8B, Llama-3.1-70B
Part 2  Alg 1 lite: search TP for the prefill and decode instances separately,
        keep the best goodput per GPU, replicate to meet a target rate
Part 3  end-to-end check of the chosen placement vs colocated replicas
Part 4  Scaling Book's P = 3G balance, redone with our step times
"""
import heapq
import math

import numpy as np

GB, GiB, MiB, KiB = 1e9, 2**30, 2**20, 2**10

# ------------------------------------------------------------------ Part 1
print("== Part 1. KV cache per request and what moving it costs ==")
opt66_tok = 2 * 64 * 9216 * 2           # K and V, 64 layers, hidden 9216, FP16 (MHA)
kv8_tok = 2 * 32 * 8 * 128 * 2          # Llama-3.1-8B GQA: 128 KiB
kv70_tok = 2 * 80 * 8 * 128 * 2         # Llama-3.1-70B GQA: 320 KiB
for name, b in (("OPT-66B", opt66_tok), ("Llama-3.1-8B", kv8_tok), ("Llama-3.1-70B", kv70_tok)):
    print(f"{name:14s} KV/token = {b:,} B = {b / KiB:,.0f} KiB")
req = 512 * opt66_tok
print(f"\nOPT-66B, 512 tokens: {req:,} B = {req / GB:.2f} GB = {req / GiB:.3f} GiB   (paper: '1.13GB')")
print(f"  at 10 req/s: {10 * req / GB:.2f} GB/s = {10 * req * 8 / 1e9:.1f} Gb/s decimal"
      f" = {10 * req * 8 / 2**30:.1f} Gib/s   (paper: '11.3GB' per s, '90Gbps')")
print(f"  on the paper's 25 Gb/s cross-node link: {req / (25e9 / 8) * 1e3:.0f} ms per request;"
      f" on A100 NVLink 300 GB/s per direction: {req / 300e9 * 1e3:.1f} ms")
print(f"  ratio OPT-66B / Llama-3.1-70B per token = {opt66_tok / kv70_tok:.1f}x (MHA vs GQA)")

opt175_tok = 2 * 96 * 12288 * 2         # OPT-175B: 96 layers, hidden 12288, FP16
r175 = 755.5 * opt175_tok               # an average ShareGPT prompt
print(f"\nOPT-175B KV/token = {opt175_tok / MiB:.1f} MiB; average ShareGPT prompt (755.5 tokens) = {r175 / GB:.2f} GB")
print(f"  moving it in 30 ms needs {r175 / 0.030 / GB:.0f} GB/s (our inference);"
      f" over 25 Gb/s it would take {r175 / (25e9 / 8):.2f} s")
print("Alg 2 layout DistServe chose for OPT-175B (Table 3): prefill TP3 x PP3, decode TP4 x PP3")
print(f"  one prefill + one decode copy = 2 x 350 GB = 700 GB > 8 x 80 = 640 GB of one node")
print(f"  per node: one stage of each = 3 + 4 = 7 GPUs <= 8; stage weights 350/3 = {350 / 3:.1f} GB"
      f" -> {350 / 3 / 3:.1f} GB per prefill GPU, {350 / 3 / 4:.1f} GB per decode GPU; 3 nodes, 21 GPUs")
stage_kv, act = 2 * 32 * 12288 * 2, 12288 * 2
print(f"  per token: one stage's KV = {stage_kv / MiB:.1f} MiB stays in the node; the activation that crosses"
      f" to the next node = {act / KiB:.0f} KiB -> {stage_kv // act}x smaller (our inference)")

LINKS = (("NVLink 450 GB/s", 450e9), ("IB 50 GB/s", 50e9))
print(f"\n{'model':14s} {'tokens':>6s} {'KV size':>10s}  " + "  ".join(f"{n:>16s}" for n, _ in LINKS) + "   10 req/s needs")
for name, b in (("OPT-66B", opt66_tok), ("Llama-3.1-8B", kv8_tok), ("Llama-3.1-70B", kv70_tok)):
    for n in (512, 8192):
        size = n * b
        ts = "  ".join(f"{size / bw * 1e3:13.2f} ms" for _, bw in LINKS)
        print(f"{name:14s} {n:6d} {size / MiB:7.0f} MiB  {ts}   {10 * size / GB:6.2f} GB/s")

pf70 = 2 * 70.55e9 * 8192 / (8 * 989e12)
print(f"(context: Llama-3.1-70B 8k prefill floor at TP8 = 2*70.55e9*8192/(8*989e12) = {pf70 * 1e3:.0f} ms;"
      f" IB transfer = {8192 * kv70_tok / 50e9 / pf70:.0%} of it)")

# ------------------------------------------------------------------ model
N_PARAMS, W_BYTES, KV_TOK = 8.03e9, 16.06e9, kv8_tok
FLOPS, BW, HBM, NVL = 989e12, 3.35e12, 80e9, 450e9
D_MODEL, N_LAYERS = 4096, 32
HOP = 1e-6                              # illustrative latency per ring hop (course shared value)
L_M = int(W_BYTES / BW / (2 * N_PARAMS / FLOPS))   # prompt tokens that saturate compute


def allreduce(tp, tokens):
    if tp == 1:
        return 0.0
    msg = tokens * D_MODEL * 2
    return 2 * N_LAYERS * (2 * (tp - 1) * HOP + 2 * (tp - 1) / tp * msg / NVL)


def t_prefill(tokens, tp):
    return max(2 * N_PARAMS * tokens / (FLOPS * tp), W_BYTES / (tp * BW)) + allreduce(tp, tokens)


def t_decode(batch, ctx_tokens, tp):
    return (max(2 * N_PARAMS * batch / (FLOPS * tp), W_BYTES / (tp * BW))
            + ctx_tokens * KV_TOK / (tp * BW) + allreduce(tp, batch))


def kv_capacity(tp):                    # tokens of KV the instance can hold
    return int((tp * HBM - W_BYTES) / KV_TOK)


def workload(mean_in, mean_out, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.8
    li = np.clip(rng.lognormal(math.log(mean_in) - s * s / 2, s, n), 16, 8192).astype(int)
    lo = np.clip(rng.lognormal(math.log(mean_out) - s * s / 2, s, n), 2, 2048).astype(int)
    gaps = rng.exponential(1.0, n)       # unit-rate Poisson; scaled by 1/rate later
    return li, lo, np.cumsum(gaps)


def sim_prefill(arr, li, tp):
    """One prefill instance, FCFS, batches queued prompts up to L_M tokens. Returns first-token times."""
    done = np.empty(len(arr))
    free, i, n = 0.0, 0, len(arr)
    while i < n:
        start = max(free, arr[i])
        j, tok = i + 1, li[i]
        while j < n and arr[j] <= start and tok + li[j] <= L_M:
            tok += li[j]
            j += 1
        free = start + t_prefill(tok, tp)
        done[i:j] = free
        i = j
    return done


def sim_decode(ready, li, lo, tp):
    """One decode instance with continuous batching. ready = when each request's KV is on the GPU.
    Returns finish times. Admission FCFS while KV (prompt + output) fits."""
    order = np.argsort(ready, kind="stable")
    cap = kv_capacity(tp)
    finish = np.empty(len(ready))
    t, k, n = 0.0, 0, len(ready)
    active, ctx, used, step = [], 0, 0, 0   # heap of (finish_step, idx)
    while k < n or active:
        if not active and ready[order[k]] > t:
            t = ready[order[k]]
        while k < n and ready[order[k]] <= t and used + li[order[k]] + lo[order[k]] <= cap:
            r = order[k]
            heapq.heappush(active, (step + lo[r] - 1, r))
            ctx += li[r] + 1
            used += li[r] + lo[r]
            k += 1
        if not active:                   # memory full is impossible here; waiting for arrivals
            t = ready[order[k]]
            continue
        t += t_decode(len(active), ctx, tp)
        step += 1
        ctx += len(active)
        while active and active[0][0] <= step:
            _, r = heapq.heappop(active)
            finish[r] = t
            ctx -= li[r] + lo[r]
            used -= li[r] + lo[r]
    return finish


def sim_colocated(arr, li, lo, budget=4096):
    """One GPU doing both, vLLM-style: waiting prefills run first (up to `budget` tokens), else a decode step."""
    cap = kv_capacity(1)
    first, finish = np.empty(len(arr)), np.empty(len(arr))
    t, k, n = 0.0, 0, len(arr)
    active, ctx, used, step = [], 0, 0, 0
    while k < n or active:
        if not active and arr[k] > t:
            t = arr[k]
        batch, tok = [], 0
        while (k < n and arr[k] <= t and used + li[k] + lo[k] <= cap
               and (not batch or tok + li[k] <= budget)):
            batch.append(k)
            tok += li[k]
            used += li[k] + lo[k]
            k += 1
        if batch:
            t += t_prefill(tok, 1)
            for r in batch:
                first[r] = t
                heapq.heappush(active, (step + lo[r] - 1, r))
                ctx += li[r] + 1
            continue
        t += t_decode(len(active), ctx, 1)
        step += 1
        ctx += len(active)
        while active and active[0][0] <= step:
            _, r = heapq.heappop(active)
            finish[r] = t
            ctx -= li[r] + lo[r]
            used -= li[r] + lo[r]
    return first, finish


def tpot(first, finish, lo):
    return (finish - first) / np.maximum(lo - 1, 1)


def max_rate(ok_at, hi=2000.0):
    """Largest request rate where ok_at(rate) holds (binary search, like DistServe)."""
    lo_r = 0.5
    if not ok_at(lo_r):
        return 0.0
    for _ in range(12):
        mid = (lo_r + hi) / 2
        lo_r, hi = (mid, hi) if ok_at(mid) else (lo_r, mid)
    return lo_r


ATTAIN = 0.90
TARGET = 400.0                          # req/s the fleet must serve
# name: (mean input, mean output, TTFT SLO s, TPOT SLO s). Means from DistServe Fig 7; SLOs are
# ILLUSTRATIVE, shaped like DistServe Table 1: code = tight TTFT, summarization = loose TTFT, tight TPOT.
WORKLOADS = {"chatbot (ShareGPT means)": (755.5, 200.3, 0.200, 0.020),
             "code (HumanEval means)": (171.3, 98.2, 0.025, 0.040),
             "summarization (LongBench means)": (1738.3, 90.7, 1.000, 0.010)}

print(f"\n== Model: Llama-3.1-8B BF16 on H100 (ours) ==")
print(f"L_m (prompt tokens that saturate compute) = W/BW / (2N/FLOPs) = {L_M}")
print(f"prefill 755 tok TP1 = {t_prefill(755, 1) * 1e3:.2f} ms, TP2 = {t_prefill(755, 2) * 1e3:.2f} ms,"
      f" TP4 = {t_prefill(755, 4) * 1e3:.2f} ms")
print(f"decode step B=64, 64x955 ctx: TP1 = {t_decode(64, 64 * 955, 1) * 1e3:.2f} ms,"
      f" TP2 = {t_decode(64, 64 * 955, 2) * 1e3:.2f} ms")

WINDOW, POOL = 20.0, 40000             # simulate ~20 s of arrivals; score requests after the first 10%


def take(pool, rate):
    li, lo, base = pool
    n = min(POOL, max(1000, int(rate * WINDOW)))
    return base[:n] / rate, li[:n], lo[:n], np.arange(n) >= n // 10


def attained(ok, scored):
    return np.mean(ok[scored]) >= ATTAIN


RESULTS = {}
for wname, (mi, mo, TTFT_SLO, TPOT_SLO) in WORKLOADS.items():
    pool = workload(mi, mo, n=POOL)
    print(f"\n== Part 2. Alg 1 lite, {wname} ==")
    print(f"sampled means: input {pool[0].mean():.0f}, output {pool[1].mean():.0f};"
          f" SLOs (illustrative): TTFT {TTFT_SLO * 1e3:.0f} ms, TPOT {TPOT_SLO * 1e3:.0f} ms, {ATTAIN:.0%} attainment")
    best = {}
    for phase in ("prefill", "decode"):
        rows = []
        for tp in (1, 2, 4):                 # TP8 never won in our runs; add it back to see
            def ok(r, tp=tp, phase=phase, TTFT_SLO=TTFT_SLO, TPOT_SLO=TPOT_SLO):
                arr, li, lo, sc = take(pool, r)
                if phase == "prefill":
                    return attained(sim_prefill(arr, li, tp) - arr <= TTFT_SLO, sc)
                return attained(tpot(arr, sim_decode(arr, li, lo, tp), lo) <= TPOT_SLO, sc)
            g = max_rate(ok)
            rows.append((g / tp, tp, g))
            print(f"  {phase:7s} TP{tp}: instance goodput {g:6.1f} req/s -> {g / tp:5.1f} req/s per GPU")
        best[phase] = max(rows)
    gp, tpp, gip = best["prefill"]
    gd, tpd, gid = best["decode"]
    n, m = math.ceil(TARGET / gip), math.ceil(TARGET / gid)
    print(f"  best prefill: TP{tpp} ({gp:.1f}/GPU)   best decode: TP{tpd} ({gd:.1f}/GPU)")
    print(f"  fluid P:D GPU ratio = {gd:.1f}/{gp:.1f} = {gd / gp:.2f} : 1;"
          f" pair goodput per GPU = 1/(1/{gp:.1f} + 1/{gd:.1f}) = {1 / (1 / gp + 1 / gd):.1f}")
    print(f"  replicate for {TARGET:.0f} req/s: n = ceil({TARGET:.0f}/{gip:.1f}) = {n} prefill,"
          f" m = ceil({TARGET:.0f}/{gid:.1f}) = {m} decode -> {n * tpp} + {m * tpd} = {n * tpp + m * tpd} GPUs")

    # ---------------------------------------------------------------- Part 3
    arr, li, lo, sc = take(pool, TARGET)
    first = np.empty(len(arr))
    for p in range(n):                   # round-robin (DistServe: shortest queue)
        idx = np.arange(p, len(arr), n)
        first[idx] = sim_prefill(arr[idx], li[idx], tpp)
    ready = first + li * KV_TOK / NVL    # decode pulls the KV over NVLink
    fin = np.empty(len(arr))
    for d in range(m):                   # round-robin (DistServe: least loaded)
        idx = np.arange(d, len(arr), m)
        fin[idx] = sim_decode(ready[idx], li[idx], lo[idx], tpd)
    ok_t = (first - arr <= TTFT_SLO)[sc]
    ok_p = (tpot(first, fin, lo) <= TPOT_SLO)[sc]
    print(f"== Part 3. end-to-end at {TARGET:.0f} req/s on {n * tpp + m * tpd} GPUs: TTFT ok {ok_t.mean():.1%},"
          f" TPOT ok {ok_p.mean():.1%}, both {np.mean(ok_t & ok_p):.1%}")

    def co_ok(r, TTFT_SLO=TTFT_SLO, TPOT_SLO=TPOT_SLO):
        arr, li, lo, sc = take(pool, r)
        f, fi = sim_colocated(arr, li, lo)
        return attained((f - arr <= TTFT_SLO) & (tpot(f, fi, lo) <= TPOT_SLO), sc)
    gc = max_rate(co_ok)
    per_gpu = TARGET / (n * tpp + m * tpd)
    if gc == 0:
        print(f"  colocated TP1 replica: cannot reach {ATTAIN:.0%} attainment at any rate")
    else:
        arr, li, lo, sc = take(pool, gc)
        f, fi = sim_colocated(arr, li, lo)
        print(f"  colocated TP1 replica: goodput {gc:.1f} req/s per GPU"
              f" (at that rate: TTFT ok {np.mean((f - arr <= TTFT_SLO)[sc]):.1%},"
              f" TPOT ok {np.mean((tpot(f, fi, lo) <= TPOT_SLO)[sc]):.1%})")
        print(f"  disaggregated: {TARGET:.0f} / {n * tpp + m * tpd} = {per_gpu:.1f} req/s per GPU"
              f" = {per_gpu / gc:.2f}x colocated; colocated needs ceil({TARGET:.0f}/{gc:.1f}) = {math.ceil(TARGET / gc)} GPUs")
        pair = 1 / (1 / gp + 1 / gd)
        print(f"  without rounding: pair goodput {pair:.1f} / colocated {gc:.1f} = {pair / gc:.2f}x")
    RESULTS[wname] = (gp, tpp, gd, tpd, n, m, gc)

# ------------------------------------------------------------------ Part 4
print("\n== Part 4. Scaling Book's balance P/T_prefill = B*G/(T_step*n_out), with our TP1 step times ==")
for wname, (mi, mo, TTFT_SLO, TPOT_SLO) in WORKLOADS.items():
    tp_ = t_prefill(int(mi), 1)
    ctx = int(mi + mo / 2)                                # average context during decode
    B = max(b for b in range(1, 2000) if t_decode(b, b * ctx, 1) <= TPOT_SLO and b * (mi + mo) <= kv_capacity(1))
    ts = t_decode(B, B * ctx, 1)
    ratio = tp_ * B / (ts * mo)
    gp, tpp, gd, tpd, n, m, gc = RESULTS[wname]
    print(f"{wname}: T_prefill({int(mi)}) = {tp_ * 1e3:.2f} ms; largest decode batch within TPOT and HBM:"
          f" B = {B}, T_step = {ts * 1e3:.2f} ms; n_out = {mo:.0f}")
    print(f"   P/G = {tp_ * 1e3:.2f} * {B} / ({ts * 1e3:.2f} * {mo:.0f}) = {ratio:.2f}   (100% busy; the simulator above adds queueing and SLO tails)")
print("(Book, LLaMA 3-70B on TPU v5e: P/0.91 = 32 G/(0.019*512) -> P = 3G)")

# Try this:
# 1. Give chatbot a tight TTFT (0.030): prefill switches to TP4 and disaggregation's edge over colocation grows.
# 2. Loosen every SLO 10x: the unrounded gains shrink to about 1.01x / 0.98x / 1.55x -- DistServe section 7's
#    throughput-oriented case, where colocated batching catches up.
# 3. Set TARGET = 10: every phase rounds up to a whole instance, so disaggregation needs 2-5 GPUs where
#    colocation needs 1-2 -- section 7's "few GPUs" case.
