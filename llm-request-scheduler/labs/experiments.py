"""Every number in the course comes from this file.  python3 experiments.py  (about a minute)

It runs scheduler.py's simulator on seeded synthetic traces and prints markdown tables.
The cost model is Llama-3.1-8B BF16 on one H100 SXM at 100% of peak, so the times are
floors: real engines are slower by their kernel efficiency and CPU overhead.
"""
import sys
import time
from scheduler import (BlockManager, CostModel, Request, Scheduler, clone, poisson_trace, simulate,
                       simulate_request_level, toy_trace, pct)

cost = CostModel()
NB = cost.kv_blocks()


def h(title):
    print(f"\n## {title}\n")


def ms(x):
    return f"{x*1e3:,.1f}"


def e1_why_batch():
    h("E1 · step time vs batch size (decode, 2,048-token context each)")
    print("| B | step ms | total tok/s | per-user tok/s |\n|---|---|---|---|")
    for b in [1, 8, 32, 64, 128, 238]:
        t = cost.raw(b, b * 2048)
        print(f"| {b} | {ms(t)} | {b/t:,.0f} | {1/t:,.0f} |")


def run_all(rate, n=1000, seed=1, **kw):
    base = poisson_trace(n, rate=rate, seed=seed, **kw)
    return {
        "static B=32": simulate_request_level(clone(base), cost, 32),
        "dynamic B=32, 50 ms window": simulate_request_level(clone(base), cost, 32, 0.05),
        "continuous (whole-prompt prefill)": simulate(clone(base), Scheduler(
            BlockManager(NB), chunked_prefill=False, max_num_batched_tokens=16384), cost),
        "continuous + chunked prefill (2,048)": simulate(clone(base), Scheduler(BlockManager(NB)), cost),
    }


def e2_strategies():
    h("E2 · batching strategies (1,000 requests, prompt median 512, output median 200; SLO TTFT<=1 s, TPOT<=50 ms)")
    for rate in [2, 20, 60]:
        print(f"\n**{rate} req/s**\n\n| strategy | out tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | in SLO | padding waste |\n|---|---|---|---|---|---|---|")
        for name, r in run_all(rate).items():
            print(f"| {name} | {r.throughput:,.0f} | {ms(r.ttft_p50)} | {ms(r.ttft_p99)} | {ms(r.tpot_p50)} | {r.slo_ok*100:.0f}% | {r.padding_waste*100:.0f}% |")
    print("\n**capacity: highest rate (req/s) with >= 99% of requests inside the SLO, 3,000-request traces"
          " (long enough to reach steady state; 600-request traces overstate it)**\n")
    for rate in [1]:
        tr = poisson_trace(3000, rate=rate, seed=2)
        print(f"- static B=32 at {rate} req/s: {simulate_request_level(clone(tr), cost, 32).slo_ok*100:.0f}% in SLO")
        print(f"- dynamic B=32, 50 ms at {rate} req/s: {simulate_request_level(clone(tr), cost, 32, 0.05).slo_ok*100:.0f}% in SLO")
    for name, slo in [("course SLO (TTFT<=1 s, TPOT<=50 ms)", (1.0, 0.05)), ("tight SLO (TTFT<=350 ms, TPOT<=25 ms)", (0.35, 0.025))]:
        best = 0
        for rate in range(34, 44):
            r = simulate(poisson_trace(3000, rate=rate, seed=2), Scheduler(BlockManager(NB)), cost,
                         slo_ttft=slo[0], slo_tpot=slo[1])
            if r.slo_ok >= 0.99:
                best = rate
            else:
                break
        print(f"- continuous + chunked, {name}: {best} req/s")


def e3_chunked():
    h("E3 · chunked prefill vs whole-prompt prefill (long prompts: median 2,048, rate 8 req/s)")
    base = poisson_trace(600, rate=8, seed=3, prompt_median=2048, output_median=200)
    print("| config | out tok/s | TTFT p50 ms | TTFT p99 ms | ITL p99 ms | ITL max ms | TPOT p99 ms |\n|---|---|---|---|---|---|---|")
    rows = [("whole prompt (no chunking)", dict(chunked_prefill=False, max_num_batched_tokens=16384))]
    rows += [(f"chunked, budget {b:,}", dict(max_num_batched_tokens=b)) for b in [256, 512, 1024, 2048, 4096]]
    for name, kw in rows:
        r = simulate(clone(base), Scheduler(BlockManager(NB), **kw), cost)
        print(f"| {name} | {r.throughput:,.0f} | {ms(r.ttft_p50)} | {ms(r.ttft_p99)} | {ms(r.itl_p99)} | {ms(r.itl_max)} | {ms(r.tpot_p99)} |")
    print("\nOne 4,096-token prefill alone:", ms(cost.raw(4096, 4096)), "ms; joined to 64 decodes at 2k:",
          ms(cost.raw(4096 + 64, 4096 + 64 * 2048)), "ms vs", ms(cost.raw(64, 64 * 2048)), "ms for the decodes alone")


def e4_memory():
    h("E4 · KV memory, admission and preemption (rate 30 req/s, 1,000 requests)")
    base = poisson_trace(1000, rate=30, seed=4)
    print("| KV blocks (tokens) | ~GB | out tok/s | TTFT p99 ms | TPOT p50 ms | preemptions | recomputed tokens |\n|---|---|---|---|---|---|---|")
    for nb in [NB, 8000, 4000, 2000, 1000]:
        tr = clone(base)
        s = Scheduler(BlockManager(nb))
        recomputed = [0]

        def log(i, t, plan, dt, sch, rc=recomputed):
            for v in plan.preempted:
                rc[0] += v.num_tokens
        r = simulate(tr, s, cost, log=log)
        print(f"| {nb:,} ({nb*16:,}) | {nb*16*131072/1e9:.1f} | {r.throughput:,.0f} | {ms(r.ttft_p99)} | {ms(r.tpot_p50)} | {r.preemptions:,} | {recomputed[0]:,} |")
    print("\nmax_num_seqs sweep at full memory, rate 60 (overload):\n\n| max_num_seqs | out tok/s | TPOT p50 ms | TTFT p50 ms |\n|---|---|---|---|")
    base = poisson_trace(1000, rate=60, seed=4)
    for m in [16, 64, 128, 256, 512]:
        r = simulate(clone(base), Scheduler(BlockManager(NB), max_num_seqs=m), cost)
        print(f"| {m} | {r.throughput:,.0f} | {ms(r.tpot_p50)} | {ms(r.ttft_p50)} |")


def e5_policies():
    h("E5 · admission policy (rate 55 req/s = just past capacity, 1,000 requests, max_num_seqs 64)")
    print("| policy | TTFT p50 ms | TTFT p99 ms | E2E p50 s | E2E p99 s | worst E2E s | out tok/s |\n|---|---|---|---|---|---|---|")
    rows = [("FCFS", "fcfs", None), ("SJF, perfect length oracle", "sjf", None),
            ("SJF, predictor with 50% noise", "sjf", 0.5), ("SJF, predictor with 100% noise", "sjf", 1.0)]
    for name, pol, noise in rows:
        tr = poisson_trace(1000, rate=55, seed=5, predictor_noise=noise)
        r = simulate(tr, Scheduler(BlockManager(NB), policy=pol, max_num_seqs=64), cost)
        worst = max(x.finish_time - x.arrival for x in tr)
        print(f"| {name} | {ms(r.ttft_p50)} | {ms(r.ttft_p99)} | {r.e2e_p50:.2f} | {r.e2e_p99:.2f} | {worst:.2f} | {r.throughput:,.0f} |")
    print("\nPriority classes (10% of requests priority 0 = interactive, 90% priority 1 = batch), same load:\n")
    print("| policy | class | TTFT p50 ms | TTFT p99 ms |\n|---|---|---|---|")
    for pol in ["fcfs", "priority"]:
        tr = poisson_trace(1000, rate=55, seed=5)
        for x in tr:
            x.priority = 0 if x.rid % 10 == 0 else 1
        simulate(tr, Scheduler(BlockManager(NB), policy=pol, max_num_seqs=64), cost)
        for cls, label in [(0, "interactive"), (1, "batch")]:
            t = [x.first_token_time - x.arrival for x in tr if x.priority == cls]
            print(f"| {pol} | {label} | {ms(pct(t, 50))} | {ms(pct(t, 99))} |")


def e6_fairness():
    h("E6 · fairness: tenant A sends 90% of traffic, B 10% (rate 60 req/s = overload, max_num_seqs 64)")
    print("| policy | tenant | requests | TTFT p50 ms | TTFT p99 ms | share of output tokens |\n|---|---|---|---|---|---|")
    for pol in ["fcfs", "fair"]:
        tr = poisson_trace(1000, rate=60, seed=6, tenants={"A": 0.9, "B": 0.1})
        simulate(tr, Scheduler(BlockManager(NB), policy=pol, max_num_seqs=64), cost)
        horizon = sorted(x.finish_time for x in tr)[len(tr) // 2]      # first half of the run
        tot = sum(sum(1 for tt in x.token_times if tt <= horizon) for x in tr)
        for ten in "AB":
            xs = [x for x in tr if x.tenant == ten]
            t = [x.first_token_time - x.arrival for x in xs]
            got = sum(sum(1 for tt in x.token_times if tt <= horizon) for x in xs)
            print(f"| {pol} | {ten} | {len(xs)} | {ms(pct(t, 50))} | {ms(pct(t, 99))} | {got/tot*100:.0f}% |")


def e7_prefix():
    h("E7 · prefix caching: 4 shared system prompts of 2,048 tokens + a unique suffix (median 128), rate 20 req/s")
    prefixes = [[p * 10000 + i for i in range(2048)] for p in range(4)]
    base = poisson_trace(800, rate=20, seed=7, prompt_median=128, output_median=200, prefixes=prefixes)
    print("| config | prefix hit rate | TTFT p50 ms | TTFT p99 ms | out tok/s | prefill tokens computed |\n|---|---|---|---|---|---|")
    for name, pc in [("no prefix cache", False), ("prefix cache (block hashing)", True)]:
        tr = clone(base)
        r = simulate(tr, Scheduler(BlockManager(NB, prefix_caching=pc)), cost)
        computed = sum(x.prompt_len - x.num_cached for x in tr)
        print(f"| {name} | {r.prefix_hit*100:.1f}% | {ms(r.ttft_p50)} | {ms(r.ttft_p99)} | {r.throughput:,.0f} | {computed:,} |")


def e8_toy():
    h("E8 · the 8-request toy trace, step by step (budget 16 tokens, max 4 seqs, 12 blocks of 4 tokens)")
    print("Requests (id: arrival ms, prompt, output):",
          ", ".join(f"{r.rid}: {r.arrival*1e3:.0f}, {r.prompt_len}, {r.output_len}" for r in toy_trace()))
    print("\n| step | t start ms | scheduled (id:tokens, P=prefill D=decode) | preempted | running | waiting | free blocks |\n|---|---|---|---|---|---|---|")

    def log(i, t, plan, dt, s):
        if i >= 16:
            return
        sc = " ".join(f"{r.rid}:{n}{'P' if r.num_computed < r.prompt_len else 'D'}" for r, n in plan.scheduled)
        pre = " ".join(str(v.rid) for v in plan.preempted) or "–"
        print(f"| {i} | {t*1e3:.2f} | {sc} | {pre} | {len(s.running)} | {len(s.waiting)} | {s.bm.num_free()} |")
    tr = toy_trace()
    r = simulate(tr, Scheduler(BlockManager(12, block_size=4), max_num_batched_tokens=16, max_num_seqs=4, watermark=0), cost, log=log)
    print(f"\nTotal steps {r.steps}, preemptions {r.preemptions}. Per request (TTFT ms, finish ms):",
          ", ".join(f"{x.rid}: {ms(x.first_token_time - x.arrival)}/{ms(x.finish_time)}" for x in tr))


def e9_overhead():
    h("E9 · scheduler CPU cost per step, Python on the Apple M3 (light, single-threaded)")
    print("| policy | waiting | running | µs per step() |\n|---|---|---|---|")
    for pol in ["fcfs", "sjf", "fair"]:
        for w in [100, 10000]:
            s = Scheduler(BlockManager(NB), policy=pol, max_num_seqs=256, max_num_batched_tokens=2048)
            for i in range(w):
                s.add(Request(i, 0.0, 64, 10**9, tenant="ABCD"[i % 4], predicted_len=100 + i % 50))
            # fill running with decoders (they never finish during the benchmark)
            while len(s.running) < min(256, w):
                plan = s.step()
                s.finish_step(plan, 0.0)
            reps = 200
            t0 = time.perf_counter()
            for _ in range(reps):
                plan = s.step()
                s.finish_step(plan, 0.0)
            dt = (time.perf_counter() - t0) / reps
            print(f"| {pol} | {w:,} | {len(s.running)} | {dt*1e6:,.0f} |")


if __name__ == "__main__":
    only = sys.argv[1:]
    for f in [e1_why_batch, e2_strategies, e3_chunked, e4_memory, e5_policies, e6_fairness, e7_prefix, e8_toy, e9_overhead]:
        if not only or f.__name__.split("_")[0] in only:
            f()
