"""Queueing-knee card: open-loop rate sweep and closed-loop concurrency sweep on the course simulator.

Run:  uv run --python 3.12 knee.py 2    (the card's numbers; the argument multiplies every trace length)
      uv run --python 3.12 knee.py      (half-length traces: faster, slightly different numbers)
The last section ("fixed-length checks") prints the same thing whatever the argument: the mean
request size, the top rate across seeds, and the test-length comparisons the card quotes.
"""
import math, random, sys
from collections import deque
from scheduler import BlockManager, CostModel, Request, Scheduler, poisson_trace, pct

cost = CostModel(); NB = cost.kv_blocks()

def stats(reqs, extra):
    ttft = [r.first_token_time - r.arrival for r in reqs]
    tpot = [(r.finish_time - r.first_token_time) / (r.output_len - 1) for r in reqs if r.output_len > 1]
    e2e = [r.finish_time - r.arrival for r in reqs]
    d = dict(n=len(reqs), ttft50=pct(ttft, 50), ttft95=pct(ttft, 95), ttft_mean=sum(ttft)/len(ttft),
             tpot50=pct(tpot, 50), tpot95=pct(tpot, 95), e2e_mean=sum(e2e)/len(e2e),
             out_mean=sum(r.output_len for r in reqs)/len(reqs))
    d.update(extra); return d

def drive(sched, next_arrivals, on_done, total):
    """generic loop. next_arrivals(t) -> list of requests arriving at or before t, or next arrival time."""
    t = 0.0; done_all = []; area_run = area_wait = 0.0
    blocked = {"seats": 0, "budget": 0, "memory": 0}; steps = 0
    qwait = {}
    while len(done_all) < total:
        for r in next_arrivals.pop_until(t):
            sched.add(r)
        if not sched.has_work():
            t = next_arrivals.peek_time(); continue
        plan = sched.step()
        for r, n in plan.scheduled:
            if r.rid not in qwait: qwait[r.rid] = t - r.arrival
        dt = cost.step_time(plan)
        nw, nr = len(sched.waiting), len(sched.running)
        if nw:
            if nr >= sched.max_num_seqs: blocked["seats"] += 1
            elif plan.num_tokens >= sched.token_budget: blocked["budget"] += 1
            else: blocked["memory"] += 1
        area_run += nr * dt; area_wait += nw * dt
        t += dt; steps += 1
        for r in sched.finish_step(plan, t):
            done_all.append(r); on_done(r, t)
    return done_all, t, dict(avg_run=area_run / t, avg_wait=area_wait / t, blocked=blocked, steps=steps, qwait=qwait)

class Open:
    def __init__(s, trace): s.q = deque(sorted(trace, key=lambda r: r.arrival))
    def pop_until(s, t):
        out = []
        while s.q and s.q[0].arrival <= t: out.append(s.q.popleft())
        return out
    def peek_time(s): return s.q[0].arrival

class Closed:
    """C users; each sends its next request the moment the previous one finishes (zero think time)."""
    def __init__(s, lengths, C):
        s.lengths = deque(lengths); s.ready = deque(); s.i = 0
        for _ in range(C): s.emit(0.0)
    def emit(s, t):
        if s.lengths:
            p, o = s.lengths.popleft(); s.ready.append(Request(s.i, t, p, o)); s.i += 1
    def pop_until(s, t):
        out = list(s.ready); s.ready.clear(); return out
    def peek_time(s): raise RuntimeError("closed loop idle")

def gamma_trace(n, rate, burst, seed):
    base = poisson_trace(n, rate=rate, seed=seed)   # same lengths as the Poisson trace
    rng = random.Random(seed + 1000); t = 0.0
    for r in base:
        t += rng.gammavariate(burst, 1.0 / (rate * burst)); r.arrival = t
    return base

def trim(reqs, lo=0.2, hi=0.8):
    """steady-state window: drop the first 20% and last 20% of requests by arrival order."""
    reqs = sorted(reqs, key=lambda r: r.arrival); n = len(reqs)
    return reqs[int(n * lo):int(n * hi)]

def open_run(rate, n, seed=2, burst=None, seqs=256, window=True):
    tr = poisson_trace(n, rate=rate, seed=seed) if burst is None else gamma_trace(n, rate, burst, seed)
    sch = Scheduler(BlockManager(NB), max_num_seqs=seqs)
    done, t, ex = drive(sch, Open(tr), lambda r, t: None, n)
    keep = trim(done) if window else done
    q = [ex["qwait"][r.rid] for r in keep]
    ex2 = dict(avg_run=ex["avg_run"], avg_wait=ex["avg_wait"], blocked=ex["blocked"], q_mean=sum(q)/len(q), q95=pct(q, 95),
               tok_s=sum(r.output_len for r in done)/t)
    return stats(keep, ex2)

def closed_run(C, n, seed=2, seqs=256):
    lens = [(r.prompt_len, r.output_len) for r in poisson_trace(n, rate=1, seed=seed)]
    src = Closed(lens, C)
    sch = Scheduler(BlockManager(NB), max_num_seqs=seqs)
    done, t, ex = drive(sch, src, lambda r, t: src.emit(t), n)
    keep = trim(done)
    t0, t1 = min(r.arrival for r in keep), max(r.arrival for r in keep)
    # delivered rate in the window = completions per second inside [t0, t1]
    comp = [r for r in done if t0 <= r.finish_time <= t1]
    X = len(comp) / (t1 - t0)
    toks = sum(r.output_len for r in comp) / (t1 - t0)
    q = [ex["qwait"][r.rid] for r in keep]
    return stats(keep, dict(X=X, tok_s=toks, avg_run=ex["avg_run"], avg_wait=ex["avg_wait"], blocked=ex["blocked"], q_mean=sum(q)/len(q)))

def ms(x): return f"{x*1e3:9.1f}"

if __name__ == "__main__":
    mult = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"# trace multiplier {mult}")
    print("\nCLOSED loop, max_num_seqs 256 | C | req/s | tok/s | meanE2E s | C/E2E | TTFT p50 | p95 | qwait mean | TPOT p50 | avg_run | avg_wait | blocked")
    for seqs in (256, 64):
        print(f"-- max_num_seqs {seqs}")
        for C in (1, 8, 16, 32, 64, 128, 256, 512):
            n = max(600, C * 30) * mult
            s = closed_run(C, n, seqs=seqs)
            print(f"{C:4d} | {s['X']:6.2f} | {s['tok_s']:7.0f} | {s['e2e_mean']:7.3f} | {C/s['e2e_mean']:6.2f} | {ms(s['ttft50'])} | {ms(s['ttft95'])} | {ms(s['q_mean'])} | {ms(s['tpot50'])} | {s['avg_run']:6.1f} | {s['avg_wait']:6.1f} | {s['blocked']}")
    print("\nOPEN loop Poisson, max_num_seqs 256 | rate | tok/s | TTFT p50 | p95 | qwait mean | TPOT p50 | meanE2E | rate*E2E | avg_run+wait | blocked")
    for rate in (5, 10, 20, 30, 34, 36, 38, 40, 42, 44, 50, 60):
        s = open_run(rate, 3000 * mult)
        print(f"{rate:4d} | {s['tok_s']:7.0f} | {ms(s['ttft50'])} | {ms(s['ttft95'])} | {ms(s['q_mean'])} | {ms(s['tpot50'])} | {s['e2e_mean']:7.3f} | {rate*s['e2e_mean']:7.1f} | {s['avg_run']+s['avg_wait']:7.1f} | {s['blocked']}")
    print("\nOPEN loop bursty (gamma shape 0.25, CV 2) | rate | TTFT p50 | p95 | TPOT p50")
    for rate in (10, 20, 26, 30, 34, 36):
        s = open_run(rate, 3000 * mult, burst=0.25)
        print(f"{rate:4d} | {ms(s['ttft50'])} | {ms(s['ttft95'])} | {ms(s['tpot50'])} | {s['blocked']}")

    print("\nFIXED-LENGTH CHECKS (independent of the argument)")
    tr = poisson_trace(6000, rate=1, seed=2)
    mp, mo = sum(r.prompt_len for r in tr) / 6000, sum(r.output_len for r in tr) / 6000
    print(f"mean request size, seed 2, 6,000 requests: prompt {mp:.0f} + output {mo:.0f} = {mp+mo:.0f} tokens")
    print("top rate, closed loop, 512 users, 15,360 requests | seed | req/s")
    for seed in (0, 1, 2, 3):
        print(f"{seed:4d} | {closed_run(512, 512 * 30, seed=seed)['X']:6.2f}")
    print("closed loop 512 users, TTFT p50 / p95 by test length")
    for n in (15360, 30720):
        s = closed_run(512, n); print(f"{n:6d} requests | {ms(s['ttft50'])} | {ms(s['ttft95'])}")
    print("open loop Poisson past the knee, TTFT p95 by test length | rate | 3,000 | 6,000")
    for rate in (38, 40, 44):
        print(f"{rate:4d} | {ms(open_run(rate, 3000)['ttft95'])} | {ms(open_run(rate, 6000)['ttft95'])}")
    print("TTFT p95, Poisson vs bursty (gamma shape 0.25) | rate | requests | Poisson | bursty")
    for rate in (26, 30, 34):
        for n in (3000, 6000, 12000):
            print(f"{rate:4d} | {n:6d} | {ms(open_run(rate, n)['ttft95'])} | {ms(open_run(rate, n, burst=0.25)['ttft95'])}")
