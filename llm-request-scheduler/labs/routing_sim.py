"""Many replicas: how the router changes the prefix-cache hit rate and the tail.

python3 routing_sim.py   (about a minute, standard library only)

Four copies of scheduler.py's engine (one Scheduler + BlockManager each, same
100%-of-peak CostModel: Llama-3.1-8B BF16 on one H100 SXM, so times are floors).
A router sees every arrival and picks a replica. Four routers:

  round robin    replica = i mod N
  least-loaded   lowest 4*len(waiting) + len(running)   (vLLM's DP router score)
  prefix-hash    replica = hash(first block of the prompt) mod N
  scored         highest W_HIT*hit - W_LOAD*load
                 hit  = fraction of the prompt already cached on that replica
                 load = that replica's 4*waiting + running, divided by the max
                        over replicas (so it lies in 0..1)

Workload A = E7 at 4x the rate: 4 shared 2,048-token system prompts + a unique
suffix (median 128), 80 req/s, 600 requests, full H100 KV per replica.
Workload B = the same shape with 64 system prompts (8,192 blocks of prefix) and
4,000 KV blocks (8.4 GB) per replica, 800 requests: the prompts no longer fit in
one replica's cache but do fit in the cluster's (an illustrative stress case).
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler import BlockManager, CostModel, Scheduler, clone, pct, poisson_trace  # noqa: E402

cost = CostModel()
N = 4
W_HIT, W_LOAD = 3.0, 3.0


class Blocks(BlockManager):
    """scheduler.py's BlockManager, but cache_full_blocks only looks at blocks
    filled since its last call (the original rescans the whole prompt every
    step, which is correct but too slow for 4 replicas x 4 routers)."""

    def cache_full_blocks(self, req):
        if not (self.prefix_caching and req.prompt_tokens):
            return
        full = min(req.num_computed, req.prompt_len) // self.block_size
        start = req.pub if getattr(req, "pub_list", None) is req.blocks else 0
        if start >= full:
            return
        hashes = self._chain_hashes(req)
        for i in range(start, full):
            b, h = req.blocks[i], hashes[i]
            if b not in self.block_hash and h not in self.hash_to_block:
                self.block_hash[b] = h
                self.hash_to_block[h] = b
        req.pub, req.pub_list = max(start, full), req.blocks


def load(s: Scheduler) -> int:
    return 4 * len(s.waiting) + len(s.running)


def cached_fraction(s: Scheduler, req) -> float:
    bm = s.bm
    hashes = bm._chain_hashes(req)
    n = 0
    for h in hashes:
        if h not in bm.hash_to_block:
            break
        n += 1
    return n * bm.block_size / req.prompt_len


def route(policy: str, i: int, req, reps: list[Scheduler]) -> int:
    if policy == "round robin":
        return i % N
    if policy == "least-loaded":
        return min(range(N), key=lambda k: load(reps[k]))
    if policy == "prefix-hash":
        first = ",".join(map(str, req.prompt_tokens[:16]))
        return int(hashlib.sha1(first.encode()).hexdigest(), 16) % N
    if policy == "scored":
        loads = [load(s) for s in reps]
        top = max(1, max(loads))
        return max(range(N), key=lambda k: W_HIT * cached_fraction(reps[k], req)
                   - W_LOAD * loads[k] / top)
    raise ValueError(policy)


def run(trace, policy: str, num_blocks: int):
    reps = [Scheduler(Blocks(num_blocks, prefix_caching=True)) for _ in range(N)]
    clock = [0.0] * N
    sent = [0] * N

    def advance(k, until):
        s = reps[k]
        while s.has_work() and clock[k] < until:
            plan = s.step()
            clock[k] += cost.step_time(plan)
            s.finish_step(plan, clock[k])
        if not s.has_work():
            clock[k] = max(clock[k], until)

    for i, req in enumerate(trace):
        for k in range(N):
            advance(k, req.arrival)
        k = route(policy, i, req, reps)
        sent[k] += 1
        reps[k].add(req)
    for k in range(N):
        advance(k, float("inf"))

    ttft = [r.first_token_time - r.arrival for r in trace]
    toks = sum(r.output_len for r in trace)
    span = max(r.finish_time for r in trace) - min(r.arrival for r in trace)
    hit = sum(r.num_cached for r in trace) / sum(r.prompt_len for r in trace)
    ok = sum(1 for r in trace if r.first_token_time - r.arrival <= 1.0 and
             (r.output_len <= 1 or (r.finish_time - r.first_token_time) / (r.output_len - 1) <= 0.05))
    return dict(hit=hit, p50=pct(ttft, 50), p99=pct(ttft, 99), thr=toks / span,
                slo=ok / len(trace), pre=sum(r.preemptions for r in trace), sent=sent)


def table(title, base, num_blocks):
    print(f"\n## {title}\n", flush=True)
    print("| router | prefix hit | TTFT p50 ms | TTFT p99 ms | out tok/s | in SLO | "
          "requests per replica | preemptions |\n|---|---|---|---|---|---|---|---|")
    for pol in ["round robin", "least-loaded", "prefix-hash", "scored"]:
        r = run(clone(base), pol, num_blocks)
        print(f"| {pol} | {r['hit']*100:.1f}% | {r['p50']*1e3:,.1f} | {r['p99']*1e3:,.1f} | "
              f"{r['thr']:,.0f} | {r['slo']*100:.0f}% | {' / '.join(map(str, r['sent']))} | "
              f"{r['pre']:,} |", flush=True)


if __name__ == "__main__":
    nb = cost.kv_blocks()
    pre4 = [[p * 10000 + i for i in range(2048)] for p in range(4)]
    table("A · E7 workload x4: 4 system prompts, 80 req/s, 600 requests, "
          f"{nb:,} blocks per replica",
          poisson_trace(600, rate=80, seed=7, prompt_median=128, output_median=200,
                        prefixes=pre4), nb)
    pre64 = [[p * 10000 + i for i in range(2048)] for p in range(64)]
    table("B · 64 system prompts (8,192 blocks of prefix), 80 req/s, 800 requests, "
          "4,000 blocks per replica",
          poisson_trace(800, rate=80, seed=8, prompt_median=128, output_median=200,
                        prefixes=pre64), 4000)
