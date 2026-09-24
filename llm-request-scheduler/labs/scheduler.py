"""A small, complete LLM request scheduler plus a simulator to measure it.

Standard library only. Run it:  python3 scheduler.py
Tests:                          python3 -m unittest test_scheduler.py

Pieces (each is one card in the course):
  Request        the lifecycle state of one request
  BlockManager   a paged KV-cache allocator (free list, block tables, ref counts, prefix cache)
  WaitQueue      pluggable admission order: fcfs, sjf, priority, fair (VTC-style)
  Scheduler      step(): pick this iteration's batch under a token budget and memory
  CostModel      how long one GPU step takes (roofline: weights + KV reads vs FLOPs)
  simulate()     drive a trace through the scheduler and report TTFT / TPOT / throughput
  simulate_request_level()  static and dynamic (time-window) batching, for comparison

The cost model is the perf-2 step formula for Llama-3.1-8B BF16 on one H100 SXM at
100% of peak: an upper bound, not a benchmark. Every time is in seconds.
"""
from __future__ import annotations

import hashlib
import heapq
import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field

# ---------------------------------------------------------------- the request

WAITING, RUNNING, FINISHED = "waiting", "running", "finished"


@dataclass
class Request:
    rid: int
    arrival: float
    prompt_len: int
    output_len: int            # true length; the scheduler never reads it (the sim does)
    priority: int = 0          # lower number = more important
    tenant: str = "a"
    predicted_len: int | None = None   # what an SJF scheduler is allowed to see
    prompt_tokens: list[int] | None = None  # only needed for prefix caching

    state: str = WAITING
    num_computed: int = 0      # tokens whose KV is in the cache
    num_generated: int = 0     # output tokens produced so far
    num_cached: int = 0        # prompt tokens served from the prefix cache
    blocks: list[int] = field(default_factory=list)
    first_token_time: float | None = None
    finish_time: float | None = None
    token_times: list[float] = field(default_factory=list)
    preemptions: int = 0
    block_hashes: list[str] | None = None

    @property
    def num_tokens(self) -> int:           # prompt + everything generated so far
        return self.prompt_len + self.num_generated

    @property
    def remaining(self) -> int:            # tokens that still need a forward pass
        return self.num_tokens - self.num_computed

    @property
    def is_prefill(self) -> bool:
        return self.num_computed < self.prompt_len


# ---------------------------------------------------------- the KV block manager

class BlockManager:
    """Paged KV cache: fixed-size blocks, a free list, one block table per request.

    allocate / free are O(blocks touched). With prefix caching on, full prompt
    blocks are keyed by a chain hash; a block whose ref count drops to 0 stays
    cached (evictable, LRU) until the free list runs dry.
    """

    def __init__(self, num_blocks: int, block_size: int = 16, prefix_caching: bool = False):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free = deque(range(num_blocks))        # never-used or evicted blocks
        self.ref = [0] * num_blocks
        self.prefix_caching = prefix_caching
        self.hash_to_block: dict[str, int] = {}
        self.block_hash: dict[int, str] = {}
        self.evictable: OrderedDict[int, None] = OrderedDict()   # cached, ref 0, LRU order

    def num_free(self) -> int:
        return len(self.free) + len(self.evictable)

    def blocks_needed(self, req: Request, new_tokens: int) -> int:
        total = req.num_computed + new_tokens
        return max(0, math.ceil(total / self.block_size) - len(req.blocks))

    def can_allocate(self, req: Request, new_tokens: int, reserve: int = 0) -> bool:
        return self.blocks_needed(req, new_tokens) <= self.num_free() - reserve

    def _take(self) -> int:
        if self.free:
            return self.free.popleft()
        b, _ = self.evictable.popitem(last=False)   # evict least recently used cached block
        del self.hash_to_block[self.block_hash.pop(b)]
        return b

    def allocate(self, req: Request, new_tokens: int) -> None:
        for _ in range(self.blocks_needed(req, new_tokens)):
            b = self._take()
            self.ref[b] = 1
            req.blocks.append(b)

    def free_request(self, req: Request) -> None:
        for b in reversed(req.blocks):      # tail first, so shared prefixes are evicted last
            self.ref[b] -= 1
            if self.ref[b] == 0:
                if b in self.block_hash:
                    self.evictable[b] = None
                else:
                    self.free.append(b)
        req.blocks = []

    # ---- prefix caching
    def _chain_hashes(self, req: Request) -> list[str]:
        if req.block_hashes is not None:        # computed once per request
            return req.block_hashes
        hashes, parent = [], ""
        bs = self.block_size
        for i in range(req.prompt_len // bs):
            chunk = req.prompt_tokens[i * bs:(i + 1) * bs]
            parent = hashlib.sha1((parent + ",".join(map(str, chunk))).encode()).hexdigest()
            hashes.append(parent)
        req.block_hashes = hashes
        return hashes

    def match_prefix(self, req: Request) -> None:
        """Reuse cached full blocks for the longest matching prompt prefix."""
        if not (self.prefix_caching and req.prompt_tokens) or req.blocks:
            return
        hits = []
        for h in self._chain_hashes(req):
            b = self.hash_to_block.get(h)
            if b is None:
                break
            hits.append(b)
        # never reuse the very last prompt token: we need a forward pass to get logits
        while hits and len(hits) * self.block_size >= req.prompt_len:
            hits.pop()
        for b in hits:
            if self.ref[b] == 0:
                self.evictable.pop(b)
            self.ref[b] += 1
        req.blocks = hits
        req.num_computed = req.num_cached = len(hits) * self.block_size

    def cache_full_blocks(self, req: Request) -> None:
        """After a prefill chunk, publish newly filled full prompt blocks."""
        if not (self.prefix_caching and req.prompt_tokens):
            return
        full = min(req.num_computed, req.prompt_len) // self.block_size
        for i, h in enumerate(self._chain_hashes(req)[:full]):
            b = req.blocks[i]
            if b not in self.block_hash and h not in self.hash_to_block:
                self.block_hash[b] = h
                self.hash_to_block[h] = b


# ------------------------------------------------------------- the wait queue

class WaitQueue:
    """Admission order. fcfs: deque, O(1). sjf / priority: heap, O(log n).
    fair: one deque per tenant + a served-token counter per tenant (VTC-style),
    pick = the tenant with the least service, O(tenants)."""

    def __init__(self, policy: str = "fcfs"):
        assert policy in ("fcfs", "sjf", "priority", "fair")
        self.policy = policy
        self.dq: deque[Request] = deque()
        self.heap: list[tuple] = []
        self.tenants: dict[str, deque[Request]] = {}
        self.served: dict[str, float] = {}
        self._n = 0
        self._seq = 0

    def __len__(self) -> int:
        return self._n

    def _key(self, r: Request) -> tuple:
        if self.policy == "sjf":
            est = r.predicted_len if r.predicted_len is not None else r.output_len
            return (r.prompt_len + est, r.arrival)
        return (r.priority, r.arrival)

    def push(self, r: Request, front: bool = False) -> None:
        self._n += 1
        if self.policy == "fcfs":
            self.dq.appendleft(r) if front else self.dq.append(r)
        elif self.policy == "fair":
            q = self.tenants.setdefault(r.tenant, deque())
            if r.tenant not in self.served:
                # VTC: a newly active tenant starts at the lowest active counter, so
                # it can't bank credit while idle
                active = [self.served[t] for t, d in self.tenants.items() if d and t in self.served]
                self.served[r.tenant] = min(active, default=0.0)
            q.appendleft(r) if front else q.append(r)
        else:
            self._seq += 1
            # a preempted request goes back with its original key, so it is not penalised
            heapq.heappush(self.heap, (self._key(r), self._seq, r))

    def _fair_tenant(self) -> str:
        return min((t for t, d in self.tenants.items() if d), key=lambda t: self.served[t])

    def peek(self) -> Request:
        if self.policy == "fcfs":
            return self.dq[0]
        if self.policy == "fair":
            return self.tenants[self._fair_tenant()][0]
        return self.heap[0][2]

    def pop(self) -> Request:
        self._n -= 1
        if self.policy == "fcfs":
            return self.dq.popleft()
        if self.policy == "fair":
            return self.tenants[self._fair_tenant()].popleft()
        return heapq.heappop(self.heap)[2]

    def charge(self, r: Request, prefill_tokens: int, decode_tokens: int) -> None:
        if self.policy == "fair":   # VTC weights: output tokens cost 2x input tokens
            self.served[r.tenant] = self.served.get(r.tenant, 0.0) + prefill_tokens + 2 * decode_tokens


# ---------------------------------------------------------------- the scheduler

@dataclass
class StepPlan:
    scheduled: list[tuple[Request, int]]      # (request, tokens to compute this step)
    preempted: list[Request]

    @property
    def num_tokens(self) -> int:
        return sum(n for _, n in self.scheduled)


class Scheduler:
    """Iteration-level (continuous) batching with chunked prefill.

    Every step:
      1. running requests first (decodes and unfinished prefill chunks), in order;
         if KV memory runs out, preempt from the back of the running list (recompute)
      2. then admit waiting requests in policy order while the token budget,
         max_num_seqs and free blocks allow.
    Cost per step: O(R + A·log W) for R running, A admitted, W waiting
    (O(R + A) for fcfs), plus O(blocks allocated).
    """

    def __init__(self, blocks: BlockManager, policy: str = "fcfs",
                 max_num_batched_tokens: int = 2048, max_num_seqs: int = 256,
                 chunked_prefill: bool = True, watermark: float = 0.01):
        self.bm = blocks
        self.waiting = WaitQueue(policy)
        self.running: list[Request] = []
        self.token_budget = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.chunked_prefill = chunked_prefill
        self.reserve = int(watermark * blocks.num_blocks)   # keep a few blocks back when admitting

    def add(self, req: Request) -> None:
        req.state = WAITING
        self.waiting.push(req)

    def has_work(self) -> bool:
        return bool(self.running) or len(self.waiting) > 0

    def _preempt(self, victim: Request) -> None:
        self.bm.free_request(victim)
        victim.num_computed = victim.num_cached = 0   # recompute: prompt + generated so far
        victim.state = WAITING
        victim.preemptions += 1
        self.waiting.push(victim, front=True)

    def step(self) -> StepPlan:
        budget = self.token_budget
        scheduled: list[tuple[Request, int]] = []
        preempted: list[Request] = []

        # 1. running requests
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.remaining, budget)
            while not self.bm.can_allocate(req, n):
                victim = self.running.pop()          # newest admitted = least work lost
                self._preempt(victim)
                preempted.append(victim)
                if victim is req:
                    break
            else:
                self.bm.allocate(req, n)
                scheduled.append((req, n))
                budget -= n
                i += 1
                continue
            break                                    # req itself was preempted

        # 2. waiting requests (skip admission in a step that had to preempt)
        while (not preempted and len(self.waiting) and budget > 0
               and len(self.running) < self.max_num_seqs):
            req = self.waiting.peek()
            self.bm.match_prefix(req)
            n = req.remaining
            if n > budget:
                if not self.chunked_prefill:
                    break                            # whole prompt must fit this step
                n = budget
            if not self.bm.can_allocate(req, n, self.reserve):
                break                                # head of line waits for memory
            self.waiting.pop()
            self.bm.allocate(req, n)
            req.state = RUNNING
            self.running.append(req)
            scheduled.append((req, n))
            budget -= n

        return StepPlan(scheduled, preempted)

    def finish_step(self, plan: StepPlan, now: float) -> list[Request]:
        """Apply the model's output: advance counters, emit tokens, free finished requests."""
        done = []
        for req, n in plan.scheduled:
            prefill_part = min(n, max(0, req.prompt_len - req.num_computed))
            req.num_computed += n
            self.bm.cache_full_blocks(req)
            self.waiting.charge(req, prefill_part, n - prefill_part)
            if req.num_computed == req.num_tokens:          # caught up: a new token comes out
                req.num_generated += 1
                req.token_times.append(now)
                if req.first_token_time is None:
                    req.first_token_time = now
                if req.num_generated >= req.output_len:
                    req.state = FINISHED
                    req.finish_time = now
                    done.append(req)
        for req in done:
            self.running.remove(req)                        # O(R); a linked list would make it O(1)
            self.bm.free_request(req)
        return done


# ---------------------------------------------------------------- the cost model

@dataclass
class CostModel:
    """One forward step on Llama-3.1-8B BF16, one H100 SXM, at 100% of peak (a floor).

    step = max(2·P·T / C, W / BW) + KV_bytes_read / BW + overhead
    T = tokens in the step; KV read = every scheduled request's context.
    """
    params: float = 8.03e9
    weight_bytes: float = 16.06e9
    kv_bytes_per_token: float = 131072
    flops: float = 989e12
    bandwidth: float = 3.35e12
    overhead: float = 0.0          # per-step CPU/launch overhead; 0 = ideal

    def step_time(self, plan: StepPlan) -> float:
        tokens = plan.num_tokens
        if tokens == 0:
            return 0.0
        kv = sum(r.num_computed + n for r, n in plan.scheduled) * self.kv_bytes_per_token
        return (max(2 * self.params * tokens / self.flops, self.weight_bytes / self.bandwidth)
                + kv / self.bandwidth + self.overhead)

    def raw(self, tokens: int, context_tokens: int) -> float:
        return (max(2 * self.params * tokens / self.flops, self.weight_bytes / self.bandwidth)
                + context_tokens * self.kv_bytes_per_token / self.bandwidth + self.overhead)

    def kv_blocks(self, free_bytes: float = 63.9e9, block_size: int = 16) -> int:
        return int(free_bytes // (self.kv_bytes_per_token * block_size))


# ---------------------------------------------------------------- simulation

@dataclass
class Report:
    n: int
    makespan: float
    tokens_out: int
    throughput: float          # output tokens / s
    ttft_p50: float
    ttft_p99: float
    tpot_p50: float
    tpot_p99: float
    e2e_p50: float
    e2e_p99: float
    itl_p99: float             # gap between consecutive tokens, pooled over all requests
    itl_max: float
    steps: int
    preemptions: int
    goodput: float = 0.0       # requests / s that met the SLO
    slo_ok: float = 0.0        # fraction meeting the SLO
    prefix_hit: float = 0.0    # fraction of prompt tokens served from cache
    padding_waste: float = 0.0 # request-level only: fraction of slot-steps that were idle

    def row(self) -> str:
        return (f"thr {self.throughput:8.0f} tok/s | TTFT p50 {self.ttft_p50*1e3:7.1f} ms p99 {self.ttft_p99*1e3:8.1f} ms"
                f" | TPOT p50 {self.tpot_p50*1e3:5.1f} ms p99 {self.tpot_p99*1e3:5.1f} ms | ITL p99 {self.itl_p99*1e3:5.1f} max {self.itl_max*1e3:6.1f} ms"
                f" | goodput {self.goodput:5.2f} req/s ({self.slo_ok*100:3.0f}% in SLO) | preempt {self.preemptions}")


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def report(reqs: list[Request], steps: int, slo_ttft: float, slo_tpot: float, extra: dict | None = None) -> Report:
    ttft = [r.first_token_time - r.arrival for r in reqs]
    tpot = [(r.finish_time - r.first_token_time) / (r.output_len - 1) for r in reqs if r.output_len > 1]
    e2e = [r.finish_time - r.arrival for r in reqs]
    gaps = [b - a for r in reqs for a, b in zip(r.token_times, r.token_times[1:])]
    start = min(r.arrival for r in reqs)
    makespan = max(r.finish_time for r in reqs) - start
    toks = sum(r.output_len for r in reqs)
    ok = [r for r in reqs if r.first_token_time - r.arrival <= slo_ttft and
          (r.output_len <= 1 or (r.finish_time - r.first_token_time) / (r.output_len - 1) <= slo_tpot)]
    prompt = sum(r.prompt_len for r in reqs)
    return Report(len(reqs), makespan, toks, toks / makespan, pct(ttft, 50), pct(ttft, 99), pct(tpot, 50),
                  pct(tpot, 99), pct(e2e, 50), pct(e2e, 99), pct(gaps, 99), max(gaps, default=0.0), steps, sum(r.preemptions for r in reqs),
                  len(ok) / makespan, len(ok) / len(reqs), sum(r.num_cached for r in reqs) / prompt,
                  **(extra or {}))


def simulate(trace: list[Request], scheduler: Scheduler, cost: CostModel,
             slo_ttft: float = 1.0, slo_tpot: float = 0.05, log=None) -> Report:
    """Event loop: admit arrivals, run one step, advance the clock by its cost."""
    pending = deque(sorted(trace, key=lambda r: r.arrival))
    t, steps = 0.0, 0
    while pending or scheduler.has_work():
        while pending and pending[0].arrival <= t:
            scheduler.add(pending.popleft())
        if not scheduler.has_work():
            t = pending[0].arrival
            continue
        plan = scheduler.step()
        if not plan.scheduled:
            raise RuntimeError("stuck: a request cannot fit in KV memory even alone")
        dt = cost.step_time(plan)
        if log is not None:
            log(steps, t, plan, dt, scheduler)
        t += dt
        steps += 1
        scheduler.finish_step(plan, t)
    return report(trace, steps, slo_ttft, slo_tpot)


def simulate_request_level(trace: list[Request], cost: CostModel, max_batch: int,
                           max_delay: float = math.inf, slo_ttft: float = 1.0,
                           slo_tpot: float = 0.05) -> Report:
    """Static (max_delay=inf: wait for a full batch) and dynamic (time-window) batching.

    A batch is padded to its longest prompt, prefilled together, then decoded until
    its longest output finishes. Nobody joins mid-flight; finished rows sit idle.
    """
    pending = deque(sorted(trace, key=lambda r: r.arrival))
    queue: deque[Request] = deque()
    t, steps, busy_slots, idle_slots = 0.0, 0, 0, 0
    while pending or queue:
        while pending and pending[0].arrival <= t:
            queue.append(pending.popleft())
        if not queue:
            t = pending[0].arrival
            continue
        if len(queue) < max_batch and pending:
            deadline = queue[0].arrival + max_delay        # inf for static batching
            if pending[0].arrival <= deadline:              # someone can still join: wait for them
                t = max(t, pending[0].arrival)
                continue
            t = max(t, deadline)                            # window closed: run what we have
        batch = [queue.popleft() for _ in range(min(max_batch, len(queue)))]
        b = len(batch)
        p = max(r.prompt_len for r in batch)
        o = max(r.output_len for r in batch)
        t += cost.raw(b * p, b * p)                        # padded prefill
        steps += 1
        for r in batch:
            r.num_generated = 1
            r.first_token_time = t
            r.token_times.append(t)
            if r.output_len == 1:
                r.finish_time = t
        for k in range(1, o):
            t += cost.raw(b, b * (p + k))                    # padded decode step
            steps += 1
            for r in batch:
                if r.num_generated < r.output_len:
                    r.num_generated += 1
                    r.token_times.append(t)
                    busy_slots += 1
                    if r.num_generated == r.output_len:
                        r.finish_time = t
                else:
                    idle_slots += 1
    rep = report(trace, steps, slo_ttft, slo_tpot)
    rep.padding_waste = idle_slots / max(1, busy_slots + idle_slots)
    return rep


# ---------------------------------------------------------------- workloads

def poisson_trace(n: int, rate: float, seed: int = 0, prompt_median: int = 512, output_median: int = 200,
                  tenants: dict[str, float] | None = None, predictor_noise: float | None = None,
                  prefixes: list[list[int]] | None = None) -> list[Request]:
    """Poisson arrivals; lognormal prompt/output lengths (sigma 0.8), capped at 4096 / 1024."""
    rng = random.Random(seed)
    t, out = 0.0, []
    names = list(tenants) if tenants else ["a"]
    weights = list(tenants.values()) if tenants else [1.0]
    for i in range(n):
        t += rng.expovariate(rate)
        pl = min(4096, max(8, int(rng.lognormvariate(math.log(prompt_median), 0.8))))
        ol = min(1024, max(2, int(rng.lognormvariate(math.log(output_median), 0.8))))
        r = Request(i, t, pl, ol, tenant=rng.choices(names, weights)[0])
        if predictor_noise is not None:
            r.predicted_len = max(1, int(ol * rng.lognormvariate(0, predictor_noise)))
        if prefixes:
            pre = rng.choice(prefixes)
            suffix = [rng.randrange(100000, 200000) for _ in range(max(1, pl))]
            r.prompt_tokens = pre + suffix
            r.prompt_len = len(r.prompt_tokens)
        out.append(r)
    return out


def clone(trace: list[Request]) -> list[Request]:
    return [Request(r.rid, r.arrival, r.prompt_len, r.output_len, r.priority, r.tenant, r.predicted_len,
                    r.prompt_tokens) for r in trace]


TOY = [  # (arrival s, prompt, output): the 8-request trace traced on the cards
    (0.000, 12, 4), (0.000, 5, 9), (0.000, 20, 2), (0.001, 7, 6),
    (0.002, 30, 3), (0.004, 4, 8), (0.006, 16, 5), (0.010, 9, 7),
]


def toy_trace() -> list[Request]:
    return [Request(i, a, p, o) for i, (a, p, o) in enumerate(TOY)]


if __name__ == "__main__":
    cost = CostModel()
    blocks = cost.kv_blocks()
    print(f"KV blocks on the H100 (16 tokens each): {blocks:,} = {blocks*16:,} tokens")

    base = poisson_trace(1000, rate=20, seed=1)
    print("\nBatching strategies, 1,000 requests at 20 req/s (SLO: TTFT<=1 s, TPOT<=50 ms):")
    print("  static  B=32        ", simulate_request_level(clone(base), cost, 32).row())
    print("  dynamic B=32, 50 ms ", simulate_request_level(clone(base), cost, 32, 0.05).row())
    print("  continuous, no chunk", simulate(clone(base), Scheduler(BlockManager(blocks), chunked_prefill=False,
                                                                     max_num_batched_tokens=8192), cost).row())
    print("  continuous + chunked", simulate(clone(base), Scheduler(BlockManager(blocks)), cost).row())
