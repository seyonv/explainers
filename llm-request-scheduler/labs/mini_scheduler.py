"""A minimal continuous-batching scheduler (a toy vLLM). Stdlib only.
v1 FCFS + token budget + seat cap; v2 paged KV blocks + admission;
v3 preempt newest on OOM (recompute), chunked prefill, cancel.
Run: python3 mini_scheduler.py  (the 8-request toy trace + asserts)"""
from collections import deque
from dataclasses import dataclass


@dataclass
class Request:
    rid: int
    arrival: float           # seconds
    prompt_len: int
    output_len: int          # max new tokens; stop here
    num_computed: int = 0    # tokens whose KV is in the cache
    num_generated: int = 0   # output tokens so far
    state: str = "waiting"   # waiting|running|finished|cancelled
    first_token: float | None = None
    finish: float | None = None

    @property
    def remaining(self):     # tokens that still need a forward pass
        return self.prompt_len + self.num_generated - self.num_computed


class BlockAllocator:
    def __init__(self, num_blocks, block_size):
        self.total, self.size = num_blocks, block_size
        self.free = deque(range(num_blocks))    # the free list
        self.tables = {}                        # rid -> [block ids]

    def needed(self, r, n):                     # new blocks for n tokens
        have = len(self.tables.get(r.rid, ()))
        return max(0, -(-(r.num_computed + n) // self.size) - have)

    def can_fit(self, r, n):                    # O(1)
        return self.needed(r, n) <= len(self.free)

    def allocate(self, r, n):                   # O(blocks added)
        table = self.tables.setdefault(r.rid, [])
        for _ in range(self.needed(r, n)):
            table.append(self.free.popleft())

    def release(self, r):                       # O(blocks held)
        self.free.extend(reversed(self.tables.pop(r.rid, [])))


class Scheduler:
    def __init__(self, kv, budget=16, max_seqs=4):
        self.kv, self.budget, self.max_seqs = kv, budget, max_seqs
        self.waiting, self.running = deque(), []
        self.live = {}                          # rid -> Request
        self.plan, self.preempted = [], []

    def add(self, r):                           # O(1)
        worst = r.prompt_len + r.output_len - 1   # peak KV tokens
        if -(-worst // self.kv.size) > self.kv.total:
            raise ValueError(f"request {r.rid} can never fit")
        self.live[r.rid] = r
        self.waiting.append(r)

    def cancel(self, rid):                      # O(W) if waiting, else O(1)
        r = self.live.pop(rid, None)
        if r is not None:
            if r.state == "waiting":
                self.waiting.remove(r)
            r.state = "cancelled"               # running: _reap frees it

    def _reap(self):                            # O(R); never mid-step
        for r in self.running:
            if r.state != "running":
                self.kv.release(r)
                self.live.pop(r.rid, None)
        self.running = [r for r in self.running if r.state == "running"]

    def _preempt(self, r):                      # recompute, not swap
        self.kv.release(r)
        r.num_computed, r.state = 0, "waiting"
        self.waiting.appendleft(r)
        self.preempted.append(r)

    def step(self):                             # -> [(rid, n_tokens)]
        self._reap()
        budget, self.plan, self.preempted = self.budget, [], []
        for r in list(self.running):            # running first
            if budget == 0 or r.state != "running":
                break
            n = min(r.remaining, budget)
            while r.state == "running" and not self.kv.can_fit(r, n):
                self._preempt(self.running.pop())   # OOM: evict newest
            if r.state == "running":
                self.kv.allocate(r, n)
                self.plan.append((r, n))
                budget -= n
        while (not self.preempted and self.waiting and budget > 0
               and len(self.running) < self.max_seqs):   # admit FCFS
            r = self.waiting[0]
            n = min(r.remaining, budget)        # chunked prefill
            if not self.kv.can_fit(r, n):
                break                           # head of line waits
            self.waiting.popleft()
            self.kv.allocate(r, n)
            r.state = "running"
            self.running.append(r)
            self.plan.append((r, n))
            budget -= n
        return [(r.rid, n) for r, n in self.plan]

    def on_step_done(self, now):                # O(R)
        for r, n in self.plan:
            if r.state != "running":
                continue                        # cancelled mid-step
            r.num_computed += n
            if r.remaining == 0:                # caught up: 1 new token
                r.num_generated += 1
                if r.first_token is None:
                    r.first_token = now
                if r.num_generated == r.output_len:
                    r.state, r.finish = "finished", now
        self._reap()


def step_time(plan):     # fake model: Llama-3.1-8B, H100, 100% of peak
    toks = sum(n for _, n in plan)
    ctx = sum(r.num_computed + n for r, n in plan)
    return (max(2 * 8.03e9 * toks / 989e12, 16.06e9 / 3.35e12)
            + ctx * 131072 / 3.35e12)


def check(s):            # invariants, after every step()
    assert sum(n for _, n in s.plan) <= s.budget
    assert len(s.running) <= s.max_seqs
    held = sum(len(t) for t in s.kv.tables.values())
    assert held + len(s.kv.free) == s.kv.total
    assert not {r.rid for r in s.running} & {r.rid for r in s.waiting}
    for r, n in s.plan:
        assert len(s.kv.tables[r.rid]) * s.kv.size >= r.num_computed + n


def run(trace, s, log=lambda *a: None):     # the fake model loop
    pending, t, steps = deque(sorted(trace, key=lambda r: r.arrival)), 0, 0
    while pending or s.waiting or s.running:
        while pending and pending[0].arrival <= t:
            s.add(pending.popleft())
        if not (s.waiting or s.running):
            t = pending[0].arrival
            continue
        s.step()
        check(s)
        log(steps, t, s)
        t += step_time(s.plan)
        s.on_step_done(t)
        steps += 1
    return steps


def show(i, t, s):
    sc = " ".join(f"{r.rid}:{n}{'DP'[r.num_computed < r.prompt_len]}"
                  for r, n in s.plan)
    pre = " ".join(str(r.rid) for r in s.preempted) or "-"
    print(f"{i:>2} {t * 1e3:6.2f}ms {sc:<17} pre {pre} run "
          f"{len(s.running)} wait {len(s.waiting)} free {len(s.kv.free)}")


TOY = [(0, 12, 4), (0, 5, 9), (0, 20, 2), (1, 7, 6),     # (ms, in, out)
       (2, 30, 3), (4, 4, 8), (6, 16, 5), (10, 9, 7)]

if __name__ == "__main__":
    trace = [Request(i, a / 1e3, p, o) for i, (a, p, o) in enumerate(TOY)]
    s = Scheduler(BlockAllocator(12, 4), budget=16, max_seqs=4)
    assert run(trace, s, show) == 21
    assert all(r.num_generated == r.output_len for r in trace)
    assert len(s.kv.free) == 12 and not s.kv.tables and not s.live
    for r in trace:
        print(f"req {r.rid}: TTFT {(r.first_token - r.arrival) * 1e3:.1f}"
              f" ms, finish {r.finish * 1e3:.1f} ms")
