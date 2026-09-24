"""Many clients, one engine loop: the concurrency card's lab.

Standard library only. Run it:  python3 async_server.py   (under 1 s)

What it simulates, inside ONE replica:
  generate(req)   the API server's handler, one coroutine per client. It checks
                  backpressure (429 + Retry-After), puts the request on the
                  inbox, then awaits its own asyncio.Queue of tokens (a stream).
  engine_loop()   the ONLY code that touches the Scheduler. Each iteration it
                  drains the inbox (new requests and aborts), calls step(),
                  "runs the GPU" (a fake model: sleep for CostModel's step
                  time), then finish_step() and pushes each token to its stream.
  20 clients      19 arrive together in a burst; the burst overflows the queue
                  cap, so one gets a 429. Client 5 disconnects after 3 tokens
                  (cancellation). Client 19 arrives later, after step 30.

Times are on the simulated GPU clock: the sum of CostModel.step_time() over the
steps so far (Llama-3.1-8B BF16, one H100, 100% of peak: a floor). The loop
really sleeps that long, so the asyncio interleaving is real, but the printed
numbers do not depend on how busy your machine is.
"""
from __future__ import annotations

import asyncio
import random
import time

from scheduler import BlockManager, CostModel, Request, Scheduler

MAX_NUM_SEQS = 8          # seats in the running batch
MAX_WAITING = 10          # bounded waiting queue: more than this -> 429
DONE = None               # end-of-stream marker on a request's token queue

inbox: asyncio.Queue = asyncio.Queue()      # API server -> engine loop
streams: dict[int, asyncio.Queue] = {}      # rid -> that client's tokens
live: dict[int, Request] = {}               # rid -> Request, O(1) lookup
sched = Scheduler(BlockManager(num_blocks=30_469, block_size=16),
                  max_num_batched_tokens=2048, max_num_seqs=MAX_NUM_SEQS)
cost = CostModel()
clock = {"gpu": 0.0, "steps": 0}            # simulated GPU time, seconds
step_done = asyncio.Event()                 # pulses after every step
log: list[str] = []


class TooBusy(Exception):
    """What the HTTP layer turns into 429 Too Many Requests + Retry-After."""


async def generate(req: Request):
    """One client's request, as an async generator of tokens (SSE-style)."""
    if len(streams) >= MAX_NUM_SEQS + MAX_WAITING:     # O(1) check
        raise TooBusy("Retry-After: 1")
    q: asyncio.Queue = asyncio.Queue()
    streams[req.rid] = q
    await inbox.put(("add", req))
    try:
        while (tok := await q.get()) is not DONE:
            yield f"data: tok{tok}\n\n"                # detokenize + SSE
    finally:                                           # client went away early
        if streams.pop(req.rid, None) is not None:
            inbox.put_nowait(("abort", req.rid))       # loop frees blocks


def abort(rid: int) -> None:
    """Called only by the engine loop, between steps: no step is in flight."""
    req = live.pop(rid, None)                          # O(1)
    if req is None:                                    # already finished
        return
    if req in sched.running:
        sched.running.remove(req)                      # O(R)
        where = "running"
    else:
        sched.waiting.dq.remove(req)                   # O(W), fcfs deque
        sched.waiting._n -= 1
        where = "waiting"
    n = len(req.blocks)
    sched.bm.free_request(req)
    log.append(f"step {clock['steps']}: abort client {rid} ({where}), "
               f"freed {n} KV blocks before the next step()")


async def engine_loop(stop: asyncio.Event) -> None:
    while not stop.is_set():
        while not inbox.empty():                       # 1. drain the inbox
            kind, x = inbox.get_nowait()
            if kind == "add":
                live[x.rid] = x
                sched.add(x)
            else:
                abort(x)
        if not sched.has_work():
            await asyncio.sleep(0.001)
            continue
        plan = sched.step()                            # 2. schedule
        dt = cost.step_time(plan)
        await asyncio.sleep(dt)                        # 3. "GPU" (fake model)
        clock["gpu"] += dt
        clock["steps"] += 1
        done = sched.finish_step(plan, clock["gpu"])
        for req, _ in plan.scheduled:                  # 4. emit tokens
            if req.remaining == 1 and req.rid in streams:  # caught up
                streams[req.rid].put_nowait(req.num_generated)
        for req in done:
            live.pop(req.rid, None)
            q = streams.pop(req.rid, None)
            if q is not None:
                q.put_nowait(DONE)
        step_done.set()
        step_done.clear()


async def client(req: Request, stats: dict, after_step: int = 0,
                 disconnect_after: int | None = None) -> None:
    while clock["steps"] < after_step:
        await step_done.wait()
    t0 = clock["gpu"]
    first, n = None, 0
    try:
        agen = generate(req)
        async for _chunk in agen:
            n += 1
            if first is None:
                first = clock["gpu"] - t0
            if disconnect_after is not None and n == disconnect_after:
                await agen.aclose()                    # the socket closed
                stats[req.rid] = ("cancelled", first, n, clock["gpu"] - t0)
                return
        stats[req.rid] = ("ok", first, n, clock["gpu"] - t0)
    except TooBusy as e:
        stats[req.rid] = ("429, " + str(e), None, 0, 0.0)


async def main() -> None:
    rng = random.Random(7)
    stop = asyncio.Event()
    loop_task = asyncio.create_task(engine_loop(stop))
    stats: dict = {}
    reqs = [Request(rid=i, arrival=0.0, prompt_len=rng.randint(100, 800),
                    output_len=rng.randint(8, 40)) for i in range(20)]
    t0 = time.perf_counter()
    await asyncio.gather(*[
        client(r, stats, after_step=30 if r.rid == 19 else 0,
               disconnect_after=3 if r.rid == 5 else None) for r in reqs])
    wall = time.perf_counter() - t0
    stop.set()
    await loop_task

    print(f"{MAX_NUM_SEQS} seats + waiting cap {MAX_WAITING}; 20 clients; "
          f"{clock['steps']} engine steps; {clock['gpu'] * 1000:.1f} ms "
          f"of GPU time ({wall * 1000:.0f} ms wall)")
    print(f"{'client':>6} {'prompt':>6} {'out':>4}  {'status':<20}"
          f"{'TTFT ms':>8} {'tokens':>7} {'E2E ms':>7}")
    for r in reqs:
        st, ttft, n, e2e = stats[r.rid]
        tt = f"{ttft * 1000:.1f}" if ttft is not None else "-"
        e2 = f"{e2e * 1000:.1f}" if ttft is not None else "-"
        print(f"{r.rid:>6} {r.prompt_len:>6} {r.output_len:>4}  {st:<20}"
              f"{tt:>8} {n:>7} {e2:>7}")
    for line in log:
        print(line)
    print(f"free KV blocks at the end: {sched.bm.num_free()} of 30469 "
          f"(every block released exactly once)")


if __name__ == "__main__":
    asyncio.run(main())
