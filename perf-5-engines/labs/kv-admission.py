"""kv-admission.py - a vLLM-V1-style scheduler under a KV-block budget.

Course 5 card: "The scheduler: KV budget, admission and preemption".
Pure simulator: stdlib only, CPU, runs in a few seconds. Nothing here is measured;
every time comes from the course-2 step-time model for Llama-3.1-8B BF16 on one H100.

What it models (our simplification of vllm/v1/core/sched/scheduler.py):
  * KV pool: (gpu_memory_utilization x 80 GB - 16.06 GB weights) / 2 MiB per 16-token block.
    (Real vLLM also subtracts activations and CUDA graphs, so it gets a little less.)
  * Every step has a token budget (max_num_batched_tokens). Running requests go first,
    in admission order: a decoding request asks for 1 token, a request still in prefill
    asks for the rest of its prompt, chunked to whatever budget is left.
  * If a running request can't get a new block, preempt the LAST running request
    (FCFS: latest arrived goes first) and retry; if the victim is the request itself, stop.
  * Preemption = recompute: free all its blocks, reset computed tokens to 0, put it at the
    FRONT of the waiting queue. When it comes back it re-prefills prompt + tokens so far.
  * Waiting requests are admitted (FCFS) only if nothing was preempted this step, while
    budget is left, running < max_num_seqs, and the blocks for this step's chunk fit.
  * A request samples a token at the end of any step that computes its last known token.
  * Prefix caching is off here (vLLM keeps freed blocks as cache, so a real recompute
    can partly hit the cache; we ignore that).

Step time (course 2):  step = KV bytes read / BW + max(2 * N * tokens / FLOPs, weights / BW)
"""
import math
import random

# ---- Running example (course facts) ----
HBM = 80e9            # H100 SXM, bytes
WEIGHTS = 16.06e9     # Llama-3.1-8B BF16
N_PARAMS = 8.03e9
KV_TOK = 131_072      # bytes of KV per token (128 KiB)
BW = 3.35e12          # HBM bytes/s
FLOPS = 989e12        # dense BF16
BLOCK = 16            # tokens per KV block (vLLM default)


def kv_blocks(util):
    kv_bytes = util * HBM - WEIGHTS
    return kv_bytes, int(kv_bytes // (KV_TOK * BLOCK))


def step_time(n_tokens, kv_tokens_read):
    t_kv = kv_tokens_read * KV_TOK / BW
    t_lin = max(2 * N_PARAMS * n_tokens / FLOPS, WEIGHTS / BW)
    return t_kv + t_lin


class Req:
    def __init__(self, rid, prompt, out):
        self.id, self.prompt, self.target = rid, prompt, out
        self.out = 0          # output tokens generated so far
        self.computed = 0     # tokens whose KV is in the cache
        self.blocks = 0
        self.preempted = 0
        self.first_token = None
        self.done = None

    def known(self):          # tokens the model has seen or must see
        return self.prompt + self.out


def simulate(n_req=200, prompt=2048, out_lo=512, out_hi=1024, util=0.92,
             max_num_seqs=256, budget=8192, seed=0, trace_every=0, label="", reserve=0, dump_step=0):
    """reserve=L (tokens): Orca-style. Admit a request only if blocks for L tokens (its
    maximum possible length) fit, and hold them all from admission on. Never preempts."""
    rng = random.Random(seed)
    reqs = [Req(i, prompt, rng.randint(out_lo, out_hi)) for i in range(n_req)]
    _, total_blocks = kv_blocks(util)
    free = total_blocks
    waiting = list(reqs)       # all arrive at t = 0, FCFS by id
    running = []
    t, step = 0.0, 0
    n_preempt, recompute_tok, first_preempt = 0, 0, None
    peak_running, max_step_ms, all_in = 0, 0.0, None
    trace = []
    step_ms = []

    def need_blocks(r, n_new):
        return max(0, math.ceil((r.computed + n_new) / BLOCK) - r.blocks)

    while waiting or running:
        step += 1
        tok_budget = budget
        sched = {}                          # request -> tokens this step  ({request_id: num_tokens})
        preempted_this_step = False
        i = 0
        while i < len(running) and tok_budget > 0:
            r = running[i]
            n_new = min(r.known() - r.computed, tok_budget)
            ok = True
            while need_blocks(r, n_new) > free:
                victim = running[-1]        # FCFS: latest admitted is preempted first
                running.pop()
                if victim in sched:
                    tok_budget += sched.pop(victim)
                free += victim.blocks
                recompute_tok += victim.computed
                victim.blocks, victim.computed = 0, 0
                victim.preempted += 1
                n_preempt += 1
                preempted_this_step = True
                if first_preempt is None:
                    first_preempt = (step, t, len(running) + 1,
                                     sum(x.out for x in running + [victim]) / (len(running) + 1),
                                     victim.id, victim.out)
                waiting.insert(0, victim)
                if victim is r:
                    ok = False
                    break
            if not ok:
                break
            if not getattr(r, "reserved", False):
                free -= need_blocks(r, n_new)
                r.blocks = math.ceil((r.computed + n_new) / BLOCK)
            sched[r] = n_new
            tok_budget -= n_new
            i += 1

        if not preempted_this_step:
            while waiting and tok_budget > 0 and len(running) < max_num_seqs:
                r = waiting[0]
                n_new = min(r.known() - r.computed, tok_budget)
                nb = (math.ceil(reserve / BLOCK) if reserve else need_blocks(r, n_new))
                if nb > free:
                    break
                waiting.pop(0)
                free -= nb
                r.blocks += nb
                if reserve:
                    r.reserved = True
                sched[r] = n_new
                tok_budget -= n_new
                running.append(r)

        n_tok = sum(sched.values())
        kv_read = sum(r.computed + n for r, n in sched.items())
        dt = step_time(n_tok, kv_read)
        if step == dump_step:
            dec = [r.id for r, n in sched.items() if n == 1 and r.computed + 1 == r.known() and r.out > 0]
            pre = {r.id: n for r, n in sched.items() if r.id not in dec}
            print(f"  step {step} schedule {{request_id: num_tokens}}: {len(dec)} decodes x 1 token "
                  f"(requests {dec[0]}..{dec[-1]}), prefill chunks {pre}; total {n_tok:,} of {budget:,}; "
                  f"free blocks after: {free:,}; step time {dt * 1e3:.1f} ms")
        t += dt
        max_step_ms = max(max_step_ms, dt * 1e3)
        step_ms.append(dt * 1e3)
        peak_running = max(peak_running, len(running))
        if all_in is None and not waiting:
            all_in = (step, t)

        for r, n in sched.items():
            r.computed += n
            if r.computed == r.known():     # caught up: sample one token
                r.out += 1
                if r.first_token is None:
                    r.first_token = t
                if r.out >= r.target:
                    r.done = t
                    free += r.blocks
                    r.blocks = 0
                    running.remove(r)
        if trace_every and (step % trace_every == 0 or step == 1):
            trace.append((step, t, len(running), len(waiting),
                          sum(1 for x in reqs if x.done), free, n_preempt, n_tok))

    out_tok = sum(r.target for r in reqs)
    ttft = sorted(r.first_token for r in reqs)
    return dict(label=label, steps=step, time=t, out_tok=out_tok, thr=out_tok / t,
                preempt=n_preempt, recompute=recompute_tok, first=first_preempt,
                peak=peak_running, ttft_p50=ttft[len(ttft) // 2], ttft_max=ttft[-1],
                max_step_ms=max_step_ms, p50_step_ms=sorted(step_ms)[len(step_ms) // 2], trace=trace, blocks=total_blocks,
                all_in=all_in, reqs_preempted=sum(1 for r in reqs if r.preempted),
                mean_out=out_tok / n_req)


def row(r):
    fp = f"{r['first'][1]:.2f} s" if r["first"] else "never"
    return (f"{r['label']:<26} {r['peak']:8d} {r['preempt']:8d} {r['reqs_preempted']:8d} "
            f"{r['recompute']:13,d} {fp:>11} {r['ttft_p50']:8.2f}s {r['p50_step_ms']:7.1f}ms "
            f"{r['max_step_ms']:7.1f}ms {r['time']:8.2f} {r['thr']:7,.0f}")


HEAD = (f"{'config':<26} {'peak run':>8} {'preempt':>8} {'reqs hit':>8} {'recompute tok':>13} "
        f"{'1st preempt':>11} {'TTFT p50':>9} {'p50 step':>9} {'max step':>9} {'total s':>8} {'tok/s':>7}")

if __name__ == "__main__":
    print("=== KV pool for Llama-3.1-8B BF16 on one H100 (our recomputation) ===")
    for util in (1.0, 0.92, 0.9, 0.5):
        kvb, nb = kv_blocks(util)
        print(f"util {util:4.2f}: {util:.2f} x 80 GB - 16.06 GB = {kvb / 1e9:6.2f} GB "
              f"-> {kvb / KV_TOK:,.0f} tokens -> {nb:,} blocks of 16 (2 MiB each)")
    kvb, nb = kv_blocks(0.92)
    for ctx in (2048, 8192):
        per = math.ceil(ctx / BLOCK)
        print(f"  at 0.92, a {ctx:,}-token sequence = {per} blocks -> max {nb // per} sequences "
              f"at full length ({nb:,} // {per})")
    reserve = math.ceil((2048 + 1024) / BLOCK)
    print(f"  reserve for max length (2,048 + 1,024 = 3,072 tokens = {reserve} blocks): "
          f"{nb // reserve} requests admitted ({nb:,} // {reserve})")
    print(f"  prompts only: 200 x 128 = {200 * 128:,} blocks; left for outputs: {nb - 200 * 128:,} blocks "
          f"= {(nb - 200 * 128) * 16:,} tokens = {(nb - 200 * 128) * 16 / 200:.0f} per request")

    print("\n=== Same weights, other GPUs, util 0.92 (vLLM's default batch knobs differ by GPU class) ===")
    for name, mem in (("A100 40 GB", 40e9), ("A100 80 GB", 80e9), ("H100 80 GB", 80e9),
                      ("H200 141 GB", 141e9), ("B200 192 GB", 192e9)):
        kvb = 0.92 * mem - WEIGHTS
        nb = int(kvb // (KV_TOK * BLOCK))
        print(f"  {name:<12} KV {kvb / 1e9:7.2f} GB -> {nb:6,} blocks -> {nb // 128:4d} x 2k, {nb // 512:4d} x 8k sequences")

    print("\n=== Step-time model (course 2): step = KV/BW + max(2*N*T/FLOPs, W/BW) ===")
    for b in (64, 200):
        print(f"  pure decode, {b} seqs at 2,048 ctx: {step_time(b, b * 2048) * 1e3:.2f} ms")
    print(f"  decode 200 + a 7,992-token prefill chunk (8,192 total): "
          f"{step_time(8192, 200 * 2048 + 7992) * 1e3:.2f} ms")
    print(f"  4 prompts of 2,048 prefilled in one 8,192-token step: {step_time(8192, 8192) * 1e3:.2f} ms "
          f"(compute {2 * N_PARAMS * 8192 / FLOPS * 1e3:.1f} ms)")

    print("\n=== Burst: 200 requests at t = 0, prompt 2,048, output U[512, 1024] (seed 0), util 0.92 ===")
    base = simulate(max_num_seqs=256, budget=8192, trace_every=100, label="seqs 256, budget 8,192",
                    dump_step=30)
    print(f"mean output length {base['mean_out']:.1f} tokens, total {base['out_tok']:,}")
    print(f"{'step':>5} {'time s':>7} {'running':>8} {'waiting':>8} {'done':>5} {'free blk':>9} {'preempt':>8} {'tok/step':>9}")
    for s_, tt, ru, wa, dn, fr, pr, nt in base["trace"]:
        print(f"{s_:5d} {tt:7.2f} {ru:8d} {wa:8d} {dn:5d} {fr:9,d} {pr:8d} {nt:9,d}")
    f = base["first"]
    print(f"first preemption: step {f[0]}, t = {f[1]:.2f} s, {f[2]} running, mean {f[3]:.0f} output tokens each; "
          f"victim = request {f[4]} (the latest admitted) with {f[5]} output tokens")
    print(f"all 200 admitted (waiting queue first empty) at step {base['all_in'][0]}, t = {base['all_in'][1]:.2f} s")

    print("\n=== Knob comparison (same burst) ===")
    print(HEAD)
    for seqs in (64, 256):
        for bud in (2048, 8192):
            print(row(simulate(max_num_seqs=seqs, budget=bud, label=f"seqs {seqs}, budget {bud:,}")))
    print(row(simulate(max_num_seqs=256, budget=8192, reserve=3072, label="reserve 3,072 (Orca)")))
    print(row(simulate(max_num_seqs=256, budget=8192, reserve=8192, label="reserve 8,192 (Orca)")))

    print("\n=== When more sequences stops helping: prompt 512, output U[2048, 4096], util 0.5 ===")
    print(HEAD)
    for seqs in (64, 128, 256):
        print(row(simulate(prompt=512, out_lo=2048, out_hi=4096, util=0.5, max_num_seqs=seqs,
                           budget=8192, label=f"util 0.5, seqs {seqs}")))

# Try this:
#  1. prompt=8192 in the burst: the pool holds only 53 full 8k sequences, so any max_num_seqs above ~53
#     just adds preemptions.
#  2. out_lo=1: many answers finish early, the pool never runs dry, and nothing is preempted.
#  3. budget=16384 (the LLM-class default on an H100): faster admission, longer worst steps.
