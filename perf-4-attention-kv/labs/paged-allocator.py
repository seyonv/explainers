"""Paged KV allocator lab: PagedAttention (Kwon et al., SOSP '23) sections 1-4.3.

Run: python3 labs/paged-allocator.py   (stdlib + numpy, CPU, a few seconds)

Prints:
  1. the paper's OPT-13B arithmetic (800 KB/token, 1.6 GB per 2048-token request, Table 1 slots)
  2. the running example's KV budget: Llama-3.1-8B BF16 on one H100 (63.94 GB free, 128 KiB/token)
  3. a saturated serving simulation on that budget with four allocators:
       (a) contiguous, reserve the max length (8,192 tokens)       ~ the paper's Orca (Max)
       (b) contiguous, reserve prompt + next power of 2 of output   ~ Orca (Pow2)
       (c) contiguous, reserve exactly prompt + output (oracle)     ~ Orca (Oracle)
       (d) paged, blocks of B = 16 tokens allocated on demand       ~ vLLM
     reporting time-averaged concurrent requests and where the KV memory goes.
  4. the exact Fig 6 block-table trace (B = 4, "Four score and seven years ago our ...").

The request lengths are SYNTHETIC: lognormal draws whose means are set to the paper's
ShareGPT means (input 161.31, output 337.99 tokens, Fig 11). They are not the ShareGPT data.
The contiguous schemes ignore allocator (external) fragmentation, so they are shown at their best.
"""
import math
import numpy as np

# ---------- 1. the paper's numbers ----------
print("== 1. The paper's OPT-13B arithmetic (section 3) ==")
opt_tok = 2 * 5120 * 40 * 2  # K and V x hidden 5120 x 40 layers x 2 bytes (FP16)
print(f"KV per token  2 x 5120 x 40 x 2 B   = {opt_tok:,} B = {opt_tok/1024:.0f} KiB (paper: 800 KB)")
print(f"2048 tokens   2048 x {opt_tok:,}     = {2048*opt_tok:,} B = {2048*opt_tok/1e9:.2f} GB"
      f" = {2048*opt_tok/2**30:.2f} GiB (paper: 1.6 GB)")
slots = 12 * 2**30 / opt_tok
print(f"Table 1: 12 GB of KV on an A100-40GB -> 12 GiB / {opt_tok:,} B = {slots:,.0f} slots (paper: 15.7K)")
print(f"  reserve-max at 2048 tokens: {slots:,.0f} / 2048 = {slots/2048:.2f} -> {int(slots//2048)} requests"
      f" (Fig 13: Orca (Max) batches 7.00)")

# ---------- 2. running example budget ----------
print("\n== 2. Running example: Llama-3.1-8B BF16 on one H100 SXM ==")
KV_TOK = 2 * 32 * 8 * 128 * 2          # 131,072 B = 128 KiB per token
FREE = 80e9 - 16.06e9                  # 63.94 GB after BF16 weights (_facts.md)
B = 16
MAXLEN = 8192
budget_tokens = int(FREE // KV_TOK)
n_blocks = int(FREE // (B * KV_TOK))
print(f"KV per token = {KV_TOK:,} B = {KV_TOK//1024} KiB; free HBM = {FREE/1e9:.2f} GB")
print(f"token slots  = {FREE:.4g} / {KV_TOK:,} = {budget_tokens:,}")
print(f"block of B={B} = {B} x 128 KiB = {B*KV_TOK:,} B = {B*KV_TOK/2**20:.0f} MiB -> {n_blocks:,} blocks"
      f" ({n_blocks*B:,} token slots)")
print(f"reserve {MAXLEN:,} tokens = {MAXLEN*KV_TOK/1e9:.3f} GB each -> {budget_tokens//MAXLEN} requests")
print(f"worst waste per paged request = {B-1} tokens = {(B-1)*KV_TOK/2**20:.3f} MiB (under one block)")

# ---------- 3. simulation ----------
MEAN_IN, MEAN_OUT, SIGMA, SEED = 161.31, 337.99, 1.0, 0
WATERMARK = 0.01   # paged admission keeps 1% of blocks free for running requests to grow into (our choice)
STEPS, WARM = 4000, 1000


def lognormal(rng, mean, n):
    mu = math.log(mean) - SIGMA**2 / 2   # E[lognormal] = exp(mu + sigma^2/2) = mean
    return np.maximum(1, np.rint(rng.lognormal(mu, SIGMA, n))).astype(int)


rng = np.random.default_rng(SEED)
POOL = 60000
ins = np.minimum(lognormal(rng, MEAN_IN, POOL), MAXLEN - 1)
outs = np.minimum(lognormal(rng, MEAN_OUT, POOL), MAXLEN - ins)
print(f"\n== 3. Saturated serving simulation ({STEPS:,} decode steps, averaged over steps {WARM:,}-{STEPS:,}) ==")
print(f"synthetic lengths, seed {SEED}: lognormal sigma {SIGMA}, sample means input {ins.mean():.1f}, "
      f"output {outs.mean():.1f} (paper's ShareGPT: {MEAN_IN}, {MEAN_OUT})")


def next_pow2(x):
    return 1 << (int(x) - 1).bit_length()


def simulate(scheme):
    """Each step: admit waiting requests while memory allows, then every running request
    appends one output token; finished requests free their memory. Queue never empties."""
    nxt = 0                 # next request id in the arrival stream
    queue = []              # preempted requests to re-admit first (paged only)
    resume = {}             # preempted id -> tokens it held (recomputed on re-admission)
    run = []                # [id, tokens_held, reservation_or_blocks]
    used = 0                # reserved token slots (contiguous) or blocks (paged)
    cap = n_blocks if scheme == "paged" else budget_tokens
    admit_cap = cap - int(WATERMARK * n_blocks) if scheme == "paged" else cap
    stats = np.zeros(5)     # concurrent, token states, reserved-future, internal, slack-in-last-block
    preempt = 0
    for step in range(STEPS):
        while True:          # admission
            rid = queue[-1] if queue else nxt
            i, o = ins[rid], outs[rid]
            start = resume.get(rid, i)           # a preempted request comes back with all its tokens
            if scheme == "max":
                need = MAXLEN
            elif scheme == "pow2":
                need = i + next_pow2(o)
            elif scheme == "oracle":
                need = i + o
            else:
                need = -(-(start + 1) // B)      # blocks for the tokens so far plus the next slot
            if used + need > admit_cap:
                break
            if queue:
                queue.pop()
            else:
                nxt += 1
            used += need
            run.append([rid, start, need, False])
        for r in run:        # one decode step
            if r[3]:
                continue
            r[1] += 1
            if scheme == "paged" and r[1] > r[2] * B:
                if used + 1 > cap:               # out of blocks: preempt the newest live request
                    victim = next(v for v in reversed(run) if not v[3] and v is not r)
                    used -= victim[2]
                    victim[3] = True
                    resume[victim[0]] = victim[1]
                    queue.append(victim[0])
                    preempt += 1
                used += 1
                r[2] += 1
        keep = []
        for r in run:
            if r[3]:
                continue
            if r[1] >= ins[r[0]] + outs[r[0]]:
                used -= r[2]
            else:
                keep.append(r)
        run = keep
        if step >= WARM:
            held = sum(r[1] for r in run)
            final = sum(ins[r[0]] + outs[r[0]] for r in run)
            if scheme == "paged":
                slack = sum(r[2] * B for r in run) - held
                stats += (len(run), held, 0, 0, slack)
            else:
                res = sum(r[2] for r in run)
                stats += (len(run), held, final - held, res - final, 0)
    stats /= STEPS - WARM
    return stats, preempt


slots_total = {"paged": n_blocks * B}
rows = []
names = {"max": "(a) reserve max 8,192", "pow2": "(b) reserve pow2 output",
         "oracle": "(c) reserve exact (oracle)", "paged": "(d) paged, B = 16"}
print(f"{'scheme':<28}{'concurrent':>11}{'tokens':>9}{'future':>9}{'internal':>10}{'last blk':>10}{'free':>8}")
for s in ["max", "pow2", "oracle", "paged"]:
    st, pre = simulate(s)
    tot = slots_total.get(s, budget_tokens)
    conc, held, fut, internal, slack = st
    free = tot - held - fut - internal - slack
    pct = [100 * v / tot for v in (held, fut, internal, slack, free)]
    rows.append((s, conc, pct, held / conc, pre))
    print(f"{names[s]:<28}{conc:>11.1f}" + "".join(f"{p:>9.1f}%" if k else f"{p:>8.1f}%" for k, p in enumerate(pct))
          )
print("columns are % of the KV pool: token states | reserved for future tokens | internal fragmentation |"
      " unfilled slots in each last block | free (too small for the next reservation, or the paged watermark)")
base = rows[0][1]
for s, conc, pct, avg_held, _ in rows:
    print(f"{names[s]:<28} {conc:7.1f} requests = {conc/base:5.1f}x reserve-max; "
          f"avg tokens held per request {avg_held:.0f}")
paged = rows[3]
slack = paged[2][3] / 100 * n_blocks * B / paged[1]
used_slots = (paged[2][0] + paged[2][3]) / 100 * n_blocks * B
print(f"check (d): {used_slots:,.0f} used slots / ({paged[3]:.0f} tokens + {slack:.1f} empty slots in the last block)"
      f" = {used_slots/(paged[3]+slack):,.1f} requests")
print(f"paged preempted {paged[4]:,} times in {STEPS:,} steps (requests restart later by recompute, section 4.5)")

# ---------- 4. Fig 6 trace ----------
print("\n== 4. Fig 6 block-table trace (B = 4). Physical IDs come from the free list; we seed it"
      " with the paper's 7, 1, 3 ==")
FB = 4
free_list = [7, 1, 3, 0, 2, 4, 5, 6]
table = []            # [physical, filled]
phys = {}             # physical block -> list of tokens


def append(tok):
    if not table or table[-1][1] == FB:
        p = free_list.pop(0)
        table.append([p, 0])
        phys[p] = []
        note = f"allocate physical block {p}"
    else:
        note = "fill reserved slot"
    table[-1][1] += 1
    phys[table[-1][0]].append(tok)
    return note


prompt = "Four score and seven years ago our".split()
for t in prompt:
    append(t)


def show(label):
    print(label)
    for lg, (p, f) in enumerate(table):
        print(f"   logical {lg} -> physical {p}  #filled {f}  {phys[p]}")


show("(1) prefill: 7-token prompt -> 2 logical blocks, 1 slot reserved")
for k, t in enumerate(["fathers", "brought"], start=2):
    note = append(t)
    show(f"({k}) decode '{t}': {note}")
held = sum(f for _, f in table)
print(f"memory held: {len(table)} blocks x {FB} = {len(table)*FB} slots for {held} tokens -> "
      f"{len(table)*FB-held} empty, always < 1 block")

# Try this:
#   B = 32 or B = 128                -> fewer, bigger blocks: more slack per request (section 7.2 ablation)
#   MAXLEN = 2048                    -> the paper's OPT max; reserve-max fits 4x more, still far below paged
#   SIGMA = 2.0                      -> a longer tail at the same means: pow2 and oracle lose more
#   WATERMARK = 0.0                  -> paged admits more greedily: slightly more requests, far more preemptions
