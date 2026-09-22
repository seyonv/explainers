"""Designing an inference engine: static vs interleaved vs disaggregated, plus a prefix cache.

A tiny discrete-event simulator of one Llama-3.1-8B (BF16) server on an H100 SXM.
Step durations come from the step-time formula at ideal peaks (100% MFU, 100% HBM
bandwidth), so every time here is a floor, not a measurement:

  prefill of n tokens      = 2 * params * n / FLOPs          (compute-bound; attention FLOPs ignored)
  one decode step, batch B = max( (weights + sum_i ctx_i * kv_per_token) / HBM_bw ,
                                  2 * params * B / FLOPs )

Arrivals, prompt lengths and output lengths are synthetic (seeded). Run: python3 inference-engine-design.py
Stdlib + numpy only; runs in a few seconds.
"""
import heapq
import numpy as np

# ---- constants from _facts.md (Llama-3.1-8B BF16 on one H100 SXM) ----
PARAMS = 8.03e9
W_BYTES = 16.06e9
KV_PER_TOK = 131072          # 2 * L * K * H * 2 bytes = 128 KiB
FLOPS = 989e12
HBM = 3.35e12
MAX_SLOTS = 32               # decode batch slots (illustrative engine setting)
NET = 50e9                   # KV transfer bandwidth prefill->decode, bytes/s (illustrative, ~400 Gb/s)


def t_prefill(n_tokens):
    return 2 * PARAMS * n_tokens / FLOPS


def t_decode(ctx_lengths):
    b = len(ctx_lengths)
    mem = (W_BYTES + sum(ctx_lengths) * KV_PER_TOK) / HBM
    comp = 2 * PARAMS * b / FLOPS
    return max(mem, comp)


def section(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)


# ---------------------------------------------------------------------------
section("1. Why one prefill stalls a whole decode step (our recomputation)")
print(f"prefill 1,000 tokens (floor)          : {t_prefill(1000)*1e3:6.1f} ms")
print(f"prefill 8,192 tokens (floor)          : {t_prefill(8192)*1e3:6.1f} ms")
for b, ctx in [(1, 0), (1, 4096), (32, 2048), (32, 4096), (32, 8192)]:
    print(f"decode step, B={b:2d}, {ctx:5d}-token contexts : {t_decode([ctx]*b)*1e3:6.1f} ms")
r = t_prefill(1000) / t_decode([4096] * 32)
print(f"1k prefill / decode step (B=32, 4k ctx) = {r:.2f}  -> every decoding user waits that many extra steps")
print(f"8k prefill / decode step (B=32, 4k ctx) = {t_prefill(8192)/t_decode([4096]*32):.1f} steps of stall")


# ---------------------------------------------------------------------------
section("2. Book numbers from Part 8 (LLaMA 3-70B on TPU v5e), re-derived")
prefill_lat, step_lat, B, G_len = 0.91, 0.019, 32, 512
# prefill servers feed P / prefill_lat sequences/s; generate servers drain B*G/(step*G_len)
ratio = prefill_lat * B / (step_lat * G_len)
print(f"prefill 8192 tok on 16 v5e @40% MFU  = 2*70e9*8192/(16*1.97e14*0.4) = "
      f"{2*70e9*8192/(16*1.97e14*0.4):.3f} s  (book: 0.91 s)")
print(f"P / 0.91 = 32 G / (0.019 * 512)  ->  P = {ratio:.2f} G   (book: P = 3G)")
P, G, Bsz = 8192, 4096, 32
print(f"sequences finishing per step = B/G = {Bsz}/{G} = {Bsz/G:.4f}")
print(f"KV tokens evicted per step   = B(P+G)/G = {Bsz}*({P}+{G})/{G} = {Bsz*(P+G)/G:.0f}  (book: 96)")


# ---------------------------------------------------------------------------
section("3. Discrete-event simulation: static vs interleaved vs disaggregated")

N_REQ = 400
RATE = 8.0                    # requests/s, Poisson (synthetic)
rng = np.random.default_rng(0)
arrivals = np.cumsum(rng.exponential(1 / RATE, N_REQ))
prompt_len = rng.integers(200, 4001, N_REQ)     # synthetic, uniform 200..4000
out_len = rng.integers(50, 501, N_REQ)          # synthetic, uniform 50..500


def stats(name, ttft, gaps, n_out, t_end, n_gpus):
    ttft = np.array(ttft) * 1e3
    gaps = np.array(gaps) * 1e3
    thr = n_out / t_end
    print(f"{name:<34} TTFT mean {ttft.mean():7.0f} ms  P99 {np.percentile(ttft,99):7.0f} ms | "
          f"gap mean {gaps.mean():5.1f} ms  P99 {np.percentile(gaps,99):5.1f}  max {gaps.max():6.1f} ms | "
          f"{thr:6.0f} tok/s  ({thr/n_gpus:5.0f}/GPU)")
    return dict(ttft_mean=ttft.mean(), ttft_p99=np.percentile(ttft, 99),
                gap_mean=gaps.mean(), gap_p99=np.percentile(gaps, 99), thr=thr)


def sim_static():
    """Collect up to MAX_SLOTS waiting requests, prefill them as one padded batch,
    decode until the longest finishes, then take the next batch."""
    t, i = 0.0, 0
    ttft, gaps, n_out = [0.0] * N_REQ, [], 0
    while i < N_REQ:
        t = max(t, arrivals[i])
        batch = [j for j in range(i, N_REQ) if arrivals[j] <= t][:MAX_SLOTS]
        i = batch[-1] + 1
        pad = max(prompt_len[j] for j in batch)
        t += t_prefill(pad * len(batch))         # padded to the longest prompt
        last = {}
        for j in batch:
            ttft[j] = t - arrivals[j]
            last[j] = t
            n_out += 1
        remaining = {j: out_len[j] - 1 for j in batch}
        ctx = {j: prompt_len[j] + 1 for j in batch}
        while any(v > 0 for v in remaining.values()):
            active = [j for j in batch if remaining[j] > 0]
            t += t_decode([ctx[j] for j in active])  # finished rows are empty slots
            for j in active:
                gaps.append(t - last[j]); last[j] = t
                remaining[j] -= 1; ctx[j] += 1; n_out += 1
    return ttft, gaps, n_out, t


def sim_continuous(disaggregated):
    """Continuous batching: free slots are refilled between decode steps.
    interleaved: prefill runs at batch 1 on the same GPU and pauses every decode.
    disaggregated: a second GPU prefills at batch 1; KV is shipped over NET."""
    t, nxt = 0.0, 0
    queue, ready = [], []            # ready: (time KV lands on decode GPU, id)
    active = {}                      # id -> [remaining, ctx, last_token_time]
    ttft, gaps, n_out = [0.0] * N_REQ, [], 0
    prefill_free = 0.0               # disaggregated prefill GPU
    done = 0
    while done < N_REQ:
        if disaggregated:
            # prefill GPU works FIFO on arrivals, independently of decode
            while nxt < N_REQ and (arrivals[nxt] <= t or not active and not ready):
                start = max(prefill_free, arrivals[nxt])
                prefill_free = start + t_prefill(prompt_len[nxt])
                ttft[nxt] = prefill_free - arrivals[nxt]   # first token comes out of prefill
                n_out += 1
                land = prefill_free + prompt_len[nxt] * KV_PER_TOK / NET
                heapq.heappush(ready, (land, nxt, prefill_free))
                nxt += 1
            if not active and ready and ready[0][0] > t:
                t = ready[0][0]
            while ready and ready[0][0] <= t and len(active) < MAX_SLOTS:
                _, j, t_first = heapq.heappop(ready)
                active[j] = [out_len[j] - 1, prompt_len[j] + 1, t_first]
        else:
            while nxt < N_REQ and arrivals[nxt] <= t:
                queue.append(nxt); nxt += 1
            if not active and not queue:
                t = arrivals[nxt]; continue
            # orchestrator prioritises prefill whenever a slot is free (book, Part 7)
            if queue and len(active) < MAX_SLOTS:
                j = queue.pop(0)
                t += t_prefill(prompt_len[j])      # everyone else's decode is paused
                ttft[j] = t - arrivals[j]; n_out += 1
                active[j] = [out_len[j] - 1, prompt_len[j] + 1, t]
                continue
        if not active:
            continue
        t += t_decode([a[1] for a in active.values()])
        for j in list(active):
            a = active[j]
            gaps.append(t - a[2]); a[2] = t
            a[0] -= 1; a[1] += 1; n_out += 1
            if a[0] <= 0:
                del active[j]; done += 1
    return ttft, gaps, n_out, t


print(f"{N_REQ} synthetic requests, Poisson {RATE}/s, prompts U[200,4000], outputs U[50,500], "
      f"{MAX_SLOTS} decode slots, ideal-peak step times")
print("TTFT = arrival -> first token; gap = time between a user's consecutive tokens (TPOT jitter)\n")
res = {}
res["static"] = stats("static batch (prefill+generate)", *sim_static(), 1)
res["interleaved"] = stats("continuous, interleaved prefill", *sim_continuous(False), 1)
res["disagg"] = stats("continuous, disaggregated (1P+1D)", *sim_continuous(True), 2)


# ---------------------------------------------------------------------------
section("4. Prefix caching: LRU block trie, 8 replicas, random vs affinity routing")

BLOCK = 16                          # tokens per cached KV block (page)
print('"I like dogs" then "I like cats": reuse 2 of 3 tokens -> compute 1/3 of the second prefill')


class BlockTrie:
    """KV prefix cache: one node per 16-token block, LRU eviction of leaves."""
    def __init__(self, cap_blocks):
        self.root, self.cap, self.n, self.clock = {"kids": {}, "parent": None}, cap_blocks, 0, 0

    def match_and_insert(self, blocks):
        self.clock += 1
        node, hit = self.root, 0
        for k, b in enumerate(blocks):
            child = node["kids"].get(b)
            if child is None:
                child = {"kids": {}, "parent": node, "key": b}
                node["kids"][b] = child
                self.n += 1
            elif hit == k:
                hit += 1
            child["t"] = self.clock
            node = child
        while self.n > self.cap:
            self._evict_lru_leaf()
        return hit

    def _evict_lru_leaf(self):
        best, stack = None, [self.root]
        while stack:
            nd = stack.pop()
            if nd is not self.root and not nd["kids"] and (best is None or nd["t"] < best["t"]):
                best = nd
            stack.extend(nd["kids"].values())
        del best["parent"]["kids"][best["key"]]
        self.n -= 1


def to_blocks(tokens):
    return [tuple(tokens[i:i + BLOCK]) for i in range(0, len(tokens) - len(tokens) % BLOCK, BLOCK)]


def prefix_demo(sys_len, n_conv=48, turns=5, replicas=8, cap_blocks=600, seed=1):
    r = np.random.default_rng(seed)
    system = list(r.integers(0, 50000, sys_len))
    convs = [list(system) for _ in range(n_conv)]
    order = [c for c in range(n_conv) for _ in range(turns)]
    r.shuffle(order)
    out = {}
    for mode in ("one replica", "8 replicas, random routing", "8 replicas, affinity routing"):
        caches = [BlockTrie(cap_blocks * (replicas if mode == "one replica" else 1))
                  for _ in range(1 if mode == "one replica" else replicas)]
        hist = [list(system) for _ in range(n_conv)]
        rr = np.random.default_rng(seed + 1)
        hit_tok = tot_tok = 0
        for c in order:
            hist[c] += list(rr.integers(0, 50000, 120))        # new user message
            if mode == "one replica":
                cache = caches[0]
            elif mode.endswith("random routing"):
                cache = caches[rr.integers(replicas)]
            else:
                cache = caches[c % replicas]                    # sticky: conversation -> replica
            blocks = to_blocks(hist[c])
            hit = cache.match_and_insert(blocks)
            hit_tok += hit * BLOCK
            tot_tok += len(hist[c])
            hist[c] += list(rr.integers(0, 50000, 250))         # model reply appended
        out[mode] = hit_tok / tot_tok
    return out


print(f"48 synthetic chats x 5 turns, shared system prompt, +120 user / +250 reply tokens per turn,")
print(f"each replica caches 600 blocks = {600*BLOCK:,} tokens (the one-replica case gets all 8x600)")
for sys_len in (1000, 4000):
    hr = prefix_demo(sys_len)
    print(f"system prompt {sys_len:5d} tokens: " +
          "  ".join(f"{k}: {v*100:4.1f}%" for k, v in hr.items()))
print("hit rate = prompt tokens whose KV was already cached / all prompt tokens")

print("\nwhere the cache lives (our recomputation for Llama-3.1-8B at 128 KiB/token):")
print(f"  free H100 HBM 63.9 GB           -> {63.9e9/KV_PER_TOK:>12,.0f} tokens")
print(f"  8xv5e host DRAM ~450 GiB (book) -> {450*2**30/KV_PER_TOK:>12,.0f} tokens")
print(f"  8xv5e HBM 128 GiB (book)        -> {128*2**30/KV_PER_TOK:>12,.0f} tokens (before weights)")

# Try this:
# 1. Burstier arrivals: replace the exponential gaps with rng.exponential(1/RATE, N_REQ) * rng.choice([0.2, 4], N_REQ, p=[.8, .2])
#    and watch static batching's P99 TTFT and the interleaved scheme's P99 gap both grow.
# 2. Longer system prompt: call prefix_demo(8000) -- the shared prefix dominates, so even random routing hits more.
# 3. Set MAX_SLOTS = 128 and RATE = 12: more users decode at once, so each prefill now stalls 128 users instead of 32.
