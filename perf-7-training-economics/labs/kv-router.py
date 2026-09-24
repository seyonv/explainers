"""KV-aware routing: sending requests where their cache lives.

Run: python3 labs/kv-router.py
numpy + stdlib, CPU, a few seconds.

A seeded simulation of N = 8 replicas of Llama-3.1-8B on H100s, each with an LRU
prefix cache, fed a chat workload: a few system prompts with Zipf popularity, and
per-user multi-turn conversations whose history grows every turn. Four routers:

  round-robin     cache-blind, perfectly even
  prefix hash     hash the system prompt -> always the same replica (sticky)
  session hash    hash the user id -> a conversation stays on one replica
  cost function   Dynamo-style: cost = new prefill blocks + queued prefill blocks,
                  lowest cost wins (our simplification of Dynamo's formula)

Prints the numbers on kv-aware-routing.html:
  1. The prefill floor for the 8B (course default) and the cache size
  2. The workload
  3. Per router: token hit rate, follow-up turns that land where their history is,
     max/mean load, mean and p99 TTFT, prefill GPU-seconds and $ at $3.99/GPU-hr
  4. Dynamo's worked example (18 / 10 / 11)

Everything is "our model": prefill is compute-bound at 100% of dense BF16 peak
(16.2 ms per 1,000 tokens), one FIFO prefill queue per replica, decode not modelled.
The workload shape and cache size are illustrative, not measured.
"""
import collections
import heapq
import zlib
import numpy as np

SEED = 7
rng = np.random.default_rng(SEED)

# ---- hardware and model (_facts.md)
N_PARAMS = 8.03e9
C_BF16 = 989e12                       # dense BF16 FLOP/s, H100 SXM
PRICE = 3.99                          # $/GPU-hour, Lambda 8x H100 (course default)
KV_TOK = 128 * 1024                   # bytes of KV per token, Llama-3.1-8B BF16
S_PER_TOK = 2 * N_PARAMS / C_BF16     # prefill seconds per token at peak = 16.2 us
BLOCK = 16                            # tokens per KV block (vLLM default, llm-d approx mode)

# ---- fleet (illustrative)
N_REP = 8
CACHE_GB = 40                         # KV memory each replica keeps for cached prefixes
CACHE_BLOCKS = int(CACHE_GB * 2**30 / KV_TOK) // BLOCK

# ---- workload (illustrative)
SYS_TOKENS = [2048, 1536, 3072, 1024]  # system prompts, most popular first
ZIPF_S = 1.2                           # popularity of system prompt k is 1 / k^s
N_SESSIONS = 18000
HORIZON = 600.0                        # seconds over which sessions start
MEAN_TURNS = 4                         # turns per conversation (geometric)
THINK_S = 20.0                         # mean seconds between a reply and the next turn
USER_MSG = (48, 256)                   # tokens a user types per turn (uniform)
REPLY = (128, 512)                     # tokens the model writes per turn (uniform)
CREDITS = (1.0, 4.0)                   # Dynamo overlap_score_credit values to compare
DECODE_S_PER_TOK = 0.010               # 100 tok/s per user (cost-per-token.html, B = 64 row: 101)


def blocks(tokens):
    return -(-tokens // BLOCK)


def build_workload():
    p = 1.0 / np.arange(1, len(SYS_TOKENS) + 1) ** ZIPF_S
    p /= p.sum()
    reqs = []   # (arrival_s, session, turn, sys_id, history_blocks, new_blocks, reply_tokens)
    for s in range(N_SESSIONS):
        sid = int(rng.choice(len(SYS_TOKENS), p=p))
        t = rng.uniform(0, HORIZON)
        turns = int(rng.geometric(1 / MEAN_TURNS))
        hist = 0   # conversation tokens after the system prompt already produced
        for k in range(turns):
            msg = int(rng.integers(*USER_MSG))
            reply = int(rng.integers(*REPLY))
            reqs.append((t, s, k, sid, blocks(hist), blocks(hist + msg) - blocks(hist), reply))
            hist += msg + reply
            t += rng.exponential(THINK_S)
    reqs.sort()
    return reqs, p


class Replica:
    """LRU prefix cache at segment granularity: one entry per system prompt and
    one per conversation (its cached length in blocks). A conversation's blocks
    only count as a hit if the system prompt in front of them is cached too."""

    def __init__(self):
        self.lru = collections.OrderedDict()   # key -> blocks held
        self.used = 0
        self.free_at = 0.0                     # when the FIFO prefill queue drains
        self.decoding = []                     # heap of (finish_s, kv_blocks) in decode
        self.dec_sum = 0
        self.work_tokens = 0                   # prefill tokens computed here
        self.n = 0

    def match(self, sid, sess, hist_blocks):
        if ("s", sid) not in self.lru:
            return 0
        return blocks(SYS_TOKENS[sid]) + min(self.lru.get(("u", sess), 0), hist_blocks)

    def insert(self, sid, sess, conv_blocks):
        for key, size in ((("s", sid), blocks(SYS_TOKENS[sid])), (("u", sess), conv_blocks)):
            self.used += size - self.lru.pop(key, 0)
            self.lru[key] = size
        while self.used > CACHE_BLOCKS:
            _, size = self.lru.popitem(last=False)
            self.used -= size

    def backlog_blocks(self, now):
        return max(0.0, self.free_at - now) / S_PER_TOK / BLOCK

    def decode_blocks(self, now):
        while self.decoding and self.decoding[0][0] <= now:
            self.dec_sum -= heapq.heappop(self.decoding)[1]
        return self.dec_sum


def simulate(policy, reqs, credit=1.0):
    reps = [Replica() for _ in range(N_REP)]
    rr = 0
    total = hit = 0
    ttft, follow, follow_home = [], 0, 0
    home = {}   # session -> replica that served its previous turn
    sys_tot = sys_hit = hist_tot = hist_hit = 0
    jitter = np.random.default_rng(SEED + 1)
    dec_imb = []
    for (t, sess, turn, sid, hist_b, new_b, reply) in reqs:
        d = [x.decode_blocks(t) for x in reps]
        if sum(d):
            dec_imb.append(max(d) / (sum(d) / N_REP))
        prompt_b = blocks(SYS_TOKENS[sid]) + hist_b + new_b
        if policy == "round-robin":
            r = rr % N_REP
            rr += 1
        elif policy == "prefix hash":
            r = zlib.crc32(f"system-prompt-{sid}".encode()) % N_REP
        elif policy == "session hash":
            r = zlib.crc32(f"user-{sess}".encode()) % N_REP
        else:   # cost function
            # Dynamo: prefill_load_scale * adjusted_prefill_blocks + potential_decode_blocks,
            # with overlap credit 1.0 and prefill_load_scale 1.0; queued prefill counts as prefill
            costs = [(prompt_b - credit * x.match(sid, sess, hist_b)) + x.backlog_blocks(t)
                     + x.decode_blocks(t) + jitter.random() * 1e-3 for x in reps]
            r = int(np.argmin(costs))
        x = reps[r]
        m = x.match(sid, sess, hist_b)
        new_tokens = (prompt_b - m) * BLOCK
        start = max(t, x.free_at)
        x.free_at = start + new_tokens * S_PER_TOK
        ttft.append(x.free_at - t)
        out_b = blocks(hist_b * BLOCK + new_b * BLOCK + reply) - hist_b - new_b
        heapq.heappush(x.decoding, (x.free_at + reply * DECODE_S_PER_TOK, prompt_b + out_b))
        x.dec_sum += prompt_b + out_b
        sb = blocks(SYS_TOKENS[sid])
        sys_tot += sb
        hist_tot += hist_b
        sys_hit += min(m, sb)
        hist_hit += max(0, m - sb)
        x.work_tokens += new_tokens
        x.n += 1
        x.insert(sid, sess, hist_b + new_b + out_b)   # speculative: this turn's KV will be here
        total += prompt_b
        hit += m
        if turn > 0:
            follow += 1
            follow_home += home.get(sess) == r
        home[sess] = r
    work = np.array([x.work_tokens for x in reps], float)
    n = np.array([x.n for x in reps], float)
    ttft = np.array(ttft)
    return dict(hit=hit / total, home=follow_home / follow,
                sys_hit=sys_hit / sys_tot, dec_imb=float(np.mean(dec_imb)), hist_hit=hist_hit / hist_tot,
                load=work.max() / work.mean(), nmax=n.max() / n.mean(),
                share_max=n.max() / n.sum(),
                ttft=ttft.mean(), p99=np.percentile(ttft, 99),
                prefill_tok=work.sum(), total_tok=total * BLOCK, n=len(reqs))


if __name__ == "__main__":
    print("== 1. Constants (our model)")
    print(f"prefill floor, 8B at 989 TF dense BF16: 2 x 8.03e9 x 1,000 / 989e12 = {S_PER_TOK*1e6*1000/1000:.1f} ms per 1,000 tokens")
    print(f"prefix cache per replica: {CACHE_GB} GB / 128 KiB per token = {CACHE_BLOCKS*BLOCK:,} tokens = {CACHE_BLOCKS:,} blocks of {BLOCK}")

    reqs, p = build_workload()
    span = max(r[0] for r in reqs)
    # steady state: after the first 2 minutes of ramp-up, while conversations are still starting
    steady = [r for r in reqs if 120 <= r[0] < HORIZON]
    st_tok = sum((blocks(SYS_TOKENS[r[3]]) + r[4] + r[5]) * BLOCK for r in steady)
    print("\n== 2. Workload (illustrative, seed", SEED, ")")
    print("system prompts (tokens):", SYS_TOKENS, " Zipf shares:", " ".join(f"{x:.0%}" for x in p))
    print(f"{N_SESSIONS:,} conversations, {len(reqs):,} requests over {span:.0f} s = {len(reqs)/span:.1f} req/s")
    tot_tok = sum((blocks(SYS_TOKENS[r[3]]) + r[4] + r[5]) * BLOCK for r in reqs)
    hist_tok = sum(r[4] * BLOCK for r in reqs)
    print(f"share of prompt tokens that are earlier turns of the same conversation: {hist_tok/tot_tok:.0%}")
    print(f"steady state (120-{HORIZON:.0f} s): {len(steady)/(HORIZON-120):.1f} req/s; with no cache at all the fleet "
          f"would be {st_tok*S_PER_TOK/(HORIZON-120)/N_REP:.0%} busy with prefill")
    print(f"mean prompt {tot_tok/len(reqs):,.0f} tokens; follow-up turns: {sum(r[2] > 0 for r in reqs):,}")
    print(f"round-robin: chance a follow-up turn lands on the replica holding its history = 1/N = {1/N_REP:.1%}")

    print("\n== 3. Routers, N =", N_REP, "replicas")
    print(f"{'router':<24}{'hit':>7}{'sys':>7}{'hist':>7}{'home':>7}{'prefill':>9}{'reqs':>7}{'decode':>8}"
          f"{'mean TTFT':>11}{'p99 TTFT':>10}{'GPU-s':>8}{'$':>7}")
    print(f"{'':<24}{'':>7}{'':>7}{'':>7}{'':>7}{'max/mean':>9}{'max/mn':>7}{'max/mn':>8}")
    runs = [("round-robin", "round-robin", 1.0), ("prefix hash", "prefix hash", 1.0),
            ("session hash", "session hash", 1.0)]
    runs += [(f"cost function, credit {c:g}", "cost function", c) for c in CREDITS]
    res = {}
    for name, pol, c in runs:
        s = res[name] = simulate(pol, reqs, c)
        gpu_s = s["prefill_tok"] * S_PER_TOK
        print(f"{name:<24}{s['hit']:>7.1%}{s['sys_hit']:>7.1%}{s['hist_hit']:>7.1%}{s['home']:>7.1%}"
              f"{s['load']:>9.2f}{s['nmax']:>7.2f}{s['dec_imb']:>8.2f}"
              f"{s['ttft']*1e3:>9.0f}ms{s['p99']*1e3:>8.0f}ms{gpu_s:>8.0f}{gpu_s/3600*PRICE:>7.2f}")
    print("hit = cached prompt tokens / all prompt tokens; sys / hist = the same for the system-prompt part and")
    print("the earlier-turns part; home = follow-up turns routed to the replica that served the previous turn;")
    print("decode max/mn = busiest replica's active decode KV blocks / mean, averaged over arrivals")
    print(f"prefix hash: hottest replica takes {res['prefix hash']['share_max']:.0%} of requests (fair share {1/N_REP:.1%})")

    base = res["round-robin"]
    for name in [r[0] for r in runs[1:]]:
        s = res[name]
        saved = base["prefill_tok"] - s["prefill_tok"]
        per_m = saved / s["n"] * 1e6
        print(f"{name:<24} prefill tokens avoided vs round-robin: {saved/1e6:,.1f}M of {base['prefill_tok']/1e6:,.1f}M "
              f"({saved/base['prefill_tok']:.0%}); per million requests {per_m/1e9:,.2f}B tokens x 16.2 us = "
              f"{per_m*S_PER_TOK/3600:,.1f} GPU-h = ${per_m*S_PER_TOK/3600*PRICE:,.0f}")
    print("($ at $3.99/GPU-hr and 100% of peak FLOPs; double it at 50% MFU)")

    print("\n== 4. Dynamo's worked example (router-design.md), overlap_score_credit = 1.0")
    for w, (pre, ov, dec) in {"W1": (10, 2, 10), "W2": (10, 5, 5), "W3": (10, 8, 9)}.items():
        print(f"{w}: prefill {pre} - overlap {ov} = {pre-ov}, + decode {dec} -> cost {pre-ov+dec}")

# Try this:
# 1. N_REP = 32 (same traffic, more replicas): round-robin sends 3.1% of turns home; cost function at
#    credit 4 keeps 49% home and needs 584 prefill GPU-s instead of 1,760 (session hash: 187).
# 2. ZIPF_S = 2.0 (one prompt takes 70%): prefix hash puts 71% of requests on one replica; mean TTFT 114 s.
# 3. CACHE_GB = 2: memory gets scarce. Session hash drops to 45.4% cached, 53 ms mean / 259 ms p99 TTFT
#    (every replica must hold every system prompt); cost function at credit 4 keeps 53.7%, 34 / 149 ms.
