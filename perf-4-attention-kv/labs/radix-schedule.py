"""Cache-aware scheduling for RadixAttention: FCFS vs random vs longest-shared-prefix-first.

SGLang (arXiv 2312.07104) §3 and Appendix A.2-A.4. The radix tree and LRU leaf
eviction are copied from labs/radix-tree.py (the sibling card's lab). Each
token stands for one KV slot. A token budget plays the role of GPU memory.

Workload (illustrative, our recomputation): 64 requests = 8 few-shot templates
x 8 questions. Each template is 1,500 tokens, each question 50 tokens, and the
question is the end of the prompt (MMLU-style: prefill only, outputs ignored).
Requests run one at a time, as in the proof of Theorem 3.1.

Run: python3 labs/radix-schedule.py   (stdlib only, a few seconds)
"""
import random
from itertools import count

TEMPLATES, QUESTIONS = 8, 8
T_LEN, Q_LEN = 1500, 50
BUDGET = 3 * (T_LEN + Q_LEN)      # 4,650 slots: "a cache that holds 3 templates"
SEED = 0
FLOOR_MS_PER_TOKEN = 2 * 8.03e9 / 989e12 * 1e3   # Llama-3.1-8B BF16 on H100, 100% MFU
KV_BYTES = 2 * 32 * 8 * 128 * 2                    # 131,072 B per token


# ---------- the radix tree (copied from labs/radix-tree.py) ----------
_ids = count()


class Node:
    def __init__(self, key, parent):
        self.key, self.parent, self.children = key, parent, {}
        self.ref, self.last, self.id = 0, 0, next(_ids)


class RadixCache:
    def __init__(self, budget):
        self.root, self.budget, self.size = Node((), None), budget, 0

    def match_prefix(self, tokens, now=None, split=True):
        node, m = self.root, 0
        while m < len(tokens) and tokens[m] in node.children:
            child = node.children[tokens[m]]
            k = 0
            while k < len(child.key) and m + k < len(tokens) and child.key[k] == tokens[m + k]:
                k += 1
            if k < len(child.key):
                if not split:
                    return node, m + k
                child = self._split(child, k)
            if now is not None:
                child.last = now
            node, m = child, m + k
        return node, m

    def _split(self, child, k):
        mid = Node(child.key[:k], child.parent)
        mid.last, mid.ref = child.last, child.ref
        child.parent.children[child.key[0]] = mid
        child.key, child.parent = child.key[k:], mid
        mid.children[child.key[0]] = child
        return mid

    def insert_leaf(self, node, tokens, now):
        leaf = Node(tokens, node)
        leaf.last = now
        node.children[tokens[0]] = leaf
        self.size += len(tokens)
        return leaf

    def lock(self, node, d):
        while node is not None:
            node.ref += d
            node = node.parent

    def leaves(self):
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root and not n.children:
                out.append(n)
            stack.extend(n.children.values())
        return out

    def make_room(self, need):
        while self.budget - self.size < need:
            cands = [l for l in self.leaves() if l.ref == 0]
            if not cands:
                raise MemoryError("out of KV memory: every leaf is in use")
            v = min(cands, key=lambda n: (n.last, n.id))
            del v.parent.children[v.key[0]]
            self.size -= len(v.key)


def run_one(cache, prompt, now):
    """Prefill one request: reuse the matched prefix, compute the rest, keep it cached."""
    node, m = cache.match_prefix(prompt, now)
    rest = prompt[m:]
    if rest:
        cache.lock(node, +1)                    # the matched path can't be evicted
        cache.make_room(len(rest))
        cache.insert_leaf(node, rest, now)
        cache.lock(node, -1)
    return m


# ---------- the workload ----------
def make_requests():
    reqs = []
    for t in range(TEMPLATES):
        template = tuple(f"T{t}#{i}" for i in range(T_LEN))
        for q in range(QUESTIONS):
            reqs.append(template + tuple(f"T{t}Q{q}#{i}" for i in range(Q_LEN)))
    return reqs


def sum_edges(prompts):
    """Theorem 3.1's lower bound: build the request radix tree, sum its edge lengths."""
    tree = RadixCache(10**12)
    for now, p in enumerate(prompts):
        run_one(tree, p, now)
    return tree.size


def run_order(prompts, budget, trace=None):
    cache, hit = RadixCache(budget), 0
    for now, p in enumerate(prompts):
        m = run_one(cache, p, now)
        hit += m
        if trace is not None:
            trace.append(f"T{p[0][1:].split('#')[0]}{'+' if m else '-'}")
    return hit


def run_lspf(prompts, budget):
    """Alg 1 with one request per step: match every waiting request, run the longest match."""
    cache, waiting, hit, order = RadixCache(budget), list(range(len(prompts))), 0, []
    now = 0
    while waiting:
        best = max(waiting, key=lambda i: (cache.match_prefix(prompts[i], split=False)[1], -i))
        waiting.remove(best)
        hit += run_one(cache, prompts[best], now)
        order.append(best)
        now += 1
    return hit, order


def report(name, hit, total, bound):
    comp = total - hit
    print(f"{name:<34}: hit {hit:>6,}/{total:,} = {hit / total:5.1%}   computed {comp:>6,} tok"
          f"   prefill floor {comp * FLOOR_MS_PER_TOKEN:6.1f} ms   {hit / (total - bound):5.1%} of optimal hit rate")


reqs = make_requests()
total = sum(map(len, reqs))
bound = sum_edges(reqs)
print(f"== Batch: {TEMPLATES} templates x {QUESTIONS} questions = {len(reqs)} requests, "
      f"{T_LEN}+{Q_LEN} tokens each ==")
print(f"prompt tokens                 : {len(reqs)} x {T_LEN + Q_LEN} = {total:,}")
print(f"cache budget                  : {BUDGET:,} slots = 3 x {T_LEN + Q_LEN} "
      f"= {BUDGET * KV_BYTES / 1e6:.0f} MB of Llama-3.1-8B KV (max request {T_LEN + Q_LEN})")
print(f"lower bound sum |e| (Thm 3.1)  : {TEMPLATES} x {T_LEN} + {len(reqs)} x {Q_LEN} = {bound:,} tokens")
print(f"optimal hit rate              : 1 - {bound:,}/{total:,} = {1 - bound / total:.1%}")
print(f"prefill floor per 1,000 tokens: {FLOOR_MS_PER_TOKEN * 1000:.1f} ms  (2 x 8.03e9 x 1000 / 989e12)\n")

rng = random.Random(SEED)
arrival = reqs[:]
rng.shuffle(arrival)
fcfs_trace = []
h_fcfs = run_order(arrival, BUDGET, fcfs_trace)
report("FCFS, shuffled arrival (seed 0)", h_fcfs, total, bound)
print(f"{'':<34}  first 16 (template, + = template hit): {' '.join(fcfs_trace[:16])}")

rand_hits = []
for s in range(1, 201):
    order = reqs[:]
    random.Random(s).shuffle(order)
    rand_hits.append(run_order(order, BUDGET))
h_rand = sum(rand_hits) / len(rand_hits)
report("random schedule, mean of 200", round(h_rand), total, bound)
print(f"{'':<34}  (range {min(rand_hits) / total:.1%} to {max(rand_hits) / total:.1%})")

rr = [reqs[t * QUESTIONS + q] for q in range(QUESTIONS) for t in range(TEMPLATES)]
h_rr = run_order(rr, BUDGET)
report("FCFS, round-robin arrival", h_rr, total, bound)

h_lspf, lspf_order = run_lspf(arrival, BUDGET)
report("LSPF (= DFS), same arrivals", h_lspf, total, bound)
lspf_trace = []
assert run_order([arrival[i] for i in lspf_order], BUDGET, lspf_trace) == h_lspf
print(f"{'':<34}  first 16 (template, + = template hit): {' '.join(lspf_trace[:16])}")
print(f"LSPF computes exactly the lower bound: {total - h_lspf == bound}")

def template_prefills(trace):
    misses = [t[:-1] for t in trace if t.endswith("-")]
    return len(misses), len(misses) - len(set(misses))


for name, tr in (("FCFS", fcfs_trace), ("LSPF", lspf_trace)):
    n, rep = template_prefills(tr)
    print(f"{name} template prefills          : {n} = {n - rep} first time + {rep} recomputed after eviction "
          f"({rep} x {T_LEN} = {rep * T_LEN:,} wasted tokens)")
print("FCFS prefills per template      : " + "  ".join(
    f"{t}:{sum(1 for x in fcfs_trace if x == t + '-')}" for t in dict.fromkeys(x[:-1] for x in lspf_trace)))
print(f"LSPF template order           : {' '.join(dict.fromkeys(t[:-1] for t in lspf_trace))}")
print(f"FCFS vs LSPF computed tokens  : {total - h_fcfs:,} / {bound:,} = {(total - h_fcfs) / bound:.2f}x")

# ---------- online: arrivals over time, starvation ----------
print("\n== Online: requests arrive while others run (one request served per slot) ==")
# Illustrative: a hot template (T0) gets 70% of traffic, the other 7 share 30%; every request has
# a fresh question. 1.2 arrivals per slot for 400 slots (a burst: the queue builds), then it drains.
# Every request takes one slot whatever it hits, so reordering can't change the MEAN wait, only who
# waits: this isolates the fairness question.
def online(policy, age_cap=None, seed=1):
    rng = random.Random(seed)
    cache, queue, hit, ptok, now, qn = RadixCache(BUDGET), [], 0, 0, 0, 0
    waits = {"hot": [], "cold": []}
    while now < 400 or queue:
        if now < 400:
            for _ in range(1 + (rng.random() < 0.2)):
                t = 0 if rng.random() < 0.7 else rng.randint(1, TEMPLATES - 1)
                p = tuple(f"T{t}#{i}" for i in range(T_LEN)) + tuple(f"q{qn}#{i}" for i in range(Q_LEN))
                qn += 1
                queue.append((now, p))
        old = [i for i, (a, _) in enumerate(queue) if age_cap is not None and now - a >= age_cap]
        if policy == "fcfs":
            pick = 0
        elif old:
            pick = old[0]                            # oldest request over the cap goes first
        else:
            pick = max(range(len(queue)),
                       key=lambda i: (cache.match_prefix(queue[i][1], split=False)[1], -i))
        a, p = queue.pop(pick)
        hit, ptok = hit + run_one(cache, p, now), ptok + len(p)
        waits["hot" if p[0] == "T0#0" else "cold"].append(now - a)
        now += 1
    allw = sorted(waits["hot"] + waits["cold"])
    return (hit / ptok, sum(allw) / len(allw), max(waits["hot"]), max(waits["cold"]),
            sum(waits["cold"]) / len(waits["cold"]), len(allw))


for name, pol, cap in (("FCFS", "fcfs", None), ("greedy LSPF", "lspf", None),
                       ("LSPF + age cap 60 slots (ours)", "lspf", 60)):
    hr, mean_w, mx_hot, mx_cold, mean_cold, n = online(pol, cap)
    print(f"{name:<31}: hit rate {hr:5.1%}   mean wait {mean_w:5.1f}   max wait hot {mx_hot:3d} / "
          f"cold {mx_cold:3d} slots   cold mean {mean_cold:5.1f}  ({n} requests)")

# ---------- data parallel: 4 workers, router meta-tree ----------
print("\n== Data parallel: 4 workers, each with the same 4,650-slot cache, same shuffled arrivals ==")
W = 4


def dp(policy):
    workers = [RadixCache(BUDGET) for _ in range(W)]
    meta = [RadixCache(10**12) for _ in range(W)]     # router's view: what it sent where (evictions not seen)
    load, hit, rng = [0] * W, [0] * W, random.Random(7)
    for now, p in enumerate(arrival):
        if policy == "random":
            w = rng.randrange(W)
        else:
            match = [meta[i].match_prefix(p, split=False)[1] for i in range(W)]
            best = max(match)
            w = min((i for i in range(W) if match[i] == best), key=lambda i: load[i])
        hit[w] += run_one(workers[w], p, now)
        run_one(meta[w], p, now)
        load[w] += 1
    return sum(hit) / total, load

for name in ("random", "affinity"):
    hr, load = dp(name)
    print(f"{name + ' routing':<18}: hit rate {hr:5.1%}   computed {total - round(hr * total):>6,} tok   requests per worker {load}")
print(f"(1 worker, FCFS, same arrivals: {h_fcfs / total:.1%}; optimal: {1 - bound / total:.1%})")

# Try this:
# 1. Set BUDGET = 1 * (T_LEN + Q_LEN): FCFS collapses further, LSPF still hits the lower bound
#    (Theorem 3.1 only needs cache >= the longest request).
# 2. Set BUDGET = 15_200 (the whole request tree fits): FCFS, random and LSPF all reach 84.7%.
#    With 8 * (T_LEN + Q_LEN) = 12,400, FCFS still loses (78.6%): the 64 question leaves crowd out templates.
# 3. Change the age cap 60 to 20 or 150 in the online loop: hit rate trades against the worst wait.
