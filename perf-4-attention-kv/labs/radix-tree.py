"""RadixAttention in miniature: a radix tree of KV caches with LRU leaf eviction.

Replays a trace shaped like Figure 3 of the SGLang paper (arXiv 2312.07104):
two chat sessions, a few-shot query, a batch of few-shot queries, a chat
revisit, and self-consistency samples. The tree holds token ids; each token
stands for one KV slot (SGLang's page size is one token). A token budget plays
the role of GPU memory, shared by cached tokens and running requests.

Token counts are illustrative (chosen so the evictions land where Figure 3's
caption puts them). Run: python3 labs/radix-tree.py   (stdlib only, < 1 s)
"""
from itertools import count

BUDGET = 140          # KV slots (tokens) in the pool: cache + running requests


def seg(name, n):
    """n distinct 'tokens' belonging to one text segment, e.g. sys#0..sys#19."""
    return [f"{name}#{i}" for i in range(n)]


# ---------- the radix tree ----------
_ids = count()


class Node:
    def __init__(self, key, parent):
        self.key = key            # the edge label: a run of tokens
        self.parent = parent
        self.children = {}        # first token of child's key -> child
        self.ref = 0              # running requests using this node
        self.last = 0             # time of last access (for LRU)
        self.id = next(_ids)      # tie-break: older node first


class RadixCache:
    def __init__(self, budget):
        self.root = Node([], None)
        self.budget = budget
        self.size = 0             # tokens held by the tree
        self.evicted = []

    def match_prefix(self, tokens, now):
        """Walk down as far as tokens match; split an edge if we stop inside it."""
        node, m = self.root, 0
        node.last = now
        while m < len(tokens) and tokens[m] in node.children:
            child = node.children[tokens[m]]
            k = 0
            while k < len(child.key) and m + k < len(tokens) and child.key[k] == tokens[m + k]:
                k += 1
            if k < len(child.key):            # stopped inside the edge: split it
                child = self._split(child, k)
            child.last = now
            node, m = child, m + k
        return node, m

    def _split(self, child, k):
        mid = Node(child.key[:k], child.parent)
        mid.last, mid.ref = child.last, child.ref
        child.parent.children[child.key[0]] = mid
        child.key = child.key[k:]
        child.parent = mid
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
        """Evict least-recently-used unreferenced leaves until `need` slots are free."""
        while self.budget - self.size < need:
            cands = [l for l in self.leaves() if l.ref == 0]
            if not cands:
                raise MemoryError("out of KV memory: every leaf is in use")
            v = min(cands, key=lambda n: (n.last, n.id))
            del v.parent.children[v.key[0]]
            self.size -= len(v.key)
            self.evicted.append(label(v.key))


def label(tokens):
    """sys#0..sys#19 -> 'sys[20]'; consecutive tokens of one segment are grouped."""
    parts, prev, n = [], None, 0
    for t in tokens:
        s = t.split("#")[0]
        if s != prev and prev is not None:
            parts.append(f"{prev}[{n}]")
            n = 0
        prev, n = s, n + 1
    parts.append(f"{prev}[{n}]")
    return " ".join(parts)


def show(node, now, depth=0):
    for c in sorted(node.children.values(), key=lambda n: n.id):
        tag = "  <- used now" if c.last == now else ""
        print("    " + "  " * depth + "+- " + label(c.key) + f"  ({len(c.key)} tok, ref {c.ref})" + tag)
        show(c, now, depth + 1)


# ---------- the trace (Figure 3's nine steps; step 1 is the empty tree) ----------
SYS = seg("sys", 20)                                    # "You are a helpful assistant."
C1T1 = seg("hello", 5), seg("hi", 5)                    # chat 1, turn 1
C1T2 = seg("solve", 10), seg("sure", 41)                # chat 1, turn 2 (long answer)
C2T1 = seg("whatcan", 6), seg("ican", 14)               # chat 2, turn 1
C2T2 = seg("story", 6), seg("storyans", 34)             # chat 2, turn 2 (long story)
EX = seg("examples", 24)                                # few-shot examples Q1 A1 Q2 A2
Q = {w: (seg("q" + w, 4), seg("a" + w, 4)) for w in ("What", "When", "How")}
C1T3 = seg("howabout", 4), seg("itis", 6)               # chat 1, turn 3

TRACE = [
    ("(2) chat 1, turn 1", [(SYS + C1T1[0], C1T1[1])]),
    ("(3) chat 1, turn 2", [(SYS + sum(C1T1, []) + C1T2[0], C1T2[1])]),
    ("(4) chat 2, turn 1", [(SYS + C2T1[0], C2T1[1])]),
    ("(5) chat 2, turn 2", [(SYS + sum(C2T1, []) + C2T2[0], C2T2[1])]),
    ("(6) few-shot query", [(EX + Q["What"][0], Q["What"][1])]),
    ("(7) few-shot batch", [(EX + Q[w][0], Q[w][1]) for w in ("When", "How")]),
    ("(8) chat 1, turn 3", [(SYS + sum(C1T1, []) + sum(C1T2, []) + C1T3[0], C1T3[1])]),
    ("(9) 3 self-consistency samples",
     [(EX + Q["What"][0], seg(f"sample{i}", 8)) for i in (1, 2, 3)]),
]


def replay(budget, verbose):
    cache = RadixCache(budget)
    prompt_tok = hit_tok = 0
    if verbose:
        print("(1) empty tree")
    for now, (name, batch) in enumerate(TRACE, start=2):
        cache.evicted = []
        running, step_p, step_h = [], 0, 0
        for prompt, out in batch:                      # the batch runs together
            node, m = cache.match_prefix(prompt, now)
            cache.lock(node, +1)
            rest = prompt[m:] + out
            cache.make_room(len(rest))                 # may evict LRU leaves
            leaf = cache.insert_leaf(node, rest, now)
            cache.lock(node, -1)
            cache.lock(leaf, +1)
            running.append(leaf)
            step_p, step_h = step_p + len(prompt), step_h + m
        if verbose:
            print(f"{name}: prompt {step_p} tok, cached {step_h}, "
                  f"tree {cache.size}/{budget} tok while running")
            if cache.evicted:
                print("    evicted (LRU leaves): " + "; ".join(cache.evicted))
            show(cache.root, now)
        for leaf in running:                           # requests finish
            cache.lock(leaf, -1)
        prompt_tok, hit_tok = prompt_tok + step_p, hit_tok + step_h
    return prompt_tok, hit_tok


# ---------- the same trace under simpler caches (no memory limit) ----------
def exact_match_hits():
    seen, p, h = set(), 0, 0
    for _, batch in TRACE:
        for prompt, out in batch:
            p += len(prompt)
            h += len(prompt) if tuple(prompt) in seen else 0
            seen.add(tuple(prompt))
    return p, h


def block_hash_hits(block):
    """vLLM-style: hash(parent hash, tokens in block); only full blocks are cached."""
    cached, p, h = set(), 0, 0
    for _, batch in TRACE:
        for prompt, out in batch:
            p += len(prompt)
            parent, i = None, 0
            while (i + 1) * block <= len(prompt):
                key = hash((parent, tuple(prompt[i * block:(i + 1) * block])))
                if key not in cached:
                    break
                parent, i = key, i + 1
            h += i * block
            seq, parent = prompt + out, None
            for j in range(len(seq) // block):
                parent = hash((parent, tuple(seq[j * block:(j + 1) * block])))
                cached.add(parent)
    return p, h


print(f"== Figure-3-like replay, pool of {BUDGET} KV slots (1 slot = 1 token) ==")
p, h = replay(BUDGET, verbose=True)
print(f"\nhit rate = cached prompt tokens / prompt tokens = {h}/{p} = {h / p:.1%}")

print("\n== Same trace, other caches (hit rate) ==")
pu, hu = replay(10**9, verbose=False)
pe, he = exact_match_hits()
BLOCK = 16
pb, hb = block_hash_hits(BLOCK)
print(f"no cache                        : 0/{p} = 0.0%")
print(f"exact-match prompt cache        : {he}/{pe} = {he / pe:.1%}")
print(f"block-hash prefix cache, {BLOCK:>2}-tok: {hb}/{pb} = {hb / pb:.1%}  (no memory limit)")
print(f"radix tree, pool {BUDGET:<3}           : {h}/{p} = {h / p:.1%}")
print(f"radix tree, no memory limit     : {hu}/{pu} = {hu / pu:.1%}")

print("\n== Running example: 2,000-token system prompt, Llama-3.1-8B BF16 on H100 ==")
kv_tok = 2 * 32 * 8 * 128 * 2                          # 2*L*K*H*bytes = 131,072 B
prefix = 2000
kv = prefix * kv_tok
prefill = 2 * 8.03e9 * prefix / 989e12
free = 80e9 - 16.06e9
print(f"KV of the prefix      : {prefix} x {kv_tok:,} B = {kv / 1e6:.1f} MB")
print(f"prefill skipped / hit : 2 x 8.03e9 x {prefix} / 989e12 = {prefill * 1e3:.1f} ms (100% MFU floor)")
print(f"prefixes in 10% of free HBM: {0.1 * free / 1e9:.2f} GB / {kv / 1e6:.1f} MB = {0.1 * free / kv:.1f}")
print(f"prefixes in all free HBM   : {free / 1e9:.1f} GB / {kv / 1e6:.1f} MB = {free / kv:.0f}")
users = 100
print(f"{users} chats on one prompt: private copies {users * kv / 1e9:.1f} GB vs shared {kv / 1e9:.3f} GB")

print("\n== Paper's setup for comparison: Llama-2-7B FP16 on one A10G (24 GB) ==")
kv7 = 2 * 32 * 32 * 128 * 2                            # MHA: 32 KV heads
w7 = 6.74e9 * 2
free7 = 24e9 - w7
print(f"KV/token {kv7:,} B; prefix KV {prefix * kv7 / 1e9:.2f} GB; free after weights "
      f"{free7 / 1e9:.1f} GB; prefixes in 10% of it: {0.1 * free7 / (prefix * kv7):.1f}")
print(f"RadixAttention overhead (paper, ShareGPT, no reuse): 0.2 s / 74.3 s = {0.2 / 74.3:.2%}")

# Try this:
# 1. Set BUDGET = 10**9: nothing is evicted, and step (8) reuses all of chat 1 (hit rate jumps).
# 2. Set BUDGET = 120: steps (6) and (7) now evict chat nodes early, and step (8) reuses only the system prompt (61.0%).
# 3. Set BLOCK = 1: with 1-token blocks the hash chain finds the same prefixes as the radix tree (77.7%).
