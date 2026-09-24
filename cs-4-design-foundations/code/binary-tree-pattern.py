"""The binary tree as a design pattern.

Source: ljeng/cheat-sheet, large-scale-design/system-design/
binary-trees.md. Four examples: a DNS hierarchy, a BST index, a
credit decision tree and a Huffman code A=0, B=10, C=110, D=111.
The source gives only "high / medium / low" frequencies; the counts
A:5, B:2, C:1, D:1 and the message ABACABADA are illustrative.

This file prints every number on the card:
  1. a Huffman tree with heapq, its leaf depths, canonical codes
  2. the encoded message and its bit count vs fixed width
  3. BST height: sorted inserts vs a balanced tree vs a B-tree
  4. why deep trees want iterative code (RecursionError)
  5. the source's credit decision tree
"""
import heapq
import math
import random
import sys
from collections import Counter


# 1. Huffman: merge the two lightest subtrees until one is left.
def huffman_tree(freq, log=None):
    heap = [(f, i, s) for i, (s, f) in enumerate(freq.items())]
    heapq.heapify(heap)
    tick = len(heap)                 # tie-breaker: never compare
    while len(heap) > 1:             # two subtrees directly
        f1, _, a = heapq.heappop(heap)   # lightest
        f2, _, b = heapq.heappop(heap)   # next lightest
        if log is not None:              # for the hand trace
            log.append((f1, a, f2, b))
        heapq.heappush(heap, (f1 + f2, tick, (a, b)))
        tick += 1
    return heap[0][2]                # leaf = symbol, node = (l, r)


def code_lengths(tree):             # iterative: no recursion limit
    out, stack = {}, [(tree, 0)]
    while stack:
        node, d = stack.pop()
        if isinstance(node, tuple):
            stack += [(node[0], d + 1), (node[1], d + 1)]
        else:
            out[node] = d
    return out


def leaves(t):
    return t if isinstance(t, str) else leaves(t[0]) + leaves(t[1])


def canonical(lengths):           # codes from lengths, as DEFLATE
    code, prev, out = 0, 0, {}
    for s in sorted(lengths, key=lambda s: (lengths[s], s)):
        code <<= lengths[s] - prev       # step down a level
        out[s] = format(code, f"0{lengths[s]}b")
        prev, code = lengths[s], code + 1
    return out


def decode(bits, codes):
    back = {c: s for s, c in codes.items()}
    out, cur = [], ""
    for b in bits:
        cur += b                             # walk one edge down
        if cur in back:                      # reached a leaf
            out.append(back[cur])
            cur = ""
    return "".join(out)


# 3. BST height: naive inserts vs balanced.
def bst_height(keys):
    """Height (levels) of a naive BST, built without recursion."""
    left, right, root = {}, {}, None
    height = 0
    for k in keys:
        if root is None:
            root, height = k, 1
            continue
        node, depth = root, 1
        while True:
            depth += 1
            side = left if k < node else right   # key < node.key
            if node not in side:
                side[node] = k
                break
            node = side[node]
        height = max(height, depth)
    return height


def balanced_height(n):
    return math.ceil(math.log2(n + 1))


# 4. A chain of n nodes: recursive vs iterative depth.
def chain(n):
    root = None
    for _ in range(n):
        root = (root,)                       # one child: a linked list
    return root


def depth_rec(t):
    return 0 if t is None else 1 + depth_rec(t[0])


def depth_iter(t):
    d = 0
    while t is not None:
        t, d = t[0], d + 1
    return d


# 5. The source's credit decision tree.
def credit(score, income, debt_ratio):
    if not score > 650:
        return "Decline"
    if not income > 50_000:
        return "Decline"
    if debt_ratio < 0.40:
        return "Approve"
    return "Review manually"


if __name__ == "__main__":
    msg = "ABACABADA"                        # illustrative
    freq = Counter(msg)
    log = []
    lengths = code_lengths(huffman_tree(freq, log))
    print("frequencies", dict(freq))
    for f1, a, f2, b in log:
        a, b = leaves(a), leaves(b)
        print(f"  merge {a}:{f1} + {b}:{f2} -> {a + b}:{f1 + f2}")
    print("lengths", dict(sorted(lengths.items())))
    codes = canonical(lengths)
    print("codes", codes)
    bits = "".join(codes[s] for s in msg)
    print("encoded", " ".join(codes[s] for s in msg))
    terms = " + ".join(f"{freq[s]}*{lengths[s]}" for s in codes)
    print(f"huffman bits = {terms} = {len(bits)}")
    fixed = math.ceil(math.log2(len(freq))) * len(msg)
    print(f"fixed width  = 2 * {len(msg)} = {fixed}")
    n = len(msg)
    h = -sum(f / n * math.log2(f / n) for f in freq.values())
    print(f"avg {len(bits) / n:.3f} bits/symbol,"
          f" entropy {h:.3f}, bound {h * n:.2f} bits")
    print("decoded", decode(bits, codes))

    print()
    n = 1000                                 # naive inserts are O(n^2)
    random.seed(1)
    keys = random.sample(range(n), n)
    print(f"n = {n:,}: sorted inserts -> height",
          bst_height(range(n)))
    print("  random order -> height", bst_height(keys), "(seed 1)")
    print("  balanced -> height", balanced_height(n))
    n = 10**6
    print(f"n = {n:,}: sorted inserts -> height {n:,} (a list)")
    print("  balanced -> height", balanced_height(n),
          f"(log2 n = {math.log2(n):.2f})")
    print("  B-tree, fanout 100 -> levels",
          math.ceil(math.log(n, 100) - 1e-9))

    print()
    print("recursion limit", sys.getrecursionlimit())
    t = chain(5000)
    print("iterative depth", depth_iter(t))
    try:
        depth_rec(t)
    except RecursionError as e:
        print("recursive depth: RecursionError:", e)

    print()
    for app in [(700, 60_000, 0.30), (700, 60_000, 0.45),
                (700, 40_000, 0.10), (600, 90_000, 0.10)]:
        print(app, "->", credit(*app))
