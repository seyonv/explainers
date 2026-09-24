"""Binary tree traversals: the four orders, Morris inorder, and
Binary Tree Maximum Path Sum.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md,
Trees > Binary. Morris inorder is rewritten from the source's C++;
max path sum keeps the source's iterative post-order idea.
"""
import random
import sys
from collections import deque


class Node:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def build(level):
    """LeetCode level list, e.g. [-10, 9, 20, None, None, 15, 7]."""
    nodes = [None if v is None else Node(v) for v in level]
    kids = deque(nodes[1:])
    for node in nodes:
        if node is None:
            continue
        if kids:
            node.left = kids.popleft()
        if kids:
            node.right = kids.popleft()
    return nodes[0] if nodes else None


def orders(root):
    pre, ino, post = [], [], []

    def walk(n):
        if n is None:
            return
        pre.append(n.val)    # before both children
        walk(n.left)
        ino.append(n.val)    # between them
        walk(n.right)
        post.append(n.val)   # after both

    walk(root)
    return pre, ino, post


def level_order(root):
    out, q = [], deque([root] if root else [])
    while q:
        n = q.popleft()
        out.append(n.val)
        q.extend(c for c in (n.left, n.right) if c)
    return out


def morris_inorder(root):
    out, cur = [], root
    while cur:
        if cur.left is None:          # nothing on the left
            out.append(cur.val)
            cur = cur.right           # may follow a thread
            continue
        pred = cur.left               # rightmost of left subtree
        while pred.right and pred.right is not cur:
            pred = pred.right
        if pred.right is None:        # 1st visit: lay thread
            pred.right = cur
            cur = cur.left
        else:                         # 2nd visit: remove it
            pred.right = None
            out.append(cur.val)
            cur = cur.right
    return out


def morris_trace(root):
    """Same loop, printing each step and counting pointer moves."""
    out, cur, step, moves, pred_steps = [], root, 0, 0, 0
    while cur:
        step += 1
        at = cur.val
        if cur.left is None:
            out.append(cur.val)
            what = "no left: emit, go right"
            cur = cur.right
        else:
            pred = cur.left
            pred_steps += 1
            while pred.right and pred.right is not cur:
                pred = pred.right
                pred_steps += 1
            if pred.right is None:
                what = f"pred {pred.val}: thread {pred.val}->{at}"
                pred.right, cur = cur, cur.left
            else:
                what = f"pred {pred.val}: unthread, emit"
                pred.right = None
                out.append(cur.val)
                cur = cur.right
        moves += 1
        print(f"  {step}. at {at}: {what:<28} out={out}")
    print(f"  {moves} moves + {pred_steps} predecessor steps")
    return out


def max_path_sum(root):
    best = -sys.maxsize
    gain = {None: 0}                         # best downward path
    stack = [(root, False)]
    while stack:
        node, seen = stack.pop()
        if not seen:                         # children first
            stack.append((node, True))
            stack += [(c, False) for c in (node.left, node.right) if c]
            continue
        left = max(gain[node.left], 0)       # drop negative arms
        right = max(gain[node.right], 0)
        best = max(best, node.val + left + right)  # bend here
        gain[node] = node.val + max(left, right)   # or pass one up
    return best


def brute_max_path(root):
    """Every path is a pair of endpoints: try them all via parents."""
    parent, nodes, q = {root: None}, [], deque([root])
    while q:
        n = q.popleft()
        nodes.append(n)
        for c in (n.left, n.right):
            if c:
                parent[c] = n
                q.append(c)

    def up(n):
        chain = []
        while n:
            chain.append(n)
            n = parent[n]
        return chain

    best = -sys.maxsize
    for a in nodes:
        ua = up(a)
        for b in nodes:
            ub = up(b)
            common = set(ua) & set(ub)
            path = [n for n in ua if n not in common]
            path += [n for n in ub if n not in common]
            lca = next(n for n in ua if n in common)
            best = max(best, sum(n.val for n in path) + lca.val)
    return best


def random_tree(rng, n):
    if n == 0:
        return None
    k = rng.randrange(n)
    return Node(rng.randint(-9, 9), random_tree(rng, k),
                random_tree(rng, n - 1 - k))


if __name__ == "__main__":
    t7 = build([4, 2, 6, 1, 3, 5, 7])
    pre, ino, post = orders(t7)
    print("pre:  ", pre)
    print("in:   ", ino)
    print("post: ", post)
    print("level:", level_order(t7))

    t5 = build([4, 2, 5, 1, 3])
    print("morris:", morris_inorder(t5))
    morris_trace(t5)
    print("threads removed:", t5.left.right.right is None,
          t5.left.left.right is None)

    t = build([-10, 9, 20, None, None, 15, 7])
    print("max path sum:", max_path_sum(t))

    rng = random.Random(0)
    for _ in range(1000):
        r = random_tree(rng, rng.randint(1, 9))
        shape = orders(r)
        assert morris_inorder(r) == shape[1]
        assert orders(r) == shape                # tree restored
        assert max_path_sum(r) == brute_max_path(r)
    print("1000 random trees: Morris == recursive inorder,"
          " max_path_sum == brute force")

    chain = None
    for v in range(1000):
        chain = Node(v, chain)               # left-leaning, depth 1000
    try:
        orders(chain)
    except RecursionError:
        print("recursive walk, depth 1000: RecursionError"
              f" (limit {sys.getrecursionlimit()})")
    print("morris on the same chain:", len(morris_inorder(chain)),
          "values")
