"""Fibonacci heap: insert, find_min, extract_min, decrease_key.

Source: ljeng/cheat-sheet, coding-algorithms/code/FibonacciHeap.java.
That file is Java and does not compile (Node<T> vs Node<E>, `z` and
`d` undefined, `return currents`, `current` declared twice). Its root
loop `while (current != min)` starts at min, so it never runs, and its
degree array has floor(log2 n) slots, too few. This is a corrected
Python rewrite following CLRS 3rd ed., chapter 19.
"""
import heapq
import math
import random

PHI = (1 + math.sqrt(5)) / 2


class Node:
    __slots__ = ("key", "parent", "child", "left", "right",
                 "degree", "mark")

    def __init__(self, key):
        self.key, self.parent, self.child = key, None, None
        self.left = self.right = self      # circular list of one
        self.degree, self.mark = 0, False


def _splice(a, x):
    """Put x (a single node) into a's circular list, right of a."""
    x.left, x.right = a, a.right
    a.right.left = x
    a.right = x


def _unlink(x):
    x.left.right, x.right.left = x.right, x.left
    x.left = x.right = x


def _siblings(x):
    out, y = [x], x.right
    while y is not x:
        out.append(y)
        y = y.right
    return out


class FibHeap:
    def __init__(self):
        self.min, self.n = None, 0
        self.links = self.cuts = 0         # counters for the card

    def insert(self, key):                 # O(1): just a new root
        x = Node(key)
        if self.min is None:
            self.min = x
        else:
            _splice(self.min, x)
            if x.key < self.min.key:
                self.min = x
        self.n += 1
        return x                           # handle for decrease_key

    def find_min(self):
        return self.min.key

    def extract_min(self):
        z = self.min
        if z.child:                        # children become roots
            for c in _siblings(z.child):
                _unlink(c)
                c.parent = None
                _splice(z, c)
        if z.right is z:
            self.min = None
        else:
            self.min = z.right
            _unlink(z)
            self._consolidate()
        self.n -= 1
        return z.key

    def _consolidate(self):
        size = int(math.log(self.n, PHI)) + 1   # floor(log_phi n)+1
        by_deg = [None] * size
        for x in _siblings(self.min):      # snapshot the root list
            while by_deg[x.degree] is not None:
                y = by_deg[x.degree]
                by_deg[x.degree] = None
                if y.key < x.key:
                    x, y = y, x
                self._link(y, x)           # larger key goes under
            by_deg[x.degree] = x
        roots = [x for x in by_deg if x is not None]
        self.min = min(roots, key=lambda r: r.key)

    def _link(self, y, x):
        _unlink(y)
        if x.child is None:
            x.child = y
        else:
            _splice(x.child, y)
        y.parent, y.mark = x, False
        x.degree += 1
        self.links += 1

    def decrease_key(self, x, key):
        assert key <= x.key
        x.key = key
        p = x.parent
        if p is not None and x.key < p.key:
            self._cut(x, p)
            while p.parent is not None:    # cascading cut
                if not p.mark:
                    p.mark = True          # first child lost: mark
                    break
                q = p.parent               # second child lost: cut
                self._cut(p, q)
                p = q
        if x.key < self.min.key:
            self.min = x

    def _cut(self, x, p):
        if p.child is x:
            p.child = None if x.right is x else x.right
        _unlink(x)
        p.degree -= 1
        x.parent, x.mark = None, False
        _splice(self.min, x)
        self.cuts += 1


# ---------- helpers for the card ----------

def show(h):
    """Root list as nested tuples: (key, [children]); * = marked."""
    def tree(x):
        k = f"{x.key}{'*' if x.mark else ''}"
        if x.child is None:
            return k
        return (k, [tree(c) for c in _siblings(x.child)])
    return [tree(r) for r in _siblings(h.min)] if h.min else []


def source_slots(n):
    return int(math.floor(math.log(n) / math.log(2)))


def dijkstra_fib(graph, s):
    h, dist, node, done = FibHeap(), {s: 0}, {}, set()
    node[s] = h.insert((0, s))
    dks = 0
    while h.min is not None:
        d, u = h.extract_min()
        done.add(u)
        for v, w in graph[u]:
            nd = d + w
            if v in done:
                continue
            if v not in dist:
                dist[v] = nd
                node[v] = h.insert((nd, v))
            elif nd < dist[v]:
                dist[v] = nd
                h.decrease_key(node[v], (nd, v))
                dks += 1
    return dist, dks


def dijkstra_heapq(graph, s):
    dist, pq, pushes = {s: 0}, [(0, s)], 1
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue                       # stale entry, skip
        for v, w in graph[u]:
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
                pushes += 1
    return dist, pushes


if __name__ == "__main__":
    # 1. The worked example: the course array plus 8, 3, 7
    keys = [5, 2, 9, 1, 5, 6, 8, 3, 7]
    h = FibHeap()
    handles = [h.insert(k) for k in keys]
    print("after 9 inserts:", show(h), "min", h.find_min())
    print("extract_min ->", h.extract_min(), "links", h.links)
    print("after consolidate:", show(h))
    print("source's slots for n=9:", source_slots(9),
          "  slots needed:", int(math.log(9, PHI)) + 1,
          "  degree used:", h.min.degree)

    # handles[6] is 8, handles[8] is 7, handles[5] is 6
    for i, new in [(6, 4), (8, 1), (5, 0)]:
        print(f"decrease {keys[i]} -> {new}")
        h.decrease_key(handles[i], new)
        print("  ", show(h), "min", h.find_min(), "cuts", h.cuts)

    out = []
    while h.min is not None:
        out.append(h.extract_min())
    print("drain:", out)

    # 2. Random test against heapq: interleaved insert / extract_min
    rng = random.Random(0)
    for _ in range(1000):
        fh, ref = FibHeap(), []
        for _ in range(rng.randint(1, 60)):
            if ref and rng.random() < 0.4:
                assert fh.extract_min() == heapq.heappop(ref)
            else:
                k = rng.randint(0, 50)
                fh.insert(k)
                heapq.heappush(ref, k)
        while ref:
            assert fh.extract_min() == heapq.heappop(ref)
    print("1000 random insert/extract runs match heapq")

    # 3. decrease_key against a brute-force dict (keys are unique
    #    (key, id) pairs so each extracted key names one node)
    for _ in range(1000):
        fh, live, ids = FibHeap(), {}, 0
        for _ in range(rng.randint(1, 80)):
            r = rng.random()
            if live and r < 0.3:
                k = fh.extract_min()
                assert k == min(x.key for x in live.values())
                del live[k[1]]
            elif live and r < 0.6:
                x = rng.choice(list(live.values()))
                x_new = (x.key[0] - rng.randint(0, 20), x.key[1])
                fh.decrease_key(x, x_new)
            else:
                ids += 1
                live[ids] = fh.insert((rng.randint(0, 50), ids))
        while live:
            k = fh.extract_min()
            assert k == min(x.key for x in live.values())
            del live[k[1]]
    print("1000 random runs with decrease_key match brute force")

    # 4. Dijkstra both ways on one random graph
    V, E = 1000, 10000
    g = {u: [] for u in range(V)}
    for _ in range(E):
        u, v = rng.randrange(V), rng.randrange(V)
        g[u].append((v, rng.randint(1, 100)))
    d1, dks = dijkstra_fib(g, 0)
    d2, pushes = dijkstra_heapq(g, 0)
    assert d1 == d2
    print(f"dijkstra V={V} E={E}: same distances, reached {len(d1)}")
    print(f"  fib heap: {len(d1)} inserts, {dks} decrease_keys")
    print(f"  heapq:    {pushes} pushes (lazy deletion)")
