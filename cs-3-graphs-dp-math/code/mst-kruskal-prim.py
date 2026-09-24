"""Minimum spanning trees: Kruskal and Prim.

Card: cs-3-graphs-dp-math/mst-kruskal-prim.html
Source: ljeng/cheat-sheet, coding-algorithms/graphs.md, Connectivity.
The source gives only the docstrings of Graph.kruskal() and
Graph.prim() (code/graph.py is missing), so both are written here
from CLRS 4th ed., ch. 21. Graph: the course's running example made
undirected, plus a sixth vertex F. Runs under python3 (3.10).
"""
from collections import defaultdict
from itertools import combinations
import heapq
import random

EDGES = [("A", "B", 4), ("A", "C", 2), ("C", "B", 1), ("B", "D", 5),
         ("C", "D", 8), ("C", "E", 10), ("D", "E", 2), ("C", "F", 9),
         ("E", "F", 3)]


def kruskal(vertices, edges):
    parent = {v: v for v in vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving
            x = parent[x]
        return x

    tree = []
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        ru, rv = find(u), find(v)
        if ru != rv:                 # different pieces: no cycle
            parent[ru] = rv          # merge the two pieces
            tree.append((u, v, w))
    return tree


def prim(edges, start):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((w, u, v))
        adj[v].append((w, v, u))
    seen, tree = {start}, []
    heap = list(adj[start])          # edges leaving the tree
    heapq.heapify(heap)
    while heap:
        w, u, v = heapq.heappop(heap)
        if v in seen:                # both ends inside: stale
            continue
        seen.add(v)
        tree.append((u, v, w))
        for e in adj[v]:
            if e[2] not in seen:
                heapq.heappush(heap, e)
    return tree


# ---- instrumented versions for the card's traces ----

def kruskal_trace(vertices, edges):
    parent = {v: v for v in vertices}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    rows = []
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        ru, rv = find(u), find(v)
        ok = ru != rv
        if ok:
            parent[ru] = rv
        groups = defaultdict(list)
        for x in vertices:
            groups[find(x)].append(x)
        pieces = sorted("".join(sorted(g)) for g in groups.values())
        rows.append((u, v, w, ru, rv, ok, pieces))
    return rows


def prim_trace(edges, start):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((w, u, v))
        adj[v].append((w, v, u))
    seen, rows = {start}, []
    heap = list(adj[start])
    heapq.heapify(heap)
    while heap:
        w, u, v = heapq.heappop(heap)
        stale = v in seen
        if not stale:
            seen.add(v)
            for e in adj[v]:
                if e[2] not in seen:
                    heapq.heappush(heap, e)
        rest = sorted(heap)
        rows.append((u, v, w, stale, "".join(sorted(seen)), rest))
    return rows


def brute_force(vertices, edges):
    """Every (V-1)-edge subset that connects all vertices."""
    n = len(vertices)
    trees = []
    for sub in combinations(edges, n - 1):
        parent = {v: v for v in vertices}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        ok = True
        for u, v, _ in sub:
            ru, rv = find(u), find(v)
            if ru == rv:
                ok = False
                break
            parent[ru] = rv
        if ok:
            trees.append(sum(w for *_, w in sub))
    return trees


def dijkstra_tree(edges, start):
    """Shortest-path tree from start: minimises each distance."""
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist, via, heap = {start: 0}, {}, [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist.get(v, float("inf")):
                dist[v], via[v] = d + w, (u, v, w)
                heapq.heappush(heap, (d + w, v))
    return dist, list(via.values())


if __name__ == "__main__":
    V = sorted({x for u, v, _ in EDGES for x in (u, v)})
    k = kruskal(V, EDGES)
    p = prim(EDGES, "A")
    print("kruskal:", [f"{u}{v} {w}" for u, v, w in k],
          "total", sum(w for *_, w in k))
    print("prim:   ", [f"{u}{v} {w}" for u, v, w in p],
          "total", sum(w for *_, w in p))

    print("\nKruskal trace (edges by weight):")
    for u, v, w, ru, rv, ok, pieces in kruskal_trace(V, EDGES):
        act = "take" if ok else "skip (cycle)"
        print(f"  {u}{v} {w:>2}  find {ru},{rv}  {act:<13}"
              f" pieces {' '.join(pieces)}")

    print("\nPrim trace from A (lazy heap):")
    for u, v, w, stale, tree, rest in prim_trace(EDGES, "A"):
        act = "stale, skip" if stale else "take"
        heap = " ".join(f"{a}{b}{x}" for x, a, b in rest)
        print(f"  pop {u}{v} {w:>2}  {act:<11} tree {tree:<6}"
              f" heap [{heap}]")

    trees = brute_force(V, EDGES)
    best = min(trees)
    print(f"\nbrute force: {len(trees)} spanning trees of"
          f" {len(list(combinations(EDGES, 5)))} 5-edge subsets;"
          f" min {best}, {trees.count(best)} tree(s) at min,"
          f" max {max(trees)}")

    cheap = sorted(EDGES, key=lambda e: e[2])[:5]
    print("5 cheapest edges:", [f"{u}{v} {w}" for u, v, w in cheap],
          "sum", sum(w for *_, w in cheap))

    dist, spt = dijkstra_tree(EDGES, "A")
    print("shortest-path tree from A:",
          [f"{u}{v} {w}" for u, v, w in spt],
          "total", sum(w for *_, w in spt))
    mst_d, _ = dijkstra_tree(k, "A")   # a tree has one path each
    print("dist from A, SPT:", dict(sorted(dist.items())))
    print("dist from A, MST:", dict(sorted(mst_d.items())))

    # random check: both algorithms match brute force
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(2, 7)
        vs = [chr(65 + i) for i in range(n)]
        es = [(vs[i], vs[rng.randrange(i)], rng.randint(1, 20))
              for i in range(1, n)]          # random tree: connected
        for a, b in combinations(vs, 2):
            if rng.random() < 0.4:
                es.append((a, b, rng.randint(1, 20)))
        bf = min(brute_force(vs, es))
        assert sum(w for *_, w in kruskal(vs, es)) == bf
        assert sum(w for *_, w in prim(es, vs[0])) == bf
    print("300 random graphs: kruskal == prim == brute force")
