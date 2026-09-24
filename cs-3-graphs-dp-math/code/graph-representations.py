"""Representing a graph: edge list, adjacency matrix, adjacency list,
and a grid read as an implicit graph (the island problems).

Running example (cs-3 course): A->B 4, A->C 2, C->B 1, B->D 5,
C->D 8, C->E 10, D->E 2.  Python 3.10, standard library only.
"""
from collections import defaultdict

EDGES = [("A", "B", 4), ("A", "C", 2), ("C", "B", 1), ("B", "D", 5),
         ("C", "D", 8), ("C", "E", 10), ("D", "E", 2)]


def to_matrix(edges):
    names = sorted({x for u, v, _ in edges for x in (u, v)})
    idx = {x: i for i, x in enumerate(names)}
    M = [[None] * len(names) for _ in names]   # None = no edge
    for u, v, w in edges:
        M[idx[u]][idx[v]] = w
    return names, idx, M


def to_adj(edges, directed=True):
    E = defaultdict(dict)          # E[u][v] = weight, like the source
    for u, v, w in edges:
        E[u][v] = w
        if not directed:
            E[v][u] = w            # undirected: store both ways
    return E


def get_neighbors(grid, i, j, color=None, k=4):
    steps = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if k == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = []
    for di, dj in steps:
        r, c = i + di, j + dj
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            if color is None or grid[r][c] == color:
                out.append((r, c))
    return out


def count_islands(grid, k=4):
    seen, islands = set(), 0
    for i, row in enumerate(grid):
        for j, x in enumerate(row):
            if x != 1 or (i, j) in seen:
                continue
            islands += 1
            stack = [(i, j)]
            seen.add((i, j))
            while stack:                  # iterative DFS
                cell = stack.pop()
                for nb in get_neighbors(grid, *cell, color=1, k=k):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
    return islands


GRID = [[1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]]


def _check():
    import random
    rng = random.Random(0)
    for _ in range(1000):
        n = rng.randint(1, 8)
        names = [str(i) for i in range(n)]
        es = {}
        for _ in range(rng.randint(1, n * n)):
            u, v = rng.choice(names), rng.choice(names)
            es[(u, v)] = rng.randint(1, 9)
        edges = [(u, v, w) for (u, v), w in es.items()]
        _, idx, M = to_matrix(edges)
        E = to_adj(edges)
        for u in idx:
            for v in idx:
                a = M[idx[u]][idx[v]]
                b = E[u].get(v)
                c = next((w for x, y, w in edges
                          if (x, y) == (u, v)), None)
                assert a == b == c
    return "1,000 random graphs: matrix, list, edge list agree"


if __name__ == "__main__":
    V = len({x for u, v, _ in EDGES for x in (u, v)})
    E_n = len(EDGES)
    names, idx, M = to_matrix(EDGES)
    E = to_adj(EDGES)
    print(f"V = {V}, E = {E_n}")
    print("matrix cells V^2 =", V * V, "| filled", E_n,
          "| empty", V * V - E_n,
          f"({(V * V - E_n) / (V * V):.0%})")
    print("adjacency list entries V + E =", V + E_n)
    print("max directed edges V(V-1) =", V * (V - 1),
          f"-> density {E_n / (V * (V - 1)):.0%}")
    print()
    print("   " + "  ".join(f"{x:>2}" for x in names))
    for x, row in zip(names, M):
        print(f"{x}  " + "  ".join(f"{'.' if w is None else w:>2}"
                                   for w in row))
    print()
    print("adjacency list:", {u: dict(E[u]) for u in names if u in E})
    print()
    print("has edge C->B?  matrix", M[idx["C"]][idx["B"]] is not None,
          "| list", "B" in E["C"],
          "| edge list", any(u == "C" and v == "B"
                             for u, v, _ in EDGES))
    print("has edge B->C?  matrix", M[idx["B"]][idx["C"]] is not None,
          "| list", "C" in E["B"])
    print("neighbours of C:", list(E["C"]),
          "(list touches 3; matrix scans row of", V, "cells)")
    print()
    print("get_neighbors(GRID, 1, 1, color=1, k=4):",
          get_neighbors(GRID, 1, 1, color=1, k=4))
    print("get_neighbors(GRID, 1, 1, color=1, k=8):",
          get_neighbors(GRID, 1, 1, color=1, k=8))
    print("islands k=4:", count_islands(GRID, 4),
          "| k=8:", count_islands(GRID, 8))
    print()
    for v, e in [(10**4, 5 * 10**4), (10**6, 4 * 10**6)]:
        print(f"V={v:,} E={e:,}: V^2 = {v * v:,} cells, "
              f"V+E = {v + e:,} entries, ratio {v * v / (v + e):,.0f}x")
    print(_check())
    print()
    print("card snippet:")
    print("B" in E["C"], "C" in E["B"], list(E["C"]))
    print(get_neighbors(GRID, 1, 1, color=1, k=8))
    print(count_islands(GRID, 4), count_islands(GRID, 8))
