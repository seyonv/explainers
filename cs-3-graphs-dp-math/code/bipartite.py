"""Bipartite graphs: BFS 2-colouring and the odd-cycle certificate.

Possible Bipartition and Is Graph Bipartite? from the source guide's
Graphs page (Cycle Detection), in plain Python. The source calls a
Graph.bipartite() method whose code/graph.py is missing from the repo.
Runs under python3 (3.10).
"""
from collections import deque
from itertools import product
import random
import sys


def possible_bipartition(n, dislikes):
    adj = [[] for _ in range(n + 1)]
    for a, b in dislikes:                # undirected: store both ways
        adj[a].append(b)
        adj[b].append(a)
    colour = [None] * (n + 1)
    for s in range(1, n + 1):            # every component
        if colour[s] is not None:
            continue
        colour[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if colour[v] is None:
                    colour[v] = 1 - colour[u]    # opposite side
                    queue.append(v)
                elif colour[v] == colour[u]:
                    return False                 # odd cycle
    return True


def two_colour(n, dislikes, trace=False):
    """Colouring, or the odd cycle that proves none exists."""
    adj = [[] for _ in range(n + 1)]
    for a, b in dislikes:
        adj[a].append(b)
        adj[b].append(a)
    colour, parent = [None] * (n + 1), [None] * (n + 1)
    for s in range(1, n + 1):
        if colour[s] is not None:
            continue
        colour[s] = 0
        queue = deque([s])
        if trace:
            print(f"  start {s}: colour A")
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if colour[v] is None:
                    colour[v] = 1 - colour[u]
                    parent[v] = u
                    queue.append(v)
                    if trace:
                        print(f"  pop {u} ({'AB'[colour[u]]}): "
                              f"{v} -> {'AB'[colour[v]]}, "
                              f"queue {list(queue)}")
                elif colour[v] == colour[u]:
                    if trace:
                        print(f"  pop {u} ({'AB'[colour[u]]}): "
                              f"{v} is also {'AB'[colour[v]]}"
                              " -> conflict")
                    return False, odd_cycle(parent, u, v)
                elif trace:
                    print(f"  pop {u} ({'AB'[colour[u]]}): "
                          f"{v} is {'AB'[colour[v]]}, ok")
    groups = ([x for x in range(1, n + 1) if colour[x] == 0],
              [x for x in range(1, n + 1) if colour[x] == 1])
    return True, groups


def odd_cycle(parent, u, v):
    """Tree paths u..lca and v..lca plus edge u-v: an odd cycle."""
    up = [u]
    while parent[up[-1]] is not None:
        up.append(parent[up[-1]])
    on_up = set(up)
    down = [v]
    while down[-1] not in on_up:
        down.append(parent[down[-1]])
    lca = down[-1]
    path = up[:up.index(lca) + 1]        # u .. lca
    return path + down[-2::-1] + [u]     # lca .. v, back to u


def is_bipartite(graph):
    """LeetCode 785: graph[u] lists u's neighbours, nodes 0..n-1."""
    edges = [(u + 1, v + 1) for u, vs in enumerate(graph)
             for v in vs if u < v]
    return possible_bipartition(len(graph), edges)


def greedy_edge_order(n, dislikes):
    """Wrong: colour each edge's endpoints as they come, no search."""
    colour = {}
    for a, b in dislikes:
        if a not in colour and b not in colour:
            colour[a], colour[b] = 0, 1
        elif a not in colour:
            colour[a] = 1 - colour[b]
        elif b not in colour:
            colour[b] = 1 - colour[a]
        elif colour[a] == colour[b]:
            return False
    return True


def dfs_recursive(n, dislikes):
    """Correct, but Python's recursion limit caps the path length."""
    adj = [[] for _ in range(n + 1)]
    for a, b in dislikes:
        adj[a].append(b)
        adj[b].append(a)
    colour = [None] * (n + 1)

    def go(u, c):
        colour[u] = c
        for v in adj[u]:
            if colour[v] is None:
                if not go(v, 1 - c):
                    return False
            elif colour[v] == c:
                return False
        return True

    return all(colour[s] is not None or go(s, 0)
               for s in range(1, n + 1))


def brute_force(n, dislikes):
    """Try all 2**n splits."""
    return any(all(side[a - 1] != side[b - 1] for a, b in dislikes)
               for side in product((0, 1), repeat=n))


if __name__ == "__main__":
    for n, d in [(4, [[1, 2], [1, 3], [2, 4]]),
                 (3, [[1, 2], [1, 3], [2, 3]])]:
        print(f"n={n} dislikes={d}")
        print("  possible_bipartition:", possible_bipartition(n, d))
        print("  certificate:", two_colour(n, d, trace=True))

    # the course's running graph, directions and weights ignored
    names = "ABCDE"
    es = ["AB", "AC", "CB", "BD", "CD", "CE", "DE"]
    ok, cyc = two_colour(5, [[names.index(a) + 1, names.index(b) + 1]
                             for a, b in es])
    print("running graph bipartite?", ok, "odd cycle:",
          "-".join(names[x - 1] for x in cyc))

    print("LeetCode 785 ex.1 [[1,2,3],[0,2],[0,1,3],[0,2]]:",
          is_bipartite([[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]))
    print("LeetCode 785 ex.2 [[1,3],[0,2],[1,3],[0,2]]:",
          is_bipartite([[1, 3], [0, 2], [1, 3], [0, 2]]))

    bad = [[1, 2], [3, 4], [1, 3]]
    print(f"greedy edge order on {bad}:", greedy_edge_order(4, bad),
          "| BFS:", possible_bipartition(4, bad))

    path = [[i, i + 1] for i in range(1, 2000)]
    print("recursion limit:", sys.getrecursionlimit())
    try:
        dfs_recursive(2000, path)
    except RecursionError:
        print("recursive DFS on a 2,000-node path: RecursionError")
    print("BFS on the same path:", possible_bipartition(2000, path))

    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(1, 8)
        pairs = [[a, b] for a in range(1, n + 1)
                 for b in range(a + 1, n + 1)]
        d = rng.sample(pairs, rng.randint(0, len(pairs)))
        assert possible_bipartition(n, d) == brute_force(n, d)
        ok, cert = two_colour(n, d)
        if not ok:
            assert (len(cert) - 1) % 2 == 1          # odd length
            assert all([a, b] in d or [b, a] in d
                       for a, b in zip(cert, cert[1:]))
    print("2,000 random graphs agree with brute force;",
          "every odd cycle checked")
    for n in (4, 30):
        print(f"brute force splits for n={n}: 2**{n} = {2 ** n:,}")
