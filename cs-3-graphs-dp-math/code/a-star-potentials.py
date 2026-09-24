"""A* search, and why the cheat-sheet's reweighted Dijkstra is A*.

The cheat-sheet's algorithms.md > A* is empty. Its graphs.md >
Distance solves Shortest Path in Binary Matrix with scipy's dijkstra
on edges reweighted by a Chebyshev potential. No scipy here: plain
heapq versions of A*, of that reweighted Dijkstra, and of BFS, with
counts of nodes expanded. Runs under python3 (3.10).
"""
import heapq
import random
from collections import deque

MOVES = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
         if di or dj]                    # 8 directions


def neighbours(grid, i, j):
    n = len(grid)
    for di, dj in MOVES:
        a, b = i + di, j + dj
        if 0 <= a < n and 0 <= b < n and not grid[a][b]:
            yield a, b


def astar(grid):
    n = len(grid)
    if grid[0][0] or grid[-1][-1]:
        return -1
    h = lambda i, j: max(n - 1 - i, n - 1 - j)   # Chebyshev
    g = {(0, 0): 0}
    heap = [(h(0, 0), 0, (0, 0))]       # (f, -g, cell)
    done = set()
    while heap:
        _, neg_g, (i, j) = heapq.heappop(heap)
        if (i, j) in done:
            continue                    # stale entry
        done.add((i, j))
        if (i, j) == (n - 1, n - 1):
            return -neg_g + 1           # moves + 1 = cells
        for a in (i - 1, i, i + 1):
            for b in (j - 1, j, j + 1):
                if 0 <= a < n and 0 <= b < n and not grid[a][b]:
                    gv = -neg_g + 1
                    if gv < g.get((a, b), n * n):
                        g[(a, b)] = gv
                        f = gv + h(a, b)
                        heapq.heappush(heap, (f, -gv, (a, b)))
    return -1


def cheb(n, i, j):
    """Chebyshev distance to (n-1, n-1): king moves still needed."""
    return max(n - 1 - i, n - 1 - j)


def astar_count(grid, trace=False):
    """astar with a trace; returns (cells, nodes expanded)."""
    n = len(grid)
    if grid[0][0] or grid[-1][-1]:
        return -1, 0
    g = {(0, 0): 0}
    heap = [(cheb(n, 0, 0), 0, (0, 0))]  # (f = g + h, -g, cell)
    done = set()
    while heap:
        f, neg_g, u = heapq.heappop(heap)  # ties: deepest first
        gu = -neg_g
        if u in done:
            continue                     # stale entry
        done.add(u)
        if trace:
            print(f"  pop {u} g={gu} h={f - gu} f={f}")
        if u == (n - 1, n - 1):
            return gu + 1, len(done)     # moves + 1 = cells
        for v in neighbours(grid, *u):
            if gu + 1 < g.get(v, n * n):
                g[v] = gu + 1
                heapq.heappush(heap, (gu + 1 + cheb(n, *v),
                                      -(gu + 1), v))
    return -1, len(done)


def reweighted_dijkstra(grid):
    """The source's idea: w'(u,v) = 1 + h(v) - h(u), then Dijkstra.

    Stops when the target is popped (scipy's call does not).
    """
    n = len(grid)
    if grid[0][0] or grid[-1][-1]:
        return -1, 0
    h = lambda c: cheb(n, *c)
    d = {(0, 0): 0}
    heap, done = [(0, (0, 0))], set()
    while heap:
        du, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == (n - 1, n - 1):
            # d'(t) = d(t) - h(s) + h(t), and h(t) = 0
            return du + h((0, 0)) + 1, len(done)
        for v in neighbours(grid, *u):
            w = 1 + h(v) - h(u)          # never negative
            if du + w < d.get(v, n * n):
                d[v] = du + w
                heapq.heappush(heap, (du + w, v))
    return -1, len(done)


def best_first(grid, h, use_g=True):
    """A* with any heuristic h(n, i, j); use_g=False is greedy."""
    n = len(grid)
    if grid[0][0] or grid[-1][-1]:
        return -1, 0
    g = {(0, 0): 0}
    heap, done = [(h(n, 0, 0), 0, (0, 0))], set()
    while heap:
        _, neg_g, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == (n - 1, n - 1):
            return -neg_g + 1, len(done)
        for v in neighbours(grid, *u):
            gv = -neg_g + 1
            if gv < g.get(v, n * n):
                g[v] = gv
                key = (gv if use_g else 0) + h(n, *v)
                heapq.heappush(heap, (key, -gv, v))
    return -1, len(done)


def manhattan(n, i, j):
    return (n - 1 - i) + (n - 1 - j)   # overestimates diagonals


def bfs(grid):
    """Plain BFS; counts nodes popped until the target is popped."""
    n = len(grid)
    if grid[0][0] or grid[-1][-1]:
        return -1, 0
    dist = {(0, 0): 1}
    q, popped = deque([(0, 0)]), 0
    while q:
        u = q.popleft()
        popped += 1
        if u == (n - 1, n - 1):
            return dist[u], popped
        for v in neighbours(grid, *u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return -1, popped


if __name__ == "__main__":
    small = [[0, 0, 0], [1, 1, 0], [1, 1, 0]]
    print("astar(small) =", astar(small))
    print("A* on", small)
    print("  ->", astar_count(small, trace=True))
    print("  reweighted Dijkstra ->", reweighted_dijkstra(small))
    print("  BFS ->", bfs(small))

    n = 3
    print("potentials h (Chebyshev to target):")
    for i in range(n):
        print("  ", [cheb(n, i, j) for j in range(n)])
    print("reweighted edges on the clear path:")
    path = [(0, 0), (0, 1), (1, 2), (2, 2)]
    for u, v in zip(path, path[1:]):
        print(f"  {u}->{v}: 1 + {cheb(n, *v)} - {cheb(n, *u)}"
              f" = {1 + cheb(n, *v) - cheb(n, *u)}")

    open20 = [[0] * 20 for _ in range(20)]
    print("20x20 open grid (length, expanded):")
    print("  BFS                 ", bfs(open20))
    print("  A*                  ", astar_count(open20))
    print("  reweighted Dijkstra ", reweighted_dijkstra(open20))

    wall = [[0] * 20 for _ in range(20)]
    for i in range(1, 20):
        wall[i][10] = 1                  # wall with a gap at row 0
    print("20x20, wall in column 10 open only at row 0:")
    print("  BFS                 ", bfs(wall))
    print("  A*                  ", astar_count(wall))
    print("  reweighted Dijkstra ", reweighted_dijkstra(wall))

    zero = lambda n, i, j: 0
    print("  A* with h = 0       ", best_first(wall, zero))
    print("  greedy, f = h only  ", best_first(wall, cheb, False))

    five = [[0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    print("5x5: BFS", bfs(five)[0],
          "| A* Chebyshev", astar(five),
          "| A* Manhattan", best_first(five, manhattan)[0])

    rng = random.Random(0)
    for _ in range(2000):
        m = rng.randint(1, 8)
        gr = [[int(rng.random() < 0.3) for _ in range(m)]
              for _ in range(m)]
        a, b = astar_count(gr)[0], reweighted_dijkstra(gr)[0]
        c, d = bfs(gr)[0], astar(gr)
        assert a == b == c == d, (gr, a, b, c, d)
    print("2,000 random grids: A* = reweighted Dijkstra = BFS")
