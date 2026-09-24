"""Breadth-first search in layers.

Bus Routes, Jump Game III and Snakes and Ladders from the cheat-sheet's
Graphs page, all solved with plain BFS (no Graph class, no scipy).
Runs under python3 (3.10).
"""
from collections import defaultdict, deque
import random


def bfs_layers(adj, start):
    """Hop distance from start to each reachable node, by layer."""
    dist = {start: 0}
    frontier = [start]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in dist:          # mark on discovery
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def dfs_depth(adj, start, goal):
    """Depth at which a plain DFS first reaches goal (not shortest)."""
    stack, seen = [(start, 0)], set()
    while stack:
        u, d = stack.pop()
        if u == goal:
            return d
        if u in seen:
            continue
        seen.add(u)
        for v in reversed(adj.get(u, ())):   # visit in listed order
            stack.append((v, d + 1))
    return -1


def num_buses(routes, source, target, trace=False):
    if source == target:
        return 0
    by_stop = defaultdict(list)            # stop -> buses through it
    for bus, route in enumerate(routes):
        for stop in route:
            by_stop[stop].append(bus)
    seen_stops, used_buses = {source}, set()
    frontier, buses = [source], 0
    while frontier:
        buses += 1                         # one more bus per layer
        nxt = []
        for stop in frontier:
            for bus in by_stop[stop]:
                if bus in used_buses:      # each bus boarded once
                    continue
                used_buses.add(bus)
                if trace:
                    print(f"  layer {buses}: at stop {stop} "
                          f"board bus {bus} -> {routes[bus]}")
                for s in routes[bus]:
                    if s == target:
                        return buses
                    if s not in seen_stops:
                        seen_stops.add(s)
                        nxt.append(s)
        frontier = nxt
    return -1


def num_buses_source(routes, source, target):
    """The cheat-sheet's version, queue of (stop, buses), to check."""
    graph = defaultdict(set)
    for i, route in enumerate(routes):
        for stop in route:
            graph[stop].add(i)
    queue = deque([(source, 0)])
    traveled_routes, traveled_stops = set(), {source}
    while queue:
        stop, buses = queue.popleft()
        if stop == target:
            return buses
        for i in graph[stop]:
            if i in traveled_routes:
                continue
            traveled_routes.add(i)
            for j in routes[i]:
                if j not in traveled_stops:
                    traveled_stops.add(j)
                    queue.append((j, buses + 1))
    return -1


def can_reach(arr, start):
    """Jump Game III: from i jump to i + arr[i] or i - arr[i]."""
    adj = {i: [j for j in (i + x, i - x) if 0 <= j < len(arr)]
           for i, x in enumerate(arr)}
    dist = bfs_layers(adj, start)
    zeros = [i for i in dist if arr[i] == 0]
    return bool(zeros), {i: dist[i] for i in zeros}


def snakes_and_ladders(board):
    """Fewest dice rolls from square 1 to n*n (board not modified)."""
    rows = [r[:] for r in reversed(board)]
    cells = []
    for k, row in enumerate(rows):
        cells += row if k % 2 == 0 else row[::-1]
    m = len(cells)
    adj = {}
    for u in range(m):
        adj[u] = []
        for step in range(1, 7):
            v = u + step
            if v < m:
                adj[u].append(v if cells[v] == -1 else cells[v] - 1)
    return bfs_layers(adj, 0).get(m - 1, -1)


if __name__ == "__main__":
    routes = [[1, 2, 7], [3, 6, 7]]
    print("bus routes", routes, "1 -> 6")
    print("  answer:", num_buses(routes, 1, 6, trace=True))

    # the course's running graph, weights ignored
    adj = {"A": ["B", "C"], "B": ["D"], "C": ["B", "D", "E"],
           "D": ["E"]}
    d = bfs_layers(adj, "A")
    print("running graph hop distances:", d)
    print("  BFS depth of E:", d["E"], "| DFS first reaches E at depth",
          dfs_depth(adj, "A", "E"))
    w = {("A", "C"): 2, ("C", "E"): 10, ("A", "B"): 4, ("B", "D"): 5,
         ("D", "E"): 2, ("C", "B"): 1}
    print("  cost of A-C-E:", w["A", "C"] + w["C", "E"],
          "| cost of A-C-B-D-E:",
          w["A", "C"] + w["C", "B"] + w["B", "D"] + w["D", "E"])

    print("jump game III [4,2,3,0,3,1,2], start 5:",
          can_reach([4, 2, 3, 0, 3, 1, 2], 5))
    board = [[-1] * 6 for _ in range(6)]
    board[3][1], board[3][4], board[5][1] = 35, 13, 15
    print("snakes and ladders (LeetCode example 1):",
          snakes_and_ladders(board))

    rng = random.Random(0)
    for _ in range(2000):
        rs = [rng.sample(range(12), rng.randint(1, 4))
              for _ in range(rng.randint(1, 5))]
        s, t = rng.randrange(12), rng.randrange(12)
        assert num_buses(rs, s, t) == num_buses_source(rs, s, t)
    print("2,000 random cases agree with the source's version")

    L = 100_000
    print(f"stop-to-stop edges for one route of {L:,} stops:",
          f"{L * (L - 1):,} (vs {L:,} stop-bus links)")
