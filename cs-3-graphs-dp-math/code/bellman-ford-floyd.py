"""Bellman-Ford and Floyd-Warshall.

Cheapest Flights Within K Stops (Bellman-Ford limited to K + 1 rounds)
and Find the City (Floyd-Warshall), from the source guide's Graphs page,
in plain Python (no Graph class, no scipy). Runs under python3 (3.10).
"""
import heapq
import itertools
import random

INF = float("inf")


def cheapest_price(n, flights, src, dst, k):
    cost = [INF] * n
    cost[src] = 0
    for _ in range(k + 1):           # at most k + 1 flights
        prev = cost[:]               # read last round only
        for u, v, w in flights:
            if prev[u] + w < cost[v]:
                cost[v] = prev[u] + w
    return -1 if cost[dst] == INF else cost[dst]


def find_the_city(n, edges, threshold):
    d = [[0 if i == j else INF for j in range(n)] for i in range(n)]
    for u, v, w in edges:
        d[u][v] = d[v][u] = min(d[u][v], w)
    for k in range(n):               # allow k as a middle stop
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    best, city = INF, -1
    for i in range(n):
        near = sum(d[i][j] <= threshold for j in range(n) if j != i)
        if near <= best:             # ties go to the larger i
            best, city = near, i
    return city


def cheapest_price_trace(n, flights, src, dst, k):
    """Same as cheapest_price, printing the costs after each round."""
    cost = [INF] * n
    cost[src] = 0
    for rnd in range(1, k + 2):      # round r: paths of <= r flights
        prev = cost[:]               # read last round, write this one
        for u, v, w in flights:
            if prev[u] + w < cost[v]:
                cost[v] = prev[u] + w    # relax edge u -> v
        print(f"  round {rnd}: {cost}")
    return -1 if cost[dst] == INF else cost[dst]


def cheapest_in_place(n, flights, src, dst, k):
    """Bug demo: no snapshot, so one round can chain several flights."""
    cost = [INF] * n
    cost[src] = 0
    for _ in range(k + 1):
        for u, v, w in flights:
            cost[v] = min(cost[v], cost[u] + w)
    return -1 if cost[dst] == INF else cost[dst]


def dijkstra_ignoring_stops(n, flights, src, dst):
    """Plain Dijkstra: cheapest path, however many stops it uses."""
    adj = [[] for _ in range(n)]
    for u, v, w in flights:
        adj[u].append((v, w))
    dist, parent = [INF] * n, [None] * n
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v], parent[v] = d + w, u
                heapq.heappush(heap, (d + w, v))
    path, x = [], dst
    while x is not None:
        path.append(x)
        x = parent[x]
    return dist[dst], path[::-1]


def prim_total(n, flights):
    """What the source computes: total weight of an MST (undirected)."""
    adj = [[] for _ in range(n)]
    for u, v, w in flights:
        adj[u].append((w, v))
        adj[v].append((w, u))
    seen, total, heap, tree = {0}, 0, list(adj[0]), []
    heapq.heapify(heap)
    while heap and len(seen) < n:
        w, v = heapq.heappop(heap)
        if v in seen:
            continue
        seen.add(v)
        total += w
        tree.append((v, w))
        for e in adj[v]:
            heapq.heappush(heap, e)
    return total, tree


def brute_cheapest(n, flights, src, dst, k):
    """Try every simple path of at most k + 1 flights (for testing)."""
    best = INF
    adj = [[] for _ in range(n)]
    for u, v, w in flights:
        adj[u].append((v, w))
    stack = [(src, 0, 0, {src})]
    while stack:
        u, c, f, seen = stack.pop()
        if u == dst:
            best = min(best, c)
            continue
        if f == k + 1:
            continue
        for v, w in adj[u]:
            if v not in seen:
                stack.append((v, c + w, f + 1, seen | {v}))
    return -1 if best == INF else best


def floyd_warshall(n, edges, trace=False):
    """find_the_city's table, returned (and printed if trace)."""
    d = [[0 if i == j else INF for j in range(n)] for i in range(n)]
    for u, v, w in edges:
        d[u][v] = d[v][u] = min(d[u][v], w)
    for k in range(n):                   # allow k as a middle stop
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    if trace:
                        print(f"  k={k}: d[{i}][{j}] {d[i][j]} -> "
                              f"{d[i][k]} + {d[k][j]} = "
                              f"{d[i][k] + d[k][j]}")
                    d[i][j] = d[i][k] + d[k][j]
    return d


def bellman_ford(nodes, edges, src):
    """Distances from src, or None if a negative cycle is reachable."""
    dist = {x: INF for x in nodes}
    dist[src] = 0
    for _ in range(len(nodes) - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for u, v, w in edges:            # round |V|: still improving?
        if dist[u] + w < dist[v]:
            return None
    return dist


def rounds_to(nodes, edges, src, dst):
    """Best cost to dst using at most r edges, for r = 1 .. |V| - 1."""
    dist = {x: INF for x in nodes}
    dist[src] = 0
    out = []
    for _ in range(len(nodes) - 1):
        prev = dict(dist)
        for u, v, w in edges:
            dist[v] = min(dist[v], prev[u] + w)
        out.append(dist[dst])
    return out


if __name__ == "__main__":
    flights = [[0, 1, 100], [1, 2, 100], [2, 0, 100],
               [1, 3, 600], [2, 3, 200]]
    print("cheapest flights, n=4, src 0, dst 3, k=1")
    print("  answer:", cheapest_price_trace(4, flights, 0, 3, 1))
    print("  k=2 (one more round):",
          cheapest_price_trace(4, flights, 0, 3, 2))
    print("  in-place relaxation, k=1:",
          cheapest_in_place(4, flights, 0, 3, 1),
          "| k=0:", cheapest_in_place(4, flights, 0, 3, 0),
          "(correct k=0:", cheapest_price(4, flights, 0, 3, 0), ")")
    print("  plain Dijkstra (cost, path):",
          dijkstra_ignoring_stops(4, flights, 0, 3))
    print("  source's Prim (MST total, tree):", prim_total(4, flights))

    print("card demo:")
    print(cheapest_price(4, flights, 0, 3, 1))     # → 700
    print(cheapest_price(4, flights, 0, 3, 2))     # → 400
    print(find_the_city(4, [[0,1,3], [1,2,1], [1,3,4], [2,3,1]], 4))
    # → 3

    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(2, 6)
        fl = [[u, v, rng.randint(1, 50)]
              for u, v in itertools.permutations(range(n), 2)
              if rng.random() < 0.4]
        s, t = rng.sample(range(n), 2)
        k = rng.randint(0, n - 1)
        assert (cheapest_price(n, fl, s, t, k)
                == brute_cheapest(n, fl, s, t, k))
    print("2,000 random cases agree with trying every path")

    edges = [[0, 1, 3], [1, 2, 1], [1, 3, 4], [2, 3, 1]]
    print("find the city, n=4, threshold 4")
    d = floyd_warshall(4, edges, trace=True)
    for i, row in enumerate(d):
        near = [j for j in range(4) if j != i and row[j] <= 4]
        print(f"  city {i}: {row}  within 4: {near}")
    print("  answer:", find_the_city(4, edges, 4))

    nodes = "ABCDE"
    g = [("A", "B", 4), ("A", "C", 2), ("C", "B", 1), ("B", "D", 5),
         ("C", "D", 8), ("C", "E", 10), ("D", "E", 2)]
    print("running graph, best A -> E with <= r edges:",
          rounds_to(nodes, g, "A", "E"))
    print("  bellman_ford:", bellman_ford(nodes, g, "A"))
    neg = g + [("D", "C", -9)]           # C -> D -> C costs 8 - 9 = -1
    print("  with D -> C = -9:", bellman_ford(nodes, neg, "A"))
