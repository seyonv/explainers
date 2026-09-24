"""Dijkstra's algorithm with heapq (no Graph class, no scipy).

The cheat-sheet's algorithms.md > Dijkstra's is empty; its graphs.md >
Distance calls scipy's csgraph.dijkstra (Network Delay Time) or a
Graph.dijkstra from a code/graph.py that is missing from the repo
(Shortest Path with Alternating Colors). Plain Python versions here.
Runs under python3 (3.10).
"""
import heapq
import random
from collections import defaultdict

INF = float("inf")


def dijkstra(adj, source):
    """The card's version. adj: {u: [(v, w), ...]}, every w >= 0."""
    dist = {source: 0}
    done = set()
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:                  # stale entry: skip it
            continue
        done.add(u)                    # d is now final for u
        for v, w in adj.get(u, ()):
            if d + w < dist.get(v, INF):
                dist[v] = d + w        # relax: a shorter way to v
                heapq.heappush(heap, (d + w, v))
    return dist


def network_delay_time(times, n, k):
    """LeetCode 743: time for a signal from k to reach all n nodes."""
    adj = defaultdict(list)
    for u, v, w in times:
        adj[u].append((v, w))
    dist = dijkstra(adj, k)
    return max(dist.values()) if len(dist) == n else -1


def dijkstra_traced(adj, source, trace=False):
    """Same algorithm, plus parents and a printed trace per pop."""
    dist, parent = {source: 0}, {source: None}
    done = set()
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:                    # stale entry: skip it
            if trace:
                print(f"  pop ({d}, {u})  stale, skip")
            continue
        done.add(u)                      # d is now final for u
        relaxed = []
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, INF):    # found a shorter way to v
                dist[v], parent[v] = nd, u
                heapq.heappush(heap, (nd, v))
                relaxed.append(f"{v}={nd}")
        if trace:
            print(f"  pop ({d}, {u})  final {u}={d}  relax:",
                  ", ".join(relaxed) or "-",
                  "| heap:", sorted(heap))
    return dist, parent


def path_to(parent, v):
    out = []
    while v is not None:
        out.append(v)
        v = parent[v]
    return out[::-1]


def shortest_alternating_paths(n, red_edges, blue_edges):
    """LeetCode 1129: state = (node, colour of the next edge)."""
    adj = defaultdict(list)
    for u, v in red_edges:
        adj[(u, "red")].append(((v, "blue"), 1))
    for u, v in blue_edges:
        adj[(u, "blue")].append(((v, "red"), 1))
    # one search from both start states: a super-source S
    adj["S"] = [((0, "red"), 0), ((0, "blue"), 0)]
    dist = dijkstra(adj, "S")
    out = []
    for x in range(n):
        best = min(dist.get((x, c), INF) for c in ("red", "blue"))
        out.append(best if best < INF else -1)
    return out


def bellman_ford(nodes, edges, source):
    """Reference answer: relax every edge V - 1 times."""
    dist = {u: INF for u in nodes}
    dist[source] = 0
    for _ in range(len(nodes) - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist


def count_ops(adj, source):
    """Heap pushes and pops, to compare with V and E."""
    dist, done = {source: 0}, set()
    heap, pushes, pops = [(0, source)], 1, 0
    while heap:
        d, u = heapq.heappop(heap)
        pops += 1
        if u in done:
            continue
        done.add(u)
        for v, w in adj.get(u, ()):
            if d + w < dist.get(v, INF):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
                pushes += 1
    return pushes, pops


if __name__ == "__main__":
    edges = [("A", "B", 4), ("A", "C", 2), ("C", "B", 1),
             ("B", "D", 5), ("C", "D", 8), ("C", "E", 10),
             ("D", "E", 2)]
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
    print("running graph, source A:")
    dist, parent = dijkstra_traced(adj, "A", trace=True)
    print("  dist:", dict(sorted(dist.items())))
    print("  path to E:", "-".join(path_to(parent, "E")))
    print("  heap pushes, pops:", count_ops(adj, "A"),
          "| V = 5, E =", len(edges))

    times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    print("network delay", times, "n=4 k=2:",
          network_delay_time(times, 4, 2))
    print("network delay [[1,2,1]] n=2 k=2:",
          network_delay_time([[1, 2, 1]], 2, 2))

    print("alternating n=3 red [[0,1],[1,2]] blue []:",
          shortest_alternating_paths(3, [[0, 1], [1, 2]], []))
    print("alternating n=3 red [[0,1]] blue [[1,2]]:",
          shortest_alternating_paths(3, [[0, 1]], [[1, 2]]))

    neg_edges = [("A", "B", 2), ("A", "C", 3), ("C", "B", -2),
                 ("B", "D", 1)]
    neg = defaultdict(list)
    for u, v, w in neg_edges:
        neg[u].append((v, w))
    print("negative edge C->B -2:")
    d_neg, _ = dijkstra_traced(neg, "A", trace=True)
    bf = bellman_ford("ABCD", neg_edges, "A")
    print("  Dijkstra D =", d_neg["D"], "| Bellman-Ford D =", bf["D"])

    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(1, 8)
        es = [(rng.randrange(n), rng.randrange(n), rng.randint(0, 9))
              for _ in range(rng.randint(0, 20))]
        g = defaultdict(list)
        for u, v, w in es:
            g[u].append((v, w))
        got = dijkstra(g, 0)
        ref = bellman_ford(range(n), es, 0)
        assert got == {u: d for u, d in ref.items() if d < INF}
    print("2,000 random graphs agree with Bellman-Ford")
