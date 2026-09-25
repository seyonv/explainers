"""Eulerian paths: Reconstruct Itinerary with Hierholzer's algorithm.

From the source guide's Data Structures page, Stacks section.
Runs under python3 (3.10).
"""
from collections import Counter, defaultdict
import random


def find_itinerary(tickets, trace=False):
    adj = defaultdict(list)
    for a, b in sorted(tickets, reverse=True):
        adj[a].append(b)             # smallest destination at the end
    stack, route = ["JFK"], []
    while stack:
        top = stack[-1]
        if adj[top]:                 # unused ticket out of top: take it
            stack.append(adj[top].pop())
            step = "push " + stack[-1]
        else:                        # stuck: top is final from here on
            route.append(stack.pop())
            step = "done " + route[-1]
        if trace:
            print(f"  {step:9} stack {stack}  route {route}")
    return route[::-1]               # finished last-to-first


def find_itinerary_source(tickets):
    """The source guide's version, unchanged apart from names."""
    airports = ["JFK"]
    adjacency = defaultdict(list)
    for from_, to in sorted(tickets, reverse=True):
        adjacency[from_].append(to)
    itinerary = []
    while airports:
        while adjacency[airports[-1]]:
            airports.append(adjacency[airports[-1]].pop())
        itinerary.append(airports.pop())
    return itinerary[::-1]


def euler_ends(edges):
    """Degree check for a directed graph: (start, end) or None.

    Assumes the edges form one connected piece.
    """
    out = Counter(a for a, _ in edges)
    inn = Counter(b for _, b in edges)
    nodes = set(out) | set(inn)
    diff = {v: out[v] - inn[v] for v in nodes}
    starts = [v for v in nodes if diff[v] == 1]
    ends = [v for v in nodes if diff[v] == -1]
    if any(abs(d) > 1 for d in diff.values()):
        return None
    if len(starts) == len(ends) == 1:
        return starts[0], ends[0]
    if not starts and not ends:
        return "any", "same"         # an Eulerian circuit
    return None


def greedy_walk(tickets):
    """Always take the smallest unused ticket, never back up."""
    adj = defaultdict(list)
    for a, b in sorted(tickets, reverse=True):
        adj[a].append(b)
    route = ["JFK"]
    while adj[route[-1]]:
        route.append(adj[route[-1]].pop())
    return route


def brute_force(tickets):
    """Backtracking in sorted order: first full route is smallest."""
    tickets = sorted(tickets)
    used = [False] * len(tickets)
    route = ["JFK"]

    def go():
        if len(route) == len(tickets) + 1:
            return True
        for i, (a, b) in enumerate(tickets):
            if not used[i] and a == route[-1]:
                used[i] = True
                route.append(b)
                if go():
                    return True
                used[i] = False
                route.pop()
        return False

    go()
    return route


def random_tickets(rng):
    names = ["ATL", "JFK", "LAX", "SFO"]
    walk = ["JFK"]
    for _ in range(rng.randint(1, 8)):
        walk.append(rng.choice(names))
    tickets = [[a, b] for a, b in zip(walk, walk[1:])]
    rng.shuffle(tickets)
    return tickets


if __name__ == "__main__":
    t = [["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"],
         ["ATL", "JFK"], ["ATL", "SFO"]]
    print("tickets", t)
    print("degree check (start, end):", euler_ends(t))
    print("answer:", find_itinerary(t, trace=True))
    print("add SFO->JFK:", euler_ends(t + [["SFO", "JFK"]]))

    t2 = [["JFK", "KUL"], ["JFK", "NRT"], ["NRT", "JFK"]]
    print("\ntickets", t2)
    print("degree check (start, end):", euler_ends(t2))
    print("greedy walk, no backing up:", greedy_walk(t2))
    print("answer:", find_itinerary(t2, trace=True))

    # Koenigsberg, undirected: land A has 5 bridges, B, C, D have 3
    deg = {"A": 5, "B": 3, "C": 3, "D": 3}
    odd = [v for v, d in deg.items() if d % 2]
    print("\nKoenigsberg odd-degree land masses:", len(odd),
          "(an Euler path needs 0 or 2)")

    rng = random.Random(0)
    for _ in range(2000):
        ts = random_tickets(rng)
        want = brute_force(ts)
        assert find_itinerary(ts) == want
        assert find_itinerary_source(ts) == want
    print("2,000 random ticket sets agree with brute force"
          " (and with the source's version)")
