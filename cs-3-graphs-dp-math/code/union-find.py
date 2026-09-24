"""Union-find (disjoint-set union) and Number of Islands II.

Card: cs-3-graphs-dp-math/union-find.html
Source: ljeng/cheat-sheet, coding-algorithms/graphs.md, Connectivity.
The text under "Number of Islands" there is Number of Islands II.
The source's find() has a stray `self` parameter and reads cell_root,
which is local to numIslands2 (a NameError); numIslands2 also lacks
a colon. Rewritten here with find and union as closures, plus union
by size. Runs under python3 (3.10).
"""
import random


def num_islands2(m, n, positions):
    parent, size = {}, {}

    def find(x):
        root = x
        while parent[root] != root:     # walk up to the root
            root = parent[root]
        while parent[x] != root:        # path compression:
            nxt = parent[x]             # point every cell on the
            parent[x] = root            # path straight at the root
            x = nxt
        return root

    def union(a, b):                    # True if two islands merged
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra                 # small tree under big root
        size[ra] += size[rb]
        return True

    islands, answer = 0, []
    for r, c in positions:
        if (r, c) not in parent:        # repeats change nothing
            parent[(r, c)], size[(r, c)] = (r, c), 1
            islands += 1
            for nb in [(r-1, c), (r, c-1), (r, c+1), (r+1, c)]:
                if nb in parent and union(nb, (r, c)):
                    islands -= 1
        answer.append(islands)
    return answer


class DSU:
    """Same structure, instrumented: counts parent-pointer hops."""

    def __init__(self, n, by_size=True, compress=True):
        self.parent = list(range(n))
        self.size = [1] * n
        self.by_size, self.compress = by_size, compress
        self.hops = 0

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
            self.hops += 1
        if self.compress:
            while self.parent[x] != root:
                self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.by_size and self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def depth(self, x):
        d = 0
        while self.parent[x] != x:
            x, d = self.parent[x], d + 1
        return d


def trace(positions):
    """Print each step: cell, land neighbours, roots, islands."""
    parent, size, islands = {}, {}, 0

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for r, c in positions:
        u = (r, c)
        parent[u], size[u] = u, 1
        islands += 1
        notes = []
        for nb in [(r-1, c), (r, c-1), (r, c+1), (r+1, c)]:
            if nb in parent:
                ra, rb = find(nb), find(u)
                if ra == rb:
                    notes.append(f"{nb}: same root {ra}")
                    continue
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]
                islands -= 1
                notes.append(f"{nb}: root {rb} -> {ra} (size "
                             f"{size[ra]})")
        print(f"  add {u}: {'; '.join(notes) or 'no land'}"
              f" | islands {islands}")
    return islands


def brute_islands(m, n, positions):
    """Recount with a flood fill after every add: O(k * m * n)."""
    land, out = set(), []
    for p in positions:
        land.add(tuple(p))
        seen, count = set(), 0
        for s in land:
            if s in seen:
                continue
            count += 1
            stack = [s]
            seen.add(s)
            while stack:
                r, c = stack.pop()
                for nb in [(r-1, c), (r, c-1), (r, c+1), (r+1, c)]:
                    if nb in land and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        out.append(count)
    return out


def chain_cost(n, by_size, compress):
    """union(i, i+1) for all i, then find(x) for every x."""
    d = DSU(n, by_size, compress)
    for i in range(n - 1):
        d.union(i + 1, i)               # naive: old root under new
    max_depth = max(d.depth(x) for x in range(n))
    d.hops = 0
    for x in range(n):
        d.find(x)
    return max_depth, d.hops


def pairwise_depth(k):
    """Merge equal-size trees in rounds: by-size's worst case."""
    n = 2 ** k
    d = DSU(n, by_size=True, compress=False)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            d.union(i, i + step)
        step *= 2
    return n, max(d.depth(x) for x in range(n))


if __name__ == "__main__":
    pos = [[0, 0], [0, 1], [1, 2], [2, 1]]
    print(num_islands2(3, 3, pos))                  # → [1, 1, 2, 3]
    print(num_islands2(3, 3, pos + [[1, 1]]))       # → [..., 1]
    print(num_islands2(3, 3, pos + [[0, 0]]))       # repeat → 3
    print("trace:")
    trace([tuple(p) for p in pos] + [(1, 1)])

    random.seed(1)
    for _ in range(300):
        m, n = random.randint(1, 6), random.randint(1, 6)
        ps = [[random.randrange(m), random.randrange(n)]
              for _ in range(random.randint(1, 20))]
        assert num_islands2(m, n, ps) == brute_islands(m, n, ps)
    print("300 random grids match the flood-fill recount")

    N = 10_000
    for by_size, compress in [(False, False), (False, True),
                              (True, False), (True, True)]:
        depth, hops = chain_cost(N, by_size, compress)
        print(f"n={N:,} by_size={by_size!s:5} compress="
              f"{compress!s:5}: max depth {depth:>5,},"
              f" hops for n finds {hops:>10,}")
    print("n(n-1)/2 =", f"{N * (N - 1) // 2:,}")
    n, depth = pairwise_depth(13)
    print(f"by size, equal-size merges, n={n:,}: max depth {depth}")
