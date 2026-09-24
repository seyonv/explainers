"""Grid DP, forwards and backwards: Dungeon Game and Cherry Pickup.

Run: python3 dp-grid.py   (Python 3.10, standard library only)
"""
import itertools
import math
import random


def min_hp(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    need = [float("inf")] * (n + 1)  # row below, then this row
    need[n - 1] = 1                  # seed: survive with 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            best_next = min(need[j], need[j + 1])  # down, right
            need[j] = max(best_next - dungeon[i][j], 1)
    return need[0]

def cherry_pickup(grid):
    n = len(grid)
    dp = [[-1] * n for _ in range(n)]  # dp[r1][r2], -1 = unreachable
    dp[0][0] = grid[0][0]
    for k in range(1, 2 * n - 1):
        new = [[-1] * n for _ in range(n)]
        for r1, r2 in itertools.product(range(n), repeat=2):
            c1, c2 = k - r1, k - r2
            if not (0 <= c1 < n and 0 <= c2 < n): continue
            if -1 in (grid[r1][c1], grid[r2][c2]): continue
            prev = max(dp[a][b] for a in (r1, r1 - 1)
                       for b in (r2, r2 - 1) if a >= 0 and b >= 0)
            if prev < 0: continue
            got = grid[r1][c1] + (grid[r2][c2] if r1 != r2 else 0)
            new[r1][r2] = prev + got
        dp = new
    return max(dp[n - 1][n - 1], 0)


# ---- the alternatives that fail -------------------------------------

def paths(m, n):
    """Every right/down path from (0,0) to (m-1,n-1), as cells."""
    for downs in itertools.combinations(range(m + n - 2), m - 1):
        i = j = 0
        cells = [(0, 0)]
        for step in range(m + n - 2):
            if step in downs:
                i += 1
            else:
                j += 1
            cells.append((i, j))
        yield cells


def need_for(dungeon, cells):
    """Start health a single path needs: 1 - (lowest running sum)."""
    total = low = 0
    for i, j in cells:
        total += dungeon[i][j]
        low = min(low, total)
    return 1 - low


def forward_dungeon(dungeon, keep):
    """Forward DP keeping ONE path per cell, chosen by `keep`.

    Each candidate is (low, total): lowest running sum so far and
    health change so far. keep='hp' keeps the highest total;
    keep='need' keeps the highest low (= lowest start needed).
    """
    m, n = len(dungeon), len(dungeon[0])
    key = ((lambda t: (t[1], t[0])) if keep == "hp"
           else (lambda t: t))
    best = {}
    for i in range(m):
        for j in range(n):
            before = [best[p] for p in ((i - 1, j), (i, j - 1))
                      if p in best] or [(0, 0)]
            after = []
            for low, total in before:
                total += dungeon[i][j]
                after.append((min(low, total), total))
            best[i, j] = max(after, key=key)
    return 1 - best[m - 1, n - 1][0]


def greedy_two_trips(grid):
    """Best one path, erase its cherries, best path again.

    Tries every maximal first path, so it is greedy's best case.
    """
    n = len(grid)
    ok = [p for p in paths(n, n)
          if all(grid[i][j] != -1 for i, j in p)]
    if not ok:
        return 0
    score = lambda p, skip=(): sum(grid[i][j] for i, j in p
                                   if (i, j) not in skip)
    top = max(score(p) for p in ok)
    return max(top + max(score(q, set(p)) for q in ok)
               for p in ok if score(p) == top)


def brute_cherries(grid):
    n = len(grid)
    ok = [p for p in paths(n, n)
          if all(grid[i][j] != -1 for i, j in p)]
    return max((sum(grid[i][j] for i, j in set(a) | set(b))
                for a in ok for b in ok), default=0)


def brute_hp(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    return min(need_for(dungeon, p) for p in paths(m, n))


def need_table(dungeon):
    """Full 2-D backward table, for printing the trace."""
    m, n = len(dungeon), len(dungeon[0])
    INF = float("inf")
    need = [[INF] * (n + 1) for _ in range(m + 1)]
    need[m][n - 1] = 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            best_next = min(need[i + 1][j], need[i][j + 1])
            need[i][j] = max(best_next - dungeon[i][j], 1)
    return [row[:n] for row in need[:m]]


def cherry_trace(grid):
    """Print dp[i1][i2] (i1 <= i2) for every diagonal k."""
    n = len(grid)
    dp = {(0, 0): grid[0][0]}
    print("k=0", dp)
    for k in range(1, 2 * n - 1):
        new = {}
        for i1 in range(n):
            for i2 in range(i1, n):
                j1, j2 = k - i1, k - i2
                if not (0 <= j1 < n and 0 <= j2 < n):
                    continue
                if -1 in (grid[i1][j1], grid[i2][j2]):
                    continue
                prev = [dp.get(tuple(sorted((a, b))), -1)
                        for a in (i1, i1 - 1) for b in (i2, i2 - 1)]
                if max(prev) < 0:
                    continue
                got = grid[i1][j1] + (grid[i2][j2] if i1 != i2 else 0)
                new[i1, i2] = max(prev) + got
        dp = new
        print(f"k={k}", dp)


if __name__ == "__main__":
    print(min_hp([[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]))  # → 7
    print(cherry_pickup([[0, 1, -1], [1, 0, -1], [1, 1, 1]]))  # → 5

    dungeon = [[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]
    print("Dungeon", dungeon)
    for row in need_table(dungeon):
        print("  need", row)
    print("min_hp:", min_hp(dungeon))                  # 7
    print("forward, keep most health:",
          forward_dungeon(dungeon, "hp"))              # 8, wrong
    tiny = [[0, -1, 2], [0, 0, 0], [0, 0, -2]]
    print("tiny", tiny, "true", min_hp(tiny),
          "forward, keep lowest need:",
          forward_dungeon(tiny, "need"))               # 2 vs 3
    for p in paths(3, 3):
        print("  path", p, "needs", need_for(dungeon, p))

    grid = [[0, 1, -1], [1, 0, -1], [1, 1, 1]]
    print("Cherry", grid, "->", cherry_pickup(grid))  # 5
    cherry_trace(grid)
    print("greedy two trips:", greedy_two_trips(grid))  # 5 here
    trap = [[0, 0, 0, 0], [0, 0, 1, 0],
            [1, 1, 1, 1], [0, 1, 0, 0]]
    print("trap", trap, "dp", cherry_pickup(trap),
          "greedy", greedy_two_trips(trap))            # 6 vs 5

    random.seed(0)
    for _ in range(300):
        m, n = random.randint(1, 5), random.randint(1, 5)
        d = [[random.randint(-9, 9) for _ in range(n)]
             for _ in range(m)]
        assert min_hp(d) == brute_hp(d)
        n = random.randint(1, 5)
        g = [[random.choice([0, 0, 1, 1, -1]) for _ in range(n)]
             for _ in range(n)]
        g[0][0] = g[-1][-1] = 0
        assert cherry_pickup(g) == brute_cherries(g)
    print("300 random dungeons and 300 random fields match brute force")

    n = 50
    print(f"n={n}: one-way paths C({2*n-2},{n-1}) =",
          f"{math.comb(2 * n - 2, n - 1):.3e};",
          f"DP states (2n-1)*n*n = {(2 * n - 1) * n * n:,}")
