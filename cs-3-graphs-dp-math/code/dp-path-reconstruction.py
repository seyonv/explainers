"""Coin Path: cheapest path, lexicographically smallest on ties.

Card: cs-3-graphs-dp-math/dp-path-reconstruction.html
Source: ljeng/cheat-sheet, coding-algorithms/recursion.md,
Coin Path (C++). Rewritten in Python; same algorithm: fill
cost[] from the end backwards, keep the smallest next index on
ties, then walk the next[] pointers from index 1.
Runs under python3 (3.10).
"""
import random

INF = float("inf")


def cheapest_jump(coins, max_jump):
    n = len(coins)
    cost = [INF] * n                 # cost[i]: cheapest i -> end
    nxt = [-1] * n                   # nxt[i]: first step of it
    if coins[-1] != -1:
        cost[-1] = coins[-1]
    for i in range(n - 2, -1, -1):
        if coins[i] == -1:
            continue                 # blocked: stays INF
        for j in range(i + 1, min(i + max_jump, n - 1) + 1):
            if coins[i] + cost[j] < cost[i]:   # strict <: a tie
                cost[i] = coins[i] + cost[j]   # keeps the
                nxt[i] = j                     # smaller j
    if cost[0] == INF:
        return []
    path, i = [], 0
    while i != -1:                   # walk the pointers
        path.append(i + 1)           # answer is 1-indexed
        i = nxt[i]
    return path


def trace(coins, max_jump):
    """Same fill, printing each cell as the card's table does."""
    n = len(coins)
    cost, nxt = [INF] * n, [-1] * n
    cost[-1] = coins[-1]
    print(f"i={n}  coin {coins[-1]:>2}  end        "
          f"cost {cost[-1]}")
    for i in range(n - 2, -1, -1):
        if coins[i] == -1:
            print(f"i={i + 1}  coin -1  blocked    cost inf")
            continue
        opts = []
        for j in range(i + 1, min(i + max_jump, n - 1) + 1):
            opts.append(f"{j + 1}:{coins[i] + cost[j]}")
            if coins[i] + cost[j] < cost[i]:
                cost[i], nxt[i] = coins[i] + cost[j], j
        print(f"i={i + 1}  coin {coins[i]:>2}  "
              f"{' '.join(opts):<10} cost {cost[i]}"
              f"  next {nxt[i] + 1}")


def forward_smallest_parent(coins, max_jump):
    """The tempting wrong version: DP from the start, keep the
    smallest predecessor on ties, walk back from the end."""
    n = len(coins)
    cost, par = [INF] * n, [-1] * n
    cost[0] = coins[0]
    for j in range(1, n):
        if coins[j] == -1:
            continue
        for i in range(max(0, j - max_jump), j):
            if cost[i] + coins[j] < cost[j]:
                cost[j], par[j] = cost[i] + coins[j], i
    if cost[-1] == INF:
        return []
    path, j = [], n - 1
    while j != -1:
        path.append(j + 1)
        j = par[j]
    return path[::-1]


def all_paths(coins, max_jump):
    """Every legal path from index 1 to index n, 1-indexed."""
    n = len(coins)
    out = []

    def go(i, path):
        if i == n - 1:
            out.append(path)
            return
        for j in range(i + 1, min(i + max_jump, n - 1) + 1):
            if coins[j] != -1:
                go(j, path + [j + 1])

    go(0, [1])
    return out


def brute(coins, max_jump):
    paths = all_paths(coins, max_jump)
    if not paths:
        return []
    return min(paths, key=lambda p: (
        sum(coins[k - 1] for k in p), p))   # cost, then lex


def check(trials=3000, seed=0):
    rng = random.Random(seed)
    wrong = 0                        # forward version's misses
    for _ in range(trials):
        n = rng.randint(1, 9)
        coins = [rng.choice([-1, 0, 1, 2, 3]) for _ in range(n)]
        coins[0] = max(coins[0], 0)          # coins[1] != -1
        b = rng.randint(1, 4)
        want = brute(coins, b)
        assert cheapest_jump(coins, b) == want, coins
        wrong += forward_smallest_parent(coins, b) != want
    return trials, wrong


if __name__ == "__main__":
    A, B = [1, 2, 4, -1, 2], 2
    trace(A, B)
    print("path  ", cheapest_jump(A, B))            # [1, 3, 5]
    print("brute ", brute(A, B), "of", all_paths(A, B))

    Z = [0, 0, 0, 0]                                # all ties
    print("ties  ", sorted(all_paths(Z, 2)))
    print("back  ", cheapest_jump(Z, 2))            # [1,2,3,4]
    print("front ", forward_smallest_parent(Z, 2))  # [1,2,4]

    print("unreachable", cheapest_jump([1, 2, 4, -1, 2], 1))
    ok, wrong = check()
    print(f"random tests passed: {ok}; forward version "
          f"wrong on {wrong}")

    for n in (10, 20, 30):
        print(f"paths, n={n}, B=2:",
              len(all_paths([1] * n, 2)))
