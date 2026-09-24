"""From recursion to dynamic programming, in plain Python.

Card: cs-3-graphs-dp-math/recursion-to-dp.html
Source: ljeng/cheat-sheet, coding-algorithms/recursion.md. The page
jumps straight to hard DP problems, so this card is supplemented:
Fibonacci shows the three forms (naive, memoized, table), and the
source's Coin Path example shows why greedy is not enough.
"""
import sys
import time
from functools import lru_cache

calls = 0


def fib_naive(n):                    # the recursion, as written
    global calls
    calls += 1
    return n if n < 2 else fib_naive(n - 1) + fib_naive(n - 2)


@lru_cache(maxsize=None)             # top-down: same code + a cache
def fib_memo(n):
    return n if n < 2 else fib_memo(n - 1) + fib_memo(n - 2)


def fib_table(n):                    # bottom-up: small to large
    dp = [0, 1] + [0] * (n - 1)
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


def fib_two(n):                      # keep only the last two cells
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def naive_calls(n):
    """C(n) = 1 + C(n-1) + C(n-2), C(0) = C(1) = 1  ->  2F(n+1) - 1."""
    global calls
    calls = 0
    fib_naive(n)
    return calls


def max_memo_depth():
    """Largest n a cold fib_memo(n) reaches without RecursionError."""
    lo, hi = 1, 5000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        fib_memo.cache_clear()
        try:
            fib_memo(mid)
            lo = mid
        except RecursionError:
            hi = mid - 1
    fib_memo.cache_clear()
    return lo


def max_dict_depth():
    """Same test with a plain dict memo: one frame per level."""
    def fib_dict(n, memo):
        if n not in memo:
            memo[n] = fib_dict(n - 1, memo) + fib_dict(n - 2, memo)
        return memo[n]
    lo, hi = 1, 5000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            fib_dict(mid, {0: 0, 1: 1})
            lo = mid
        except RecursionError:
            hi = mid - 1
    return lo


def coin_path_dp(coins, max_jump):
    """Cheapest path by backward DP (the source's Coin Path)."""
    n = len(coins)
    INF = float("inf")
    cost, nxt = [INF] * n, [-1] * n
    if coins[-1] != -1:
        cost[-1] = coins[-1]
    for i in range(n - 2, -1, -1):
        if coins[i] == -1:
            continue
        for j in range(i + 1, min(i + max_jump, n - 1) + 1):
            if coins[i] + cost[j] < cost[i]:  # strict: smallest j
                cost[i], nxt[i] = coins[i] + cost[j], j
    if cost[0] == INF:
        return [], INF
    path, i = [], 0
    while i != -1:
        path.append(i + 1)
        i = nxt[i]
    return path, cost[0]


def coin_path_greedy(coins, max_jump):
    """Always jump to the cheapest reachable next cell (wrong)."""
    n, i, path, total = len(coins), 0, [1], coins[0]
    while i != n - 1:
        options = [j for j in range(i + 1, min(i + max_jump, n - 1) + 1)
                   if coins[j] != -1]
        if not options:
            return path, None
        i = min(options, key=lambda j: (coins[j], j))
        path.append(i + 1)
        total += coins[i]
    return path, total


if __name__ == "__main__":
    print("naive calls:")
    for n in (5, 10, 20, 30):
        c = naive_calls(n)
        print(f"  fib({n:2}) = {fib_table(n):>7,}  calls = {c:>9,}")
    fib_memo.cache_clear()
    print("memo fib(30) =", fib_memo(30), fib_memo.cache_info())
    print("table fib(30) =", fib_table(30), " two vars =", fib_two(30))
    print("table dp[0..10] =", [fib_table(i) for i in range(11)])

    t0 = time.perf_counter()
    fib_naive(30)
    t1 = time.perf_counter()
    fib_memo.cache_clear()
    fib_memo(30)
    t2 = time.perf_counter()
    fib_two(30)
    t3 = time.perf_counter()
    print(f"one run: naive {t1 - t0:.3f} s (with call counter), "
          f"memo {(t2 - t1) * 1e6:.1f} us, "
          f"two vars {(t3 - t2) * 1e6:.1f} us")

    print("recursion limit:", sys.getrecursionlimit())
    print("largest cold fib_memo(n):", max_memo_depth())
    print("largest cold fib_dict(n):", max_dict_depth())
    try:
        fib_memo(1000)
    except RecursionError as e:
        print("fib_memo(1000): RecursionError:", e)
    big = fib_two(10**5)
    print("fib_two(10**5): no recursion, bit_length =",
          big.bit_length())

    coins = [1, 2, 4, -1, 2]
    print("coin path DP:    ", coin_path_dp(coins, 2))
    print("coin path greedy:", coin_path_greedy(coins, 2))
