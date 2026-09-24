"""State-machine DP: Best Time to Buy and Sell Stock III and IV.

From the cheat-sheet's Recursion page (III in C++ with four states,
IV in Python with a k-row table), rewritten in Python. A state is
"what I am holding after j buys / j sells"; each day every state
either rests or takes one edge (buy or sell) from its neighbour.
Runs under python3 (3.10).
"""
import itertools
import random


def max_profit_2(prices):
    """Stock III: at most two transactions, four numbers."""
    c1 = c2 = float("inf")   # cheapest effective cost of buy 1 / 2
    p1 = p2 = 0              # best profit after sell 1 / 2
    for x in prices:
        c1 = min(c1, x)          # buy 1: pay x
        p1 = max(p1, x - c1)     # sell 1: collect x - c1
        c2 = min(c2, x - p1)     # buy 2: pay x, already up p1
        p2 = max(p2, x - c2)     # sell 2
    return p2


def max_profit_k(k, prices):
    """Stock IV: at most k transactions, 2k numbers."""
    if k >= len(prices) // 2:    # k can't bind: take every rise
        return sum(max(b - a, 0)
                   for a, b in itertools.pairwise(prices))
    cost = [float("inf")] * (k + 1)
    profit = [0] * (k + 1)       # profit[0] = 0: no trades yet
    for x in prices:
        for j in range(1, k + 1):
            cost[j] = min(cost[j], x - profit[j - 1])
            profit[j] = max(profit[j], x - cost[j])
    return profit[k]


def source_iv(k, prices):
    """The source's Python for IV, verbatim apart from names."""
    n = len(prices)
    if k < n >> 1:
        dp = [[0] * n for _ in range(2)]
        for j in range(1, k + 1):
            dp[1][0] = 0
            profit = dp[0][0] - prices[0]
            for i in range(1, n):
                dp[1][i] = max(dp[1][i - 1], profit + prices[i])
                profit = max(profit, dp[0][i] - prices[i])
            dp[0] = dp[1][:]
        return dp[0][-1]
    return sum(max(b - a, 0) for a, b in itertools.pairwise(prices))


def brute(k, prices):
    """Try every set of <= k non-overlapping (buy, sell) pairs."""
    n = len(prices)
    best = 0

    def go(start, left, acc):
        nonlocal best
        best = max(best, acc)
        if left == 0:
            return
        for b in range(start, n):
            for s in range(b + 1, n):
                go(s, left - 1, acc + prices[s] - prices[b])

    go(0, k, 0)
    return best


def greedy_runs(k, prices):
    """Wrong: keep the k biggest rising runs."""
    runs, lo = [], 0
    for i in range(1, len(prices) + 1):
        if i == len(prices) or prices[i] < prices[i - 1]:
            runs.append(prices[i - 1] - prices[lo])
            lo = i
    return sum(sorted(runs, reverse=True)[:k])


def trace(prices):
    c1 = c2 = float("inf")
    p1 = p2 = 0
    print("day price   c1  p1   c2  p2")
    for d, x in enumerate(prices):
        c1 = min(c1, x)
        p1 = max(p1, x - c1)
        c2 = min(c2, x - p1)
        p2 = max(p2, x - c2)
        print(f"{d:>3} {x:>5} {c1:>4} {p1:>3} {c2:>4} {p2:>3}")
    return p2


if __name__ == "__main__":
    prices = [3, 3, 5, 0, 0, 3, 1, 4]
    print("answer", trace(prices))
    for k in range(0, 6):
        print(f"k={k}: {max_profit_k(k, prices)}")
    print("falling [7,6,4,3,1]:", max_profit_2([7, 6, 4, 3, 1]))
    print("rising  [1,2,3,4,5]:", max_profit_2([1, 2, 3, 4, 5]))
    g = [1, 5, 4, 8, 2, 6]
    print(f"{g} k=2: greedy runs {greedy_runs(2, g)},"
          f" state machine {max_profit_k(2, g)}")
    n = 8
    pairs = [(b, s) for b in range(n) for s in range(b + 1, n)]
    two = sum(1 for (b1, s1) in pairs for (b2, s2) in pairs
              if s1 <= b2)
    print(f"n={n}: {len(pairs)} single trades, {two} ordered pairs")

    random.seed(0)
    for _ in range(2000):
        ps = [random.randint(0, 9) for _ in range(random.randint(0, 9))]
        k = random.randint(0, 5)
        want = brute(k, ps)
        assert max_profit_k(k, ps) == want
        assert source_iv(k, ps) == want
        assert max_profit_2(ps) == brute(2, ps)
    print("2000 random cases match brute force (source IV too)")
