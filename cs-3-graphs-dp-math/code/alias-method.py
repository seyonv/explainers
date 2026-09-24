"""Weighted random choice: the alias method (Random Pick with Weight).

Card: cs-3-graphs-dp-math/alias-method.html
Source: ljeng/cheat-sheet, coding-algorithms/mathematics.md,
Probability, Random Pick with Weight. The source's version is correct:
it scales each weight by n so every column holds exactly sum(w), pairs
one underfull with one overfull index per column, and picks a column,
then a coin. Rewritten here with two lists as stacks (same pop order
as the source's dict.popitem, so the same columns) and an integer coin
(randrange) instead of uniform().
Runs under python3 (3.10).
"""
import bisect
import itertools
import math
import random


class AliasTable:
    def __init__(self, w):
        n, total = len(w), sum(w)
        self.total = total
        h = [x * n for x in w]          # scaled: columns hold total
        small = [i for i, x in enumerate(h) if x < total]
        large = [i for i, x in enumerate(h) if x >= total]
        self.cols = []                  # (keep, low index, alias)
        while small and large:
            s, g = small.pop(), large.pop()
            self.cols.append((h[s], s, g))
            h[g] -= total - h[s]        # g fills the rest of s's column
            (small if h[g] < total else large).append(g)
        for g in large:                 # leftovers are exactly full
            self.cols.append((total, g, g))

    def pick(self):
        keep, low, alias = random.choice(self.cols)
        r = random.randrange(self.total)    # exact integer coin
        return low if r < keep else alias


class PrefixSum:                        # the usual answer: O(log n)
    def __init__(self, w):
        self.cum = list(itertools.accumulate(w))

    def pick(self):
        r = random.randrange(self.cum[-1])
        return bisect.bisect_right(self.cum, r)


def trace(w):
    n, total = len(w), sum(w)
    h = [x * n for x in w]
    print(f"w = {w}, n = {n}, total = {total}, n*w = {h}")
    small = [i for i, x in enumerate(h) if x < total]
    large = [i for i, x in enumerate(h) if x >= total]
    step = 1
    while small and large:
        s, g = small.pop(), large.pop()
        before = h[g]
        h[g] -= total - h[s]
        side = "small" if h[g] < total else "large"
        (small if h[g] < total else large).append(g)
        print(f"step {step}: small {s} ({h[s]}) + large {g} ({before})"
              f" -> column [{s}:{h[s]} | {g}:{total - h[s]}],"
              f" {g} left {h[g]} -> {side}")
        step += 1
    for g in large:
        print(f"leftover: column [{g}:{total}]")


def column_shares(table, n):
    p = [0.0] * n
    for keep, low, alias in table.cols:
        p[low] += keep / table.total / n
        p[alias] += (table.total - keep) / table.total / n
    return p


if __name__ == "__main__":
    w = [1, 3, 2, 4]
    trace(w)
    t = AliasTable(w)
    print("columns (keep, low, alias):", t.cols)
    print("P from columns:",
          [round(x, 10) for x in column_shares(t, len(w))])

    random.seed(0)
    N = 100_000
    counts = [0] * len(w)
    for _ in range(N):
        counts[t.pick()] += 1
    print(f"{N:,} draws:", counts)
    for i, c in enumerate(counts):
        p = w[i] / sum(w)
        se = math.sqrt(N * p * (1 - p))
        print(f"  index {i}: expected {N * p:,.0f} +- {se:.0f},"
              f" got {c:,} ({(c - N * p) / se:+.2f} sd)")
    chi2 = sum((c - N * x / sum(w)) ** 2 / (N * x / sum(w))
               for c, x in zip(counts, w))
    print(f"  chi-square {chi2:.2f} on 3 df (5% cutoff 7.81)")

    random.seed(0)
    ps = PrefixSum(w)
    counts2 = [0] * len(w)
    for _ in range(N):
        counts2[ps.pick()] += 1
    print("prefix sums + bisect:", ps.cum, counts2)

    for n in (4, 10**4, 10**6):
        print(f"n = {n:>9,}: bisect ~{math.ceil(math.log2(n + 1))}"
              f" comparisons per pick, alias 1 column + 1 compare")

    random.seed(1)
    for _ in range(200):
        ws = [random.randint(1, 50)
              for _ in range(random.randint(1, 12))]
        tb = AliasTable(ws)
        assert len(tb.cols) == len(ws)
        got = column_shares(tb, len(ws))
        assert all(abs(g - x / sum(ws)) < 1e-12
                   for g, x in zip(got, ws))
    print("200 random weight lists: columns give exactly w/sum(w)")
