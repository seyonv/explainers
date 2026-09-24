"""Random Pick with Blacklist: one random call per pick.

Draw x from [0, m) where m = n - len(blacklist). The m slots below m
hold every allowed value once: the blacklisted ones below m are
remapped to the allowed values at or above m.
"""
import math
import random
from collections import Counter


class Solution:
    def __init__(self, n, blacklist):
        banned = set(blacklist)
        self.m = n - len(banned)          # draw from [0, m)
        self.remap = {}
        top = n - 1
        for b in blacklist:
            if b < self.m:                # a hole in the draw range
                while top in banned:      # skip banned values up top
                    top -= 1
                self.remap[b] = top       # fill the hole with top
                top -= 1

    def pick(self):
        x = random.randrange(self.m)      # the only random call
        return self.remap.get(x, x)


class Rejection:
    """Baseline: draw from [0, n) until the value is allowed."""

    def __init__(self, n, blacklist):
        self.n, self.banned, self.calls = n, set(blacklist), 0

    def pick(self):
        while True:
            self.calls += 1
            x = random.randrange(self.n)
            if x not in self.banned:
                return x


def chi2_sf_df3(x):
    # P(chi-square with 3 degrees of freedom > x), closed form
    return (math.erfc(math.sqrt(x / 2))
            + math.sqrt(2 * x / math.pi) * math.exp(-x / 2))


def check_bijection(trials=1000):
    # every allowed value appears exactly once among the m slots
    for _ in range(trials):
        n = random.randint(1, 40)
        bl = random.sample(range(n), random.randint(0, n - 1))
        s = Solution(n, bl)
        got = sorted(s.remap.get(x, x) for x in range(s.m))
        want = [v for v in range(n) if v not in set(bl)]
        assert got == want, (n, bl)
    return trials


if __name__ == "__main__":
    s = Solution(7, [2, 3, 5])
    print("m =", s.m, " remap =", s.remap)
    print("slots:", [s.remap.get(x, x) for x in range(s.m)])

    random.seed(2026)
    N = 70_000
    counts = Counter(s.pick() for _ in range(N))
    exp = N / s.m
    for v in sorted(counts):
        print(f"  {v}: {counts[v]:,}  ({counts[v] - exp:+,.0f})")
    chi2 = sum((c - exp) ** 2 / exp for c in counts.values())
    print(f"chi2 = {chi2:.2f}, df = 3, p = {chi2_sf_df3(chi2):.2f}")

    r = Rejection(7, [2, 3, 5])
    for _ in range(N):
        r.pick()
    print(f"rejection: {r.calls:,} calls for {N:,} picks"
          f" = {r.calls / N:.3f} per pick (expected 7/4 = 1.75)")

    print("bijection checks passed:", check_bijection())
