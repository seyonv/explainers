"""Inclusion-exclusion: add the singles, subtract the pairs, add back
the triples, and so on.

Solves the two inclusion-exclusion problems in the source guide > General
Probability > Discrete Mathematics (multiples of neither 6 nor 9, and
young female single policyholders), then checks both by brute force.
"""
import random
from itertools import combinations
from math import lcm


def count_none(n, divisors):
    """How many of 1..n are divisible by none of the divisors."""
    total = 0
    for k in range(len(divisors) + 1):
        for group in combinations(divisors, k):
            # lcm() of no numbers is 1, so k = 0 adds n itself
            total += (-1) ** k * (n // lcm(*group))
    return total


def union_size(sets):
    """|A1 u A2 u ... u Am| from the sizes of every intersection."""
    total = 0
    for k in range(1, len(sets) + 1):
        for group in combinations(sets, k):
            total += (-1) ** (k + 1) * len(set.intersection(*group))
    return total


def regions(y, m, w, ym, mw, yw, ymw, n):
    """The 8 Venn regions (young?, male?, married?) from the totals."""
    r = {(1, 1, 1): ymw,
         (1, 1, 0): ym - ymw, (0, 1, 1): mw - ymw, (1, 0, 1): yw - ymw}
    r[1, 0, 0] = y - ym - yw + ymw
    r[0, 1, 0] = m - ym - mw + ymw
    r[0, 0, 1] = w - mw - yw + ymw
    r[0, 0, 0] = n - (y + m + w) + (ym + mw + yw) - ymw
    return r


def brute_none(n, divisors):
    return sum(all(i % d for d in divisors) for i in range(1, n + 1))


if __name__ == "__main__":
    # Problem 1: multiples of neither 6 nor 9 among 1..1000
    print("1000//6 =", 1000 // 6, " 1000//9 =", 1000 // 9,
          " 1000//lcm(6,9) =", 1000 // lcm(6, 9))
    print("neither 6 nor 9:", count_none(1000, [6, 9]),
          "brute:", brute_none(1000, [6, 9]))

    # Problem 2: young, female, single
    given = dict(y=3000, m=4600, w=7000, ym=1320, mw=3010,
                 yw=1400, ymw=600, n=10_000)
    r = regions(**given)
    print("regions:", {"".join(map(str, k)): v for k, v in r.items()})
    print("young female single = 3000 - 1320 - 1400 + 600 =",
          3000 - 1320 - 1400 + 600)

    # Brute force: build all 10,000 policyholders, recount everything
    people = [k for k, v in r.items() for _ in range(v)]
    young = {i for i, p in enumerate(people) if p[0]}
    male = {i for i, p in enumerate(people) if p[1]}
    married = {i for i, p in enumerate(people) if p[2]}
    assert (len(young), len(male), len(married)) == (3000, 4600, 7000)
    assert len(young & male) == 1320 and len(male & married) == 3010
    assert len(young & married) == 1400
    assert len(young & male & married) == 600
    yfs = young - male - married
    print("brute young female single:", len(yfs))
    u = union_size([young, male, married])
    print(f"union: {u} (brute {len(young | male | married)}),"
          f" none of the three: {len(people) - u}")

    # 1,000 random cases against the brute force
    rng = random.Random(0)
    for _ in range(1000):
        n = rng.randint(1, 2000)
        ds = rng.sample(range(2, 30), rng.randint(1, 4))
        assert count_none(n, ds) == brute_none(n, ds)
    print("1,000 random (n, divisors) cases match the brute force")
