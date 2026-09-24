"""Coupon collector: rolls of a fair die until all six faces appear.

The waiting time T is a sum of independent geometric stages, so
E[T] = n * H_n and Var(T) = n^2 * sum(1/k^2) - n * H_n.
Python 3.10, standard library only.
"""
import random
from fractions import Fraction as F
from itertools import combinations
from math import comb


def stages(n):
    """(p, E, Var) of each geometric stage: a new face w.p. p."""
    out = []
    for seen in range(n):
        p = F(n - seen, n)                 # chance the next is new
        out.append((p, 1 / p, (1 - p) / p**2))
    return out


def mean_var(n):
    st = stages(n)
    return sum(e for _, e, _ in st), sum(v for _, _, v in st)


def cdf(n, t):
    """P(T <= t) by inclusion-exclusion over the missed faces."""
    return sum((-1)**j * comb(n, j) * (1 - F(j, n))**t
               for j in range(n + 1))


def mean_unequal(ps):
    """E[T] when face i has probability ps[i] (inclusion-exclusion)."""
    return sum((-1)**(r + 1) / sum(s)
               for r in range(1, len(ps) + 1)
               for s in combinations(ps, r))


def simulate(n, trials=10**5, seed=0):
    rng = random.Random(seed)
    xs = []
    for _ in range(trials):
        seen, rolls = set(), 0
        while len(seen) < n:
            seen.add(rng.randrange(n))
            rolls += 1
        xs.append(rolls)
    m = sum(xs) / trials
    return m, sum((x - m)**2 for x in xs) / (trials - 1)


if __name__ == "__main__":
    for k, (p, e, v) in enumerate(stages(6), 1):
        print(f"stage {k}: p={str(p):>4}  E={float(e):.1f}"
              f"  Var={float(v):.2f}")
    E, V = mean_var(6)
    print(f"E = {E} = {float(E)},  Var = {V} = {float(V)}")
    print(f"SD = {float(V) ** 0.5:.3f}")
    med = next(t for t in range(1, 100) if cdf(6, t) >= F(1, 2))
    print(f"median = {med},  P(T <= 6) = {float(cdf(6, 6)):.4f}"
          f",  P(T > 30) = {float(1 - cdf(6, 30)):.4f}")
    m, s2 = simulate(6)
    print(f"sim (10^5 runs, seed 0): mean {m:.3f}  var {s2:.2f}")
    for n in (6, 10, 52, 100, 365):
        E, V = mean_var(n)
        print(f"n={n:>3}: E={float(E):8.2f}  SD={float(V)**0.5:7.2f}")
    loaded = [F(1, 2)] + [F(1, 10)] * 5
    E = mean_unequal(loaded)
    print(f"loaded die (1/2, 1/10 x5): E = {E} = {float(E):.2f}")
