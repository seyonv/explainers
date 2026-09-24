"""Conditioning on a sum: the car-accident problem.

cheat-sheet > Multivariate Random Variables > Associated Applications.
Driver and passenger are each hospitalized with probability 0.3; each
hospital loss is U(0, 1), independent. Given total loss < 1, what is
the expected number hospitalized?  Answer: 102/191 = 0.5340.
The trick: P(N = n | L < c) is prior x P(L < c | N = n), normalised.
"""
from fractions import Fraction as F
from math import comb, factorial

import numpy as np


def p_sum_below(n, c):
    """P(U1 + ... + Un < c) for iid U(0,1): the Irwin-Hall CDF."""
    return sum((-1) ** k * comb(n, k) * (c - k) ** n
               for k in range(int(c) + 1) if c - k > 0) / factorial(n)


def given_sum_below(p, people, c):
    """Rows (n, prior, likelihood, joint, posterior) given L < c."""
    rows = []
    for n in range(people + 1):
        prior = comb(people, n) * p ** n * (1 - p) ** (people - n)
        like = F(p_sum_below(n, c))
        rows.append((n, prior, like, prior * like))
    total = sum(r[3] for r in rows)        # P(L < c)
    table = [(n, pr, lk, j, j / total) for n, pr, lk, j in rows]
    return table, total


def sum_density(n, x):
    """Density of U1 + ... + Un at x > 0 (n = 0 has none there)."""
    if n == 0:
        return 0
    total = sum((-1) ** k * comb(n, k) * (x - k) ** (n - 1)
                for k in range(int(x) + 1) if x - k > 0)
    return total / factorial(n - 1)


def given_sum_equals(p, people, x):
    """Same table, but the evidence is L = x exactly: use densities."""
    joint = [comb(people, n) * p ** n * (1 - p) ** (people - n)
             * F(sum_density(n, x)) for n in range(people + 1)]
    total = sum(joint)
    return sum(n * j for n, j in enumerate(joint)) / total


def mean_n(table):
    return sum(n * post for n, _, _, _, post in table)


def simulate(trials=10**6, seed=0):
    rng = np.random.default_rng(seed)
    hosp = rng.random((trials, 2)) < 0.3
    loss = (hosp * rng.random((trials, 2))).sum(axis=1)
    keep = loss < 1
    return hosp[keep].sum(axis=1).mean(), keep.mean()


if __name__ == "__main__":
    table, total = given_sum_below(F(3, 10), 2, F(1))
    for n, pr, lk, j, post in table:
        print(f"N={n}  {float(pr):.2f} x {str(lk):>3}"
              f" = {float(j):.3f} -> {float(post):.4f}")
    ans = mean_n(table)
    print(f"P(L<1) = {float(total):.3f}")
    print(f"E[N | L<1] = {ans} = {float(ans):.4f}")
    print("E[N] with no condition =", float(2 * F(3, 10)))
    for c in (F(1, 2), F(1), F(3, 2), F(2)):
        t, _ = given_sum_below(F(3, 10), 2, c)
        e = float(mean_n(t))
        print(f"c = {float(c):.1f}: E[N | L<c] = {e:.4f}")
    e = given_sum_equals(F(3, 10), 2, F(4, 5))
    print(f"E[N | L = 0.8] = {e} = {float(e):.4f}")
    m, frac = simulate()
    print(f"sim: E[N | L<1] = {m:.4f}, P(L<1) = {frac:.4f}")
