"""Poisson distribution: sums of Poissons, moments, the binomial limit.

Solves the two problems in the source guide > Univariate Random Variables >
Poisson (three highways; the claims problem the source leaves unsolved),
checks that Poisson(0.3) + Poisson(0.5) + Poisson(0.7) is Poisson(1.5)
by convolution and by a 10**6-day simulation, and measures how fast
Binomial(n, lam/n) approaches Poisson(lam).
"""
from fractions import Fraction as F
from math import comb, exp, factorial

import numpy as np


def pois(k, lam):
    return exp(-lam) * lam**k / factorial(k)


def binom(k, n, p):
    return comb(n, k) * p**k * (1 - p)**(n - k)   # comb = 0 if k > n


def tv(n, lam, kmax=60):
    """Total variation distance, Binomial(n, lam/n) vs Poisson(lam)."""
    return sum(abs(binom(k, n, lam / n) - pois(k, lam))
               for k in range(kmax)) / 2


def convolve3(k, a, b, c):
    """P(X + Y + Z = k) for independent Poissons, term by term."""
    return sum(pois(i, a) * pois(j, b) * pois(k - i - j, c)
               for i in range(k + 1) for j in range(k + 1 - i))


def simulate(rates=(0.3, 0.5, 0.7), days=10**6, seed=0):
    rng = np.random.default_rng(seed)
    total = sum(rng.poisson(r, days) for r in rates)
    return total.mean(), total.var(), (total == 0).mean()


if __name__ == "__main__":
    # 1. Highways: independent Poissons add, and so do their rates
    lam = 0.3 + 0.5 + 0.7
    print(f"lam = {lam:.1f}   P(0) = {pois(0, lam):.4f}")
    for k in range(6):
        print(f"  k={k}  conv {convolve3(k, 0.3, 0.5, 0.7):.4f}"
              f"  Poisson(1.5) {pois(k, lam):.4f}")
    m, v, p0 = simulate()
    print(f"sim: mean {m:.4f}  var {v:.4f}  P(0) {p0:.4f}")

    # 2. Claims: P(X=2) = 4 P(X=3); P(3)/P(2) = lam/3 = 1/4
    lam2 = F(3, 4)
    ex2 = lam2 + lam2**2                 # E[X^2] = Var + mean^2
    print(f"lam = {lam2}   E[X^2] = {ex2} = {float(ex2)}")

    # 3. Bin(n, 1.5/n) -> Poisson(1.5)
    for n in (10, 100, 1000):
        print(f"n={n:<5} P(0): {binom(0, n, lam / n):.5f}"
              f" vs {pois(0, lam):.5f}   TV {tv(n, lam):.5f}"
              f"   bound lam^2/n {lam**2 / n:.5f}")
