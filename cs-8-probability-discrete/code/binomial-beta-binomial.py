"""Binomial and beta-binomial: what happens when p itself is random.

Answers cheat-sheet > Univariate Random Variables > Binomial:
X ~ Bin(n, p) with p ~ Beta(a, b); find the pmf of X. Worked with
n = 10, a = b = 2 (mean 5, same as Bin(10, 1/2)), exact Fractions,
then checked with a simulation of 10**6 draws.
"""
from fractions import Fraction as F
from math import comb, factorial as fact

import numpy as np


def B(a, b):
    """Beta function for integer a, b: (a-1)!(b-1)!/(a+b-1)!."""
    return F(fact(a - 1) * fact(b - 1), fact(a + b - 1))


def binom_pmf(n, k, p):
    return comb(n, k) * p**k * (1 - p)**(n - k)


def betabinom_pmf(n, k, a, b):
    # integrate Bin(n, p) against the Beta(a, b) density over p
    return comb(n, k) * B(k + a, n - k + b) / B(a, b)


def mean_var(pmf):
    m = sum(k * q for k, q in enumerate(pmf))
    return m, sum((k - m)**2 * q for k, q in enumerate(pmf))


def simulate(n, a, b, trials=10**6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.binomial(n, rng.beta(a, b, trials))  # draw p, then X
    return x.mean(), x.var(), (x == 0).mean()


if __name__ == "__main__":
    n, a, b = 10, 2, 2
    bi = [binom_pmf(n, k, F(1, 2)) for k in range(n + 1)]
    bb = [betabinom_pmf(n, k, a, b) for k in range(n + 1)]
    print(" k  Bin(10,1/2)  BetaBin(10,2,2)")
    for k in range(n + 1):
        print(f"{k:2}  {float(bi[k]):.4f}       "
              f"{float(bb[k]):.4f}  = {bb[k]}")
    print("Bin     mean, var:", *mean_var(bi))
    print("BetaBin mean, var:", *mean_var(bb))
    # law of total variance: E[Var(X|p)] + Var(E[X|p])
    Ep, Ep2 = F(a, a + b), F(a * (a + 1), (a + b) * (a + b + 1))
    within, between = n * (Ep - Ep2), n * n * (Ep2 - Ep**2)
    print("within + between =", within, "+", between,
          "=", within + between)
    print("P(X=0) ratio BetaBin/Bin:", bb[0] / bi[0])
    m, v, p0 = simulate(n, a, b)
    print(f"sim: mean {m:.4f}  var {v:.4f}  P(X=0) {p0:.4f}")
