"""Adding independent random variables: the convolution sum.

Solves the weekly-claims problem in cheat-sheet > General Probability >
Basic Concepts of Probability: P(N = n) = 1/2**(n+1) each week, weeks
independent; P(exactly 7 claims in two weeks) = 1/64. Then checks it
three ways: the generating-function formula, numpy.convolve, and a
simulation of 10**6 two-week periods.
"""
from fractions import Fraction as F
from math import erf, sqrt

import numpy as np


def pmf_N(n):
    return F(1, 2 ** (n + 1))           # claims in one week, n >= 0


def convolve(p, q, s):
    """P(X + Y = s) for independent X, Y >= 0 with pmfs p, q."""
    return sum(p(k) * q(s - k) for k in range(s + 1))


def pmf_S(s):
    """Coefficient of z**s in G(z)**2 = 1/(2 - z)**2."""
    return F(s + 1, 2 ** (s + 2))


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


if __name__ == "__main__":
    total = F(0)
    print(" k  P(N1=k)  P(N2=7-k)  product  running")
    for k in range(8):
        a, b = pmf_N(k), pmf_N(7 - k)
        total += a * b
        print(f"{k:2}  {str(a):>7}  {str(b):>9}  {str(a * b):>7}"
              f"  {str(total):>7}")
    print(convolve(pmf_N, pmf_N, 7), pmf_S(7))
    p = np.array([0.5 ** (n + 1) for n in range(60)])
    print(np.convolve(p, p)[7])            # all totals at once
    print("pmf of S:", [str(pmf_S(s)) for s in range(9)])

    rng = np.random.default_rng(0)
    # numpy's geometric counts trials (>= 1); subtract 1 for failures
    n1 = rng.geometric(0.5, 10**6) - 1
    n2 = rng.geometric(0.5, 10**6) - 1
    hit = np.mean(n1 + n2 == 7)
    se = sqrt(hit * (1 - hit) / 10**6)
    print(f"sim: {hit:.5f} +/- {se:.5f} (exact {1/64:.5f})")

    # wrong shortcuts
    print("2N = 7 is impossible: P = 0")
    mu, sd = 2, 2                          # E[S] = 2, Var[S] = 4
    nrm = norm_cdf((7.5 - mu) / sd) - norm_cdf((6.5 - mu) / sd)
    print(f"normal approx: {nrm:.5f}")
    print("P(N1=7) + P(N2=7):", pmf_N(7) + pmf_N(7))
    print("dice sum 7:", convolve(
        lambda k: F(1, 6) if 1 <= k <= 6 else F(0),
        lambda k: F(1, 6) if 1 <= k <= 6 else F(0), 7))
