"""Negative binomial: the source's NB(5, 0.7), built three ways.

cheat-sheet > Univariate Random Variables > Negative Binomial gives
M(t) = (0.7 / (1 - 0.3 e^t))^5 and the answer NB(5, 0.70). This file
checks that X (failures before the 5th success) is
  1. the closed-form pmf C(k+r-1, k) p^r q^k,
  2. the sum of 5 independent geometrics (exact convolution),
  3. a Poisson whose rate is gamma-distributed (simulation),
and compares it with a Poisson that has the same mean.
Python 3.10; the simulations need numpy.
"""
from fractions import Fraction as F
from math import comb, exp, factorial

import numpy as np


def nb_pmf(k, r, p):
    """P(X = k), X = failures before the r-th success."""
    return comb(k + r - 1, k) * p**r * (1 - p) ** k


def sum_of_geometrics(r, p, kmax):
    """pmf of G1 + ... + Gr by convolution, each G ~ geometric."""
    geo = [p * (1 - p) ** k for k in range(kmax + 1)]
    dist = [F(1)] + [F(0)] * kmax        # the sum of zero geometrics
    for _ in range(r):
        dist = [sum(dist[j] * geo[k - j] for j in range(k + 1))
                for k in range(kmax + 1)]
    return dist


def poisson_pmf(k, lam):
    return exp(-lam) * lam**k / factorial(k)


if __name__ == "__main__":
    r, p = 5, F(7, 10)
    q = 1 - p
    mean, var = r * q / p, r * q / p**2
    print(f"NB({r}, {p}): mean {mean} = {float(mean):.4f},"
          f" var {var} = {float(var):.4f}")
    print(f"  var = mean + mean^2/r = {mean} + {mean**2 / r}"
          f" = {mean + mean**2 / r}")

    conv = sum_of_geometrics(r, p, 8)
    print("  k  closed     convolution  cumulative   Poisson(15/7)")
    cum = 0
    for k in range(9):
        c = nb_pmf(k, r, p)
        cum += c
        same = "same" if c == conv[k] else "DIFF"
        pois = poisson_pmf(k, float(mean))
        print(f"  {k}  {float(c):.4f}     {same}         "
              f"{float(cum):.4f}       {pois:.4f}")

    lam = float(mean)
    nb_tail = 1 - float(sum(nb_pmf(k, r, p) for k in range(6)))
    po_tail = 1 - sum(poisson_pmf(k, lam) for k in range(6))
    print(f"  P(X >= 6): NB {nb_tail:.4f}  Poisson {po_tail:.4f}"
          f"  ratio {nb_tail / po_tail:.2f}")
    print(f"  P(X = 0):  NB {float(nb_pmf(0, r, p)):.4f}"
          f"  Poisson {poisson_pmf(0, lam):.4f}")

    rng = np.random.default_rng(0)
    n = 10**6
    # driver's rate ~ Gamma(shape r, scale q/p); claims ~ Poisson
    rate = rng.gamma(r, 3 / 7, size=n)
    claims = rng.poisson(rate)
    # numpy's geometric counts trials (1, 2, ...): subtract 1
    geo_sum = (rng.geometric(0.7, size=(n, r)) - 1).sum(axis=1)
    for name, x in [("gamma-Poisson", claims),
                    ("sum of 5 geometrics", geo_sum)]:
        print(f"  sim {name:20} mean {x.mean():.4f}"
              f"  var {x.var():.4f}  P(0) {np.mean(x == 0):.4f}")
    print(f"  trials form Y = X + 5: mean {r / p} = {float(r / p):.4f}")
