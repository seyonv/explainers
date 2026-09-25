"""How much lies within k standard deviations? Six discrete answers.

Solves the coverage problem in the source guide > Univariate Random
Variables > Discrete Univariate Distributions: P(|X - mu| < k*sigma)
for k = 1, 2, 3, exactly, for six distributions, next to the normal
curve and Chebyshev's bound 1 - 1/k**2.
"""
from fractions import Fraction as F
from math import comb, erf, exp, factorial, floor, ceil, sqrt


def coverage(pmf, mu, var, k, strict=True):
    """P(|X - mu| < k*sigma): sum pmf over the integers inside."""
    # squares compared as Fractions: exact even when mu +- k*sigma
    # is an integer, as for Poisson(4)
    r = k * sqrt(var)                       # only to bound the loop
    lo, hi = floor(mu - r) - 1, ceil(mu + r) + 1
    inside = [x for x in range(lo, hi + 1) if pmf(x) and (
              (x - mu) ** 2 < k * k * var if strict
              else (x - mu) ** 2 <= k * k * var)]
    return sum(pmf(x) for x in inside), inside


def zoo():
    """name -> (pmf, mean, variance); pmf is 0 off the support."""
    h = F(1, 2)
    q = F(3, 4)                             # geometric failure prob
    return {
        "Uniform{1..10}": (lambda x: F(1, 10) if 1 <= x <= 10 else 0,
                           F(11, 2), F(99, 12)),
        "Bin(10, 0.6)": (lambda x: comb(10, x) * F(3, 5) ** x
                         * F(2, 5) ** (10 - x) if 0 <= x <= 10 else 0,
                         F(6), F(12, 5)),
        "Hyper(5; 10, 30)": (lambda x: F(comb(10, x) * comb(30, 5 - x),
                                         comb(40, 5))
                             if 0 <= x <= 5 else 0,
                             F(5, 4), F(5 * 10 * 30 * 35,
                                        40 * 40 * 39)),
        "Poisson(4)": (lambda x: exp(-4) * 4 ** x / factorial(x)
                       if x >= 0 else 0, F(4), F(4)),
        "Geom(0.25)": (lambda x: q ** (x - 1) / 4 if x >= 1 else 0,
                       F(4), F(12)),
        "NB(3, 0.5)": (lambda x: comb(x - 1, 2) * h ** x
                       if x >= 3 else 0, F(6), F(6)),
    }


def normal_within(k):
    return erf(k / sqrt(2))


if __name__ == "__main__":
    nb = lambda x: comb(x - 1, 2) * F(1, 2) ** x if x >= 3 else 0
    for k in (1, 2, 3):                     # NB(3, 0.5): mu 6, var 6
        p, xs = coverage(nb, F(6), F(6), k)
        print(k, f"{xs[0]}..{xs[-1]}", p, f"{float(p):.4f}",
              f"normal {erf(k / sqrt(2)):.4f}",
              f"cheb {1 - 1 / k**2:.4f}")
    print()
    print(f"{'':18}{'k=1':>8}{'k=2':>8}{'k=3':>8}")
    for name, (pmf, mu, var) in zoo().items():
        row = []
        for k in (1, 2, 3):
            p, xs = coverage(pmf, mu, var, k)
            row.append(float(p))
        print(f"{name:18}" + "".join(f"{v:8.4f}" for v in row))
    print(f"{'Normal':18}"
          + "".join(f"{normal_within(k):8.4f}" for k in (1, 2, 3)))
    print(f"{'Chebyshev >=':18}"
          + "".join(f"{1 - 1 / k**2:8.4f}" for k in (1, 2, 3)))

    for name, (pmf, mu, var) in zoo().items():   # moments check
        xs = range(150)          # tails past 150 are < 1e-17
        m = sum(x * pmf(x) for x in xs)
        v = sum((x - m) ** 2 * pmf(x) for x in xs)
        assert abs(m - mu) < 1e-9 and abs(v - var) < 1e-9, name

    print("\nk = 1 trace: mu, sigma, integers inside, sum")
    for name, (pmf, mu, var) in zoo().items():
        p, xs = coverage(pmf, mu, var, 1)
        print(f"  {name:18} mu={float(mu):5.2f} sd={sqrt(var):.3f}"
              f"  {xs[0]}..{xs[-1]}  {float(p):.4f}")

    pmf, mu, var = zoo()["Poisson(4)"]
    print("\nPoisson(4): the edges mu +- k*sigma are integers")
    for k in (1, 2, 3):
        lt, a = coverage(pmf, mu, var, k)
        le, b = coverage(pmf, mu, var, k, strict=False)
        print(f"  k={k}  <: {a[0]}..{a[-1]} {float(lt):.4f}"
              f"   <=: {b[0]}..{b[-1]} {float(le):.4f}")

    # Chebyshev is tight: X in {-k, 0, k}, P(X = +-k) = 1/(2k^2)
    for k in (2, 3):
        p0 = 1 - F(1, k * k)
        var = 2 * k * k * F(1, 2 * k * k)
        print(f"three-point k={k}: var={var}, P(|X|<k)={p0}")
