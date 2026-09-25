"""Reading a distribution off its MGF.

Solves the two MGF problems in the source guide > Univariate Random
Variables: m(t) = 1/6 e^t + 2/6 e^2t + 3/6 e^3t (mean, variance,
distribution) and M(t) = (0.7 / (1 - 0.3 e^t))^5 (which distribution?).
Moments come from derivatives at t = 0; the NB match is checked
against the closed form, by finite differences and by simulation.
"""
from fractions import Fraction as F
from math import comb, exp

import numpy as np


def moments_from_terms(terms):
    """terms: {k: c} for M(t) = sum c*e^(k t). Returns E[X], Var."""
    m1 = sum(c * k for k, c in terms.items())       # M'(0)
    m2 = sum(c * k * k for k, c in terms.items())   # M''(0)
    return m1, m2, m2 - m1 * m1


def nb_mgf(t, r, p):
    """MGF of NB(r, p) counting failures before the r-th success."""
    return (p / (1 - (1 - p) * exp(t))) ** r


def nb_pmf(k, r, p):
    return comb(k + r - 1, r - 1) * p**r * (1 - p) ** k


def derivs_at_0(f, h=1e-4):
    """M'(0) and M''(0) by central differences."""
    d1 = (f(h) - f(-h)) / (2 * h)
    d2 = (f(h) - 2 * f(0) + f(-h)) / (h * h)
    return d1, d2


if __name__ == "__main__":
    # Problem 1: each e^(k t) term is "value k with probability c"
    terms = {1: F(1, 6), 2: F(2, 6), 3: F(3, 6)}
    m1, m2, var = moments_from_terms(terms)
    print("Y: pmf", {k: str(c) for k, c in terms.items()})
    print(f"   M'(0) = E[Y] = {m1}, M''(0) = E[Y^2] = {m2},"
          f" Var = {var}")
    print(f"   = {float(m1):.4f}, {float(m2):.4f}, {float(var):.4f}")

    # Problem 2: match (p / (1 - q e^t))^r  ->  r = 5, p = 0.7
    r, p = 5, F(7, 10)
    q = 1 - p
    mean, var = r * q / p, r * q / p**2
    print(f"X ~ NB({r}, {p}) failures: mean {mean} = {float(mean):.4f},"
          f" var {var} = {float(var):.4f}")
    print("   pmf k=0..4:",
          [round(nb_pmf(k, r, 0.7), 5) for k in range(5)])

    # the pmf's own MGF equals the given closed form
    t = 0.2
    series = sum(nb_pmf(k, r, 0.7) * exp(t * k) for k in range(400))
    print(f"   M(0.2): closed {nb_mgf(t, r, 0.7):.6f},"
          f" series {series:.6f}")
    d1, d2 = derivs_at_0(lambda s: nb_mgf(s, r, 0.7))
    print(f"   M'(0) ~ {d1:.4f}, M''(0) ~ {d2:.4f},"
          f" var ~ {d2 - d1 * d1:.4f}")

    # numpy's negative_binomial counts failures too
    rng = np.random.default_rng(0)
    x = rng.negative_binomial(r, 0.7, size=10**6)
    print(f"   sim 10^6 (seed 0): mean {x.mean():.4f},"
          f" var {x.var():.4f}")

    # the wrong catalogue entry: trials form has e^(r t) on top
    print(f"   trials form (p e^t/(1-q e^t))^5 would give mean"
          f" {r / p} = {float(r / p):.4f}")
