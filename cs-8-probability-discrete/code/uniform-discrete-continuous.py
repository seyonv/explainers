"""Uniform distributions, discrete and continuous.

Reference strips (mean, variance) for the discrete uniform on 1..n and
the continuous uniform on (a, b), checked by direct summation, then the
source's problem: Y ~ U(0, 5); when does 4x^2 + 4xY + Y + 2 = 0 have
real roots? Solved exactly, then checked with 10**6 draws.
"""
import random
from fractions import Fraction as F


def discrete_uniform(n):
    """Mean and variance of X uniform on {1, ..., n}, by summation."""
    xs = range(1, n + 1)
    mean = F(sum(xs), n)
    var = F(sum(x * x for x in xs), n) - mean**2
    return mean, var          # = (n + 1)/2, (n*n - 1)/12


def continuous_uniform(a, b):
    """Mean and variance of Y ~ U(a, b), from the closed forms."""
    a, b = F(a), F(b)
    return (a + b) / 2, (b - a) ** 2 / 12


def disc(y):
    """Discriminant of 4x^2 + 4yx + (y + 2)."""
    return (4 * y) ** 2 - 4 * 4 * (y + 2)   # = 16 (y - 2)(y + 1)


def p_real_roots(a=0, b=5):
    """P(disc(Y) >= 0) for Y ~ U(a, b): good length / total length.

    16(y - 2)(y + 1) >= 0 outside the roots: y <= -1 or y >= 2.
    """
    good = max(0, b - max(a, 2)) + max(0, min(b, -1) - a)
    return F(good, b - a)


def simulate(trials=10**6, seed=0):
    rng = random.Random(seed)
    hits = sum(disc(rng.uniform(0, 5)) >= 0 for _ in range(trials))
    return hits / trials


if __name__ == "__main__":
    for n in (6, 10):
        m, v = discrete_uniform(n)
        print(f"discrete 1..{n}: mean {m}, var {v}"
              f" = (n^2-1)/12 = {F(n * n - 1, 12)}")
    m, v = continuous_uniform(0, 5)
    print(f"U(0, 5): mean {m}, var {v}")
    for y in (1, 2, 3):
        print(f"Y = {y}: discriminant {disc(y)}")
    print("P(real roots) =", p_real_roots())         # 3/5
    print("source's interval (0, 2) =", F(2, 5), "(wrong)")
    print(f"sim, 10^6 draws: {simulate():.4f}")
