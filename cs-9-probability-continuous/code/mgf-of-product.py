"""MGF of a product U = Y1 * Y2 of two independent N(0, 1).

Condition on Y2: given Y2 = y, U = y * Y1 is N(0, y^2), so
E[e^(tU) | Y2] = e^(t^2 Y2^2 / 2). Averaging over Y2 gives
M_U(t) = (1 - t^2)^(-1/2) for |t| < 1. Moments come from the
Taylor series; E and Var are checked directly (independence)
and by simulation.
"""
import math
import random
from fractions import Fraction as F


def mgf(t):
    """M_U(t) = E[e^(tU)]; finite only for |t| < 1."""
    if abs(t) >= 1:
        return math.inf
    return (1 - t * t) ** -0.5


def inner(t, y):
    """E[e^(tU) | Y2 = y]: the MGF of N(0, y^2) at t."""
    return math.exp(t * t * y * y / 2)


def simpson(f, a, b, n=4000):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


def phi(y):
    return math.exp(-y * y / 2) / math.sqrt(2 * math.pi)


def series_moment(k):
    """E[U^(2k)] = (2k)! * [t^(2k)] (1 - t^2)^(-1/2)."""
    coef = F(math.comb(2 * k, k), 4 ** k)   # binomial series
    return math.factorial(2 * k) * coef


def normal_even_moment(k):
    """E[Y^(2k)] = (2k - 1)!! for Y ~ N(0, 1)."""
    return math.prod(range(1, 2 * k, 2))


def derivative_moments(h=1e-4):
    """M'(0) and M''(0) by central differences."""
    m1 = (mgf(h) - mgf(-h)) / (2 * h)
    m2 = (mgf(h) - 2 * mgf(0) + mgf(-h)) / h ** 2
    return m1, m2


def simulate(n, rng):
    return [rng.gauss(0, 1) * rng.gauss(0, 1) for _ in range(n)]


if __name__ == "__main__":
    t = 0.3
    avg = simpson(lambda y: inner(t, y) * phi(y), -12, 12)
    print(f"t = {t}: E_Y2[e^(t^2 Y2^2/2)] by Simpson = {avg:.6f}")
    print(f"        (1 - t^2)^(-1/2)             = {mgf(t):.6f}")
    print(f"t = 1.0: M_U = {mgf(1.0)} (integrand e^(y^2(t^2-1)/2)"
          " stops decaying)")

    m1, m2 = derivative_moments()
    print(f"M'(0) = {m1:.6f} -> E[U] = 0")
    print(f"M''(0) = {m2:.6f} -> E[U^2] = 1, Var = 1 - 0^2 = 1")

    print("even moments: series vs E[Y^2k]^2 (independence)")
    for k in (1, 2, 3):
        c = F(math.comb(2 * k, k), 4 ** k)
        print(f"  k={k}: (2k)! * {c} = {series_moment(k)}"
              f"   vs ({normal_even_moment(k)})^2"
              f" = {normal_even_moment(k) ** 2}")

    rng = random.Random(0)
    n = 10 ** 6
    us = simulate(n, rng)
    mean = sum(us) / n
    var = sum((u - mean) ** 2 for u in us) / (n - 1)
    m4 = sum(u ** 4 for u in us) / n
    emgf = sum(math.exp(t * u) for u in us) / n
    print(f"sim (n=10^6, seed 0): mean {mean:+.4f}"
          f" (se {math.sqrt(1 / n):.4f}), var {var:.4f},"
          f" E[U^4] {m4:.2f}, E[e^(0.3U)] {emgf:.4f}")
    y = [rng.gauss(0, 1) for _ in range(n)]
    tail_u = sum(abs(u) > 3 for u in us) / n
    tail_y = sum(abs(v) > 3 for v in y) / n
    print(f"P(|U| > 3) = {tail_u:.4f}  vs  P(|Y| > 3) = {tail_y:.4f}")
    print(f"exact P(|Y| > 3) = {math.erfc(3 / math.sqrt(2)):.4f}")
