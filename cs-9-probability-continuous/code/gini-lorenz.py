"""Lorenz curve and Gini index for payroll density f(x) = 3(1-x)^2.

F(x) = share of payroll earned by the top fraction x of employees.
G = 2 * integral_0^1 |x - F(x)| dx, exactly, by Simpson's rule,
and from a simulated company of 10^5 employees.
"""
import random
from fractions import Fraction


def top_share(x):
    """F(x) = integral_0^x 3(1-t)^2 dt = 1 - (1-x)^3."""
    return 1 - (1 - x) ** 3


def simpson(f, a, b, n=1000):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


def gini_curve(F):
    """G = 2 * area between the curve and the diagonal."""
    return 2 * simpson(lambda x: abs(x - F(x)), 0, 1)


def gini_sample(pay):
    """Gini of a list of pay: mean |a - b| over pairs / (2 mean)."""
    w = sorted(pay)
    n = len(w)
    ranked = sum((2 * i - n + 1) * v for i, v in enumerate(w))
    return ranked / (n * sum(w))


def simulate(n=10**5, seed=0):
    rng = random.Random(seed)
    # employee at rank u from the top is paid in proportion to f(u)
    return gini_sample(3 * (1 - rng.random()) ** 2
                       for _ in range(n))


if __name__ == "__main__":
    area_F = 1 - Fraction(1, 4)          # integral of 1 - (1-x)^3
    G = 2 * (area_F - Fraction(1, 2))
    print("exact: area under F =", area_F, " G =", G)
    print("Simpson:", round(gini_curve(top_share), 6))
    print("simulated 10^5 employees:", round(simulate(), 4))
    print(" x    top x     bottom x")
    for x in (0.1, 0.2, 0.5, 0.8):
        print(f"{x:.1f}  {top_share(x):.3f}     "
              f"{1 - top_share(1 - x):.3f}")
    for k in (0, 1, 2, 3):              # f = (k+1)(1-x)^k
        g = gini_curve(lambda x: 1 - (1 - x) ** (k + 1))
        print(f"k={k}: G = {g:.4f}  (k/(k+2) = {k / (k + 2):.4f})")
    half_zero = [0] * 50 + [1] * 50      # half paid nothing
    print("half paid 0, half equal: G =", gini_sample(half_zero))
