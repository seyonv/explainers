"""Pareto(alpha=3, x_m=1): f(x) = 3x^-4 for x > 1.

P(X < 2 | X >= 1.5), heavy tail vs exponential tail, and a
simulation check by inverse transform.
"""
import math
import random
from fractions import Fraction


def sf(x, alpha=3, xm=1):
    """Survival P(X > x) = (xm / x)^alpha for x >= xm."""
    return 1 if x < xm else (xm / x) ** alpha


def cond(a, b, alpha=3):
    """P(X < b | X >= a) = 1 - S(b)/S(a) = 1 - (a/b)^alpha."""
    return 1 - sf(b, alpha) / sf(a, alpha)


def simpson(f, a, b, n=1000):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


def simulate(n=10**6, seed=0):
    rng = random.Random(seed)
    kept = hits = 0
    for _ in range(n):
        x = (1 - rng.random()) ** (-1 / 3)  # inverse of 1 - x^-3
        if x >= 1.5:
            kept += 1
            hits += x < 2
    return hits / kept, kept


if __name__ == "__main__":
    exact = 1 - Fraction(3, 4) ** 3       # (1.5/2)^3 = 27/64
    print("P(X<2 | X>=1.5) =", exact, "=", float(exact))
    print("via sf:", round(cond(1.5, 2), 6))
    num = simpson(lambda x: 3 * x**-4, 1.5, 2)
    print("Simpson num / den:", round(num, 6), "/",
          round(sf(1.5), 6), "=", round(num / sf(1.5), 6))
    p, kept = simulate()
    se = math.sqrt(p * (1 - p) / kept)
    print(f"sim: {p:.4f} from {kept:,} draws >= 1.5"
          f" (se {se:.4f})")
    print("P(X > x), three models that all have mean 1.5:")
    print(" x    Pareto     Exp(1.5)   1+Exp(0.5)")
    for x in (2, 5, 10, 20, 100):
        e = math.exp(-x / 1.5)
        s = math.exp(-2 * (x - 1))
        print(f"{x:>3}  {sf(x):.3e}  {e:.3e}  {s:.3e}")
