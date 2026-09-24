"""Beta(2, 5): pdf, closed-form CDF, and quantiles by bisection.

Beta(2, 5) is the 2nd smallest of 6 uniforms, so
F(y) = P(at least 2 of 6 uniforms <= y) = P(Bin(6, y) >= 2).
"""
import random
from fractions import Fraction
from math import comb


def pdf(y):
    return 30 * y * (1 - y) ** 4        # 30 = 6! / (1! * 4!)


def cdf(y):
    return 1 - (1 - y) ** 6 - 6 * y * (1 - y) ** 5


def beta_cdf(a, b, y):
    """Any whole-number a, b: P(Bin(a + b - 1, y) >= a)."""
    n = a + b - 1
    return sum(comb(n, j) * y**j * (1 - y) ** (n - j)
               for j in range(a, n + 1))


def quantile(p, F=cdf, tol=1e-10, trace=False):
    lo, hi, steps = 0.0, 1.0, 0
    while hi - lo > tol:                 # F rises, so halve [lo, hi]
        mid = (lo + hi) / 2
        steps += 1
        if trace and steps <= 10:
            move = "lo = mid" if F(mid) < p else "hi = mid"
            print(f"  {steps:2}  [{lo:.4f}, {hi:.4f}]  "
                  f"mid {mid:.4f}  F {F(mid):.4f}  {move}")
        if F(mid) < p:
            lo = mid                     # answer is to the right
        else:
            hi = mid
    return (lo + hi) / 2, steps


def newton(p, y, n=3):
    for _ in range(n):
        y -= (cdf(y) - p) / pdf(y)
        print(f"  newton y = {y:.4f}")


if __name__ == "__main__":
    h = Fraction(1, 2)
    print("F(1/2) exact:", 1 - h**6 - 6 * h * h**5)
    for p in (0.5, 0.95):
        print(f"p = {p}")
        q, k = quantile(p, trace=True)
        print(f"  -> y = {q:.5f} after {k} halvings")
    print("mean 2/7 =", round(2 / 7, 5), " mode 1/5, var 5/196")
    z = 1.6448536                        # N(0,1) 95th percentile
    sd = (5 / 196) ** 0.5
    y = 2 / 7 + z * sd
    print(f"normal approx q95 {y:.4f}, F there {cdf(y):.4f}")
    print("Newton for q95 from y = 0.9:")
    newton(0.95, 0.9)
    random.seed(0)
    s = sorted(random.betavariate(2, 5) for _ in range(10**5))
    print(f"sim 10^5: median {s[50_000]:.4f}"
          f"  q95 {s[95_000]:.4f}  mean {sum(s) / 1e5:.4f}")
    for a, b in ((2, 5), (20, 50), (5, 12)):
        def F(y):
            return beta_cdf(a, b, y)
        qs = [quantile(p, F)[0] for p in (0.05, 0.5, 0.95)]
        print(f"Beta({a},{b}): 5% {qs[0]:.4f}  median "
              f"{qs[1]:.4f}  95% {qs[2]:.4f}")
