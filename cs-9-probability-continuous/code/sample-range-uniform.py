"""Range R = max - min of n iid U(0, 1) draws.

pdf  f(r) = n(n-1) r^(n-2) (1-r),  0 < r < 1   (Beta(n-1, 2))
CDF  F(r) = n r^(n-1) - (n-1) r^n
E[R] = (n-1)/(n+1),  Var(R) = 2(n-1) / ((n+1)^2 (n+2))

Why: pick which draw is the min (n ways) and which is the max
(n-1 ways); the other n-2 must land in the window of width r
(r^(n-2)); the window's left edge can sit anywhere in [0, 1-r].
"""
import random
from fractions import Fraction as F


def range_pdf(r, n):
    return n * (n - 1) * r ** (n - 2) * (1 - r)


def range_cdf(r, n):                  # integral of the pdf, 0 to r
    return n * r ** (n - 1) - (n - 1) * r ** n


def mean_var(n):
    return F(n - 1, n + 1), F(2 * (n - 1), (n + 1) ** 2 * (n + 2))


def simulate(n, trials, bins=10, seed=0):
    rng = random.Random(seed)
    counts, total = [0] * bins, 0.0
    for _ in range(trials):
        ys = [rng.random() for _ in range(n)]
        r = max(ys) - min(ys)
        total += r
        counts[min(int(r * bins), bins - 1)] += 1
    return counts, total / trials


def median(n, tol=1e-12):
    lo, hi = 0.0, 1.0
    while hi - lo > tol:              # the CDF rises: bisect
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if range_cdf(mid, n) < 0.5 else (lo, mid)
    return (lo + hi) / 2


if __name__ == "__main__":
    n, trials = 5, 10**5
    counts, sim_mean = simulate(n, trials)
    print(f"n = {n}: f(r) = {n * (n - 1)} r^{n - 2} (1 - r)")
    print("bin         exact    sim     sim/exact")
    for i, c in enumerate(counts):
        a, b = i / 10, (i + 1) / 10
        p = range_cdf(b, n) - range_cdf(a, n)
        print(f"[{a:.1f}, {b:.1f})  {p:.4f}  {c / trials:.4f}"
              f"  {c / trials / p:6.3f}")
    m, v = mean_var(n)
    print(f"E[R] = {m} = {float(m):.4f}   sim {sim_mean:.4f}")
    print(f"Var(R) = {v} = {float(v):.5f}  sd {float(v)**.5:.4f}")
    print(f"mode (n-2)/(n-1) = {(n - 2) / (n - 1)}"
          f"  median {median(n):.4f}")
    wrong = F(2 * n, (n + 1) ** 2 * (n + 2))
    print(f"if min, max were independent: Var = {wrong}"
          f" = {float(wrong):.5f}")
    for k in (2, 3, 5, 10, 100):
        m, _ = mean_var(k)
        print(f"n = {k:3}: E[R] = {m} = {float(m):.4f}"
              f"  unbiased width = R * {F(k + 1, k - 1)}")
