"""How big a sample?

The cheat-sheet's gas-mileage problem: sd 3.3 mpg, and Cassie wants
to be at least 95% sure the sample mean is within 1 mpg of the true
mean. Solve n >= (z * sigma / E)^2 and round up; check the coverage
of n = 41, 42 and 43 exactly (normal) and by simulation; compare
with Chebyshev's distribution-free answer; then a table of n for
90/95/99% and three margins.
"""
import math

import numpy as np


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def norm_ppf(p, lo=-10.0, hi=10.0):
    """Inverse of norm_cdf by bisection (no scipy)."""
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if norm_cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def sample_size(sigma, margin, conf):
    """Smallest n with P(|mean - mu| < margin) >= conf (CLT)."""
    z = norm_ppf(1 - (1 - conf) / 2)       # two-sided: 2.5% each tail
    return math.ceil((z * sigma / margin) ** 2), z


def coverage(n, sigma, margin):
    """P(|mean - mu| < margin) when the mean is N(mu, sigma^2/n)."""
    return 2 * norm_cdf(margin * math.sqrt(n) / sigma) - 1


def chebyshev_size(sigma, margin, conf):
    """Smallest n with sigma^2 / (n margin^2) <= 1 - conf."""
    return math.ceil(sigma**2 / (margin**2 * (1 - conf)))


def simulate(n, sigma=3.3, margin=1.0, reps=10**5, seed=0):
    """Share of samples whose mean lands within margin of mu.

    mu = 30 mpg is illustrative; the answer doesn't depend on it.
    Two shapes with sd 3.3: normal, and a right-skewed gamma.
    """
    rng = np.random.default_rng(seed)
    mu = 30.0
    normal = rng.normal(mu, sigma, (reps, n))
    k = 4.0                                  # gamma shape: skew 1.0
    theta = sigma / math.sqrt(k)
    gamma = rng.gamma(k, theta, (reps, n)) - k * theta + mu
    hit = lambda x: np.mean(np.abs(x.mean(axis=1) - mu) < margin)
    return hit(normal), hit(gamma)


if __name__ == "__main__":
    n, z = sample_size(3.3, 1.0, 0.95)
    raw = (z * 3.3 / 1.0) ** 2
    print(f"z = {z:.4f}  (z*sigma/E)^2 = {raw:.2f}  -> n = {n}")
    print(f"with z = 1.96: (1.96*3.3)^2 = {(1.96 * 3.3) ** 2:.2f}")
    for m in (30, 41, 42, 43):
        e = z * 3.3 / math.sqrt(m)
        print(f"n = {m}: margin {e:.3f} mpg, "
              f"P(within 1) = {coverage(m, 3.3, 1.0):.4f}")
    print("Chebyshev n =", chebyshev_size(3.3, 1.0, 0.95),
          f"(10.89 / 0.05 = {3.3**2 / 0.05:.1f})")
    p = norm_ppf(0.975)
    print("poll, p = 0.5, E = 0.03:",
          math.ceil((p * 0.5 / 0.03) ** 2),
          f"({(p * 0.5 / 0.03) ** 2:.1f})")

    print("\nn for sd 3.3 mpg")
    print("conf    z      E=2   E=1  E=0.5")
    for conf in (0.90, 0.95, 0.99):
        row = [sample_size(3.3, e, conf)[0] for e in (2, 1, 0.5)]
        zc = sample_size(3.3, 1, conf)[1]
        print(f"{conf:.0%}  {zc:.4f}  {row[0]:4d}  {row[1]:4d}"
              f"  {row[2]:5d}")

    print("\nsimulated P(within 1 mpg), 10^5 samples each, seed 0")
    for m in (41, 42):
        a, b = simulate(m)
        print(f"n = {m}: normal {a:.4f}  skewed gamma {b:.4f}")
