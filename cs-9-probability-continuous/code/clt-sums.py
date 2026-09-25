"""Sums and the central limit theorem.

The source guide's Linear Combinations problems, solved with one
recipe: the mean of a sum is n*mu, the variance is n*sigma^2, and
for large n the sum is close to normal. A.J. and M.J.'s 20 jobs,
the charity's 90th percentile and the Poisson trees; then checks:
the exact Poisson sum and a gamma-jobs simulation.
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


def sum_of(n, mu, sd):
    """Mean and sd of the sum of n iid values."""
    return n * mu, sd * math.sqrt(n)


def poisson_cdf(k, lam):
    """Exact P(X <= k) for Poisson(lam), summed in log space."""
    return sum(math.exp(i * math.log(lam) - lam - math.lgamma(i + 1))
               for i in range(k + 1))


def solve():
    out = {}
    a_mu, a_sd = sum_of(20, 50, 10)            # A.J.
    m_mu, m_sd = sum_of(20, 52, 15)            # M.J.
    out["A.J. < 900"] = norm_cdf((900 - a_mu) / a_sd)
    out["M.J. < 900"] = norm_cdf((900 - m_mu) / m_sd)
    d_mu, d_sd = m_mu - a_mu, math.hypot(a_sd, m_sd)   # D = M - A
    out["A.J. first"] = 1 - norm_cdf((0 - d_mu) / d_sd)
    c_mu, c_sd = sum_of(2025, 3125, 250)       # charity
    out["charity p90"] = c_mu + norm_ppf(0.90) * c_sd
    t_mu, t_sd = sum_of(100, 60, math.sqrt(60))   # trees
    z = lambda x: (x - t_mu) / t_sd
    out["trees, cc"] = norm_cdf(z(6100.5)) - norm_cdf(z(5949.5))
    out["trees, no cc"] = norm_cdf(z(6100)) - norm_cdf(z(5950))
    out["trees, exact"] = (poisson_cdf(6100, 6000)
                           - poisson_cdf(5949, 6000))
    return out


def simulate_jobs(reps=10**6, seed=0):
    """Assume gamma job times with the given mean and sd."""
    rng = np.random.default_rng(seed)

    def totals(mu, sd):
        k, theta = (mu / sd) ** 2, sd * sd / mu
        return rng.gamma(k, theta, size=(reps, 20)).sum(axis=1)

    a, m = totals(50, 10), totals(52, 15)
    return (a < 900).mean(), (m < 900).mean(), (a < m).mean()


if __name__ == "__main__":
    a_mu, a_sd = sum_of(20, 50, 10)
    m_mu, m_sd = sum_of(20, 52, 15)
    c_mu, c_sd = sum_of(2025, 3125, 250)
    t_mu, t_sd = sum_of(100, 60, math.sqrt(60))
    print(f"A.J. total: mean {a_mu}, sd {a_sd:.2f}, "
          f"z = {(900 - a_mu) / a_sd:.4f}")
    print(f"M.J. total: mean {m_mu}, sd {m_sd:.2f}, "
          f"z = {(900 - m_mu) / m_sd:.4f}")
    d_sd = math.hypot(a_sd, m_sd)
    print(f"M - A: mean {m_mu - a_mu}, var {a_sd**2 + m_sd**2:.0f}, "
          f"sd {d_sd:.2f}, z = {(m_mu - a_mu) / d_sd:.4f}")
    print(f"charity: mean {c_mu:,}, sd {c_sd:,.0f}, "
          f"z90 = {norm_ppf(0.90):.4f}")
    print(f"trees: mean {t_mu}, sd {t_sd:.3f}, "
          f"z = {(5949.5 - t_mu) / t_sd:.4f}, "
          f"{(6100.5 - t_mu) / t_sd:.4f}")
    for name, v in solve().items():
        print(f"{name:13s} {v:,.4f}")
    a, m, first = simulate_jobs()
    print(f"sim, gamma jobs, 10^6 reps: A<900 {a:.4f}  "
          f"M<900 {m:.4f}  A first {first:.4f}")
