"""Monte Carlo option pricing: a European call, antithetic
variates, a control variate, and an arithmetic Asian call.

Shared contract: S0 = 100, K = 100, r = 5%, sigma = 20%, T = 1.
Run: python3 monte-carlo-pricing.py   (Python 3.10, numpy)
"""
import math

import numpy as np

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0
DISC = math.exp(-R * T)


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_call(s, k, r, sigma, t):
    d1 = (math.log(s / k) + (r + sigma**2 / 2) * t)
    d1 /= sigma * math.sqrt(t)
    d2 = d1 - sigma * math.sqrt(t)
    return s * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)


def geometric_asian_exact(m):
    """Closed form: the geometric mean of m GBM fixings is
    lognormal, so a Black-Scholes-style formula prices it."""
    mu = math.log(S0) + (R - SIGMA**2 / 2) * T * (m + 1) / (2 * m)
    var = SIGMA**2 * T * (m + 1) * (2 * m + 1) / (6 * m * m)
    d2 = (mu - math.log(K)) / math.sqrt(var)
    d1 = d2 + math.sqrt(var)
    fwd = math.exp(mu + var / 2)
    return DISC * (fwd * norm_cdf(d1) - K * norm_cdf(d2))


def paths(z):
    """Risk-neutral GBM at m equal steps; z has shape (n, m).
    Each step is exact, so m = 1 jumps straight to T."""
    dt = T / z.shape[1]
    inc = (R - SIGMA**2 / 2) * dt + SIGMA * math.sqrt(dt) * z
    return S0 * np.exp(np.cumsum(inc, axis=1))


def european(s):
    return DISC * np.maximum(s[:, -1] - K, 0)


def asian(s):
    return DISC * np.maximum(s.mean(axis=1) - K, 0)


def asian_geo(s):
    geo = np.exp(np.log(s).mean(axis=1))
    return DISC * np.maximum(geo - K, 0)


def estimate(x):
    """Sample mean and its standard error s / sqrt(n)."""
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))


def mc_call(n, seed=0):
    z = np.random.default_rng(seed).standard_normal((n, 1))
    return estimate(european(paths(z)))


def antithetic(payoff, n, m, seed=0):
    """n payoffs from n/2 pairs (z, -z): average each pair."""
    z = np.random.default_rng(seed).standard_normal((n // 2, m))
    return estimate((payoff(paths(z)) + payoff(paths(-z))) / 2)


def control_variate(y, c, c_mean):
    """y - b (c - E[c]), with b = cov(y, c) / var(c)."""
    b = np.cov(y, c)[0, 1] / c.var(ddof=1)
    return estimate(y - b * (c - c_mean)), b


if __name__ == "__main__":
    bs = black_scholes_call(S0, K, R, SIGMA, T)
    print(f"Black-Scholes call      {bs:.4f}")

    z = np.random.default_rng(0).standard_normal((5, 1))
    print("first 5 draws (seed 0): z, S_T, discounted payoff")
    for zi, st, pay in zip(z[:, 0], paths(z)[:, 0],
                           european(paths(z))):
        print(f"  {zi:+.4f}  {st:8.3f}  {pay:7.3f}")

    print("n          estimate   std err   error    err/se")
    for n in (10**4, 10**5, 10**6):
        m, se = mc_call(n)
        print(f"{n:<9,}  {m:8.4f}  {se:8.4f}  "
              f"{m - bs:+.4f}  {(m - bs) / se:+.2f}")

    n = 10**6
    m, se = mc_call(n)
    print(f"\nEuropean call, {n:,} payoffs each")
    print(f"  plain           {m:.4f} +- {se:.4f}")
    ma, sea = antithetic(european, n, 1)
    print(f"  antithetic      {ma:.4f} +- {sea:.4f}"
          f"   variance / {(se / sea)**2:.2f}")
    z = np.random.default_rng(0).standard_normal((n, 1))
    s = paths(z)
    (mc, sec), b = control_variate(european(s), DISC * s[:, -1], S0)
    print(f"  control S_T     {mc:.4f} +- {sec:.4f}"
          f"   variance / {(se / sec)**2:.2f}   b = {b:.3f}")

    n, m = 10**5, 12
    ex = geometric_asian_exact(m)
    print(f"\nAsian call, {m} monthly fixings, {n:,} payoffs each")
    z = np.random.default_rng(0).standard_normal((n, m))
    s = paths(z)
    a, sa = estimate(asian(s))
    g, sg = estimate(asian_geo(s))
    print(f"  plain           {a:.4f} +- {sa:.4f}")
    aa, saa = antithetic(asian, n, m)
    print(f"  antithetic      {aa:.4f} +- {saa:.4f}"
          f"   variance / {(sa / saa)**2:.2f}")
    (ac, sac), b = control_variate(asian(s), asian_geo(s), ex)
    print(f"  control geo     {ac:.4f} +- {sac:.4f}"
          f"   variance / {(sa / sac)**2:,.0f}   b = {b:.3f}")
    print(f"  geometric MC {g:.4f} +- {sg:.4f} vs exact {ex:.4f}")
