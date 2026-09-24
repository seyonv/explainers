"""Two-asset portfolio mean and variance, and the minimum-variance mix.

Assets (expected returns are illustrative):
  A: mu 8%, sigma 20%     B: mu 12%, sigma 30%     rho = 0.2
Run: python3 portfolio-mean-variance.py   (Python 3.10; numpy for the
n-asset check at the end)
"""
import math


def portfolio(w, mu1, mu2, s1, s2, rho):
    """Mean and sigma with weight w in asset 1 and 1 - w in asset 2."""
    mean = w * mu1 + (1 - w) * mu2
    var = (w * s1) ** 2 + ((1 - w) * s2) ** 2 \
        + 2 * w * (1 - w) * rho * s1 * s2
    return mean, math.sqrt(var)


def min_var_weight(s1, s2, rho):
    """Weight in asset 1 that minimises variance (set d var/dw = 0)."""
    cov = rho * s1 * s2
    return (s2 ** 2 - cov) / (s1 ** 2 + s2 ** 2 - 2 * cov)


def min_var_n(cov):
    """n assets: w = inv(Cov) @ 1, rescaled so the weights sum to 1."""
    import numpy as np
    x = np.linalg.solve(cov, np.ones(len(cov)))
    return x / x.sum()


def frontier(mu1, mu2, s1, s2, rho, weights):
    """(w, mean, sigma) for each weight w in asset 1."""
    return [(w, *portfolio(w, mu1, mu2, s1, s2, rho)) for w in weights]


if __name__ == "__main__":
    mu1, mu2, s1, s2, rho = 0.08, 0.12, 0.20, 0.30, 0.2
    w = min_var_weight(s1, s2, rho)
    m, s = portfolio(w, mu1, mu2, s1, s2, rho)
    print(f"min-variance: w_A = {w:.4f}, w_B = {1 - w:.4f}")
    print(f"  mean = {m:.2%}, sigma = {s:.2%}")
    print("  w_A     mean    sigma   naive avg sigma")
    ws = (0, 0.25, 0.5, w, 1)
    for wa, mean, sig in frontier(mu1, mu2, s1, s2, rho, ws):
        naive = wa * s1 + (1 - wa) * s2
        print(f"  {wa:.4f}  {mean:.2%}  {sig:.2%}  {naive:.2%}")
    for r in (1.0, 0.0, -1.0):
        wr = min_var_weight(s1, s2, r)
        if 0 <= wr <= 1:
            _, sr = portfolio(wr, mu1, mu2, s1, s2, r)
            print(f"rho = {r:+.1f}: min at w_A = {wr:.3f}, "
                  f"sigma = {sr:.2%}")
        else:
            print(f"rho = {r:+.1f}: w_A = {wr:.3f} means shorting B;"
                  f" long-only min is all A, {s1:.2%}")
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    print("numpy n-asset check:", min_var_n(cov).round(4))
