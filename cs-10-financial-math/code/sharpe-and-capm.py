"""Sharpe ratio, beta by least squares, and the CAPM line.

Run: python3 sharpe-and-capm.py   (Python 3.10, numpy)
"""
import math

import numpy as np


def sharpe(mean, sd, rf=0.0, periods=252):
    """Annualized Sharpe ratio from per-period mean and sd."""
    return (mean - rf) / sd * math.sqrt(periods)


def beta_alpha(stock, market):
    """Least-squares slope and intercept of stock on market."""
    dm = market - market.mean()
    ds = stock - stock.mean()
    beta = (dm * ds).sum() / (dm * dm).sum()   # cov / var
    alpha = stock.mean() - beta * market.mean()
    return beta, alpha


def capm(beta, rf, market_return):
    """Expected return the CAPM line assigns to a given beta."""
    return rf + beta * (market_return - rf)


def sharpe_se(sr, n):
    """Lo (2002) iid standard error of a per-period Sharpe."""
    return math.sqrt((1 + sr * sr / 2) / n)


def hand_trace():
    m = np.array([1.0, -1.0, 2.0, -2.0, 0.0])    # market, %
    s = np.array([1.5, -1.0, 2.5, -2.5, 0.5])    # stock, %
    dm, ds = m - m.mean(), s - s.mean()
    print("day   m     s   m-mbar s-sbar  product (m-mbar)^2")
    for d, row in enumerate(zip(m, s, dm, ds), 1):
        a, b, x, y = row
        print(f"{d:>3} {a:5.1f} {b:5.1f} {x:6.1f} {y:6.1f}"
              f" {x * y:8.2f} {x * x:8.2f}")
    b, a = beta_alpha(s, m)
    print(f"sum products {(dm * ds).sum():.2f}, "
          f"sum squares {(dm * dm).sum():.2f}")
    print(f"beta = {b:.2f}, alpha = {a:.2f}% per day")


def synthetic(n=1000, seed=0):
    """Market plus two stocks with known betas 1.3 and 0.5."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0005, 0.01, n)
    a = 1.3 * mkt + rng.normal(0, 0.012, n)
    b = 0.5 * mkt + rng.normal(0, 0.006, n)
    return mkt, a, b


if __name__ == "__main__":
    sr = sharpe(0.0005, 0.01)
    print(f"annual mean {0.0005 * 252:.3f}, "
          f"annual sd {0.01 * math.sqrt(252):.4f}")
    print(f"Sharpe (rf = 0): {sr:.4f}")
    print(f"Sharpe (rf = 5%): "
          f"{sharpe(0.0005, 0.01, 0.05 / 252):.4f}")
    print(f"Sharpe's market example 6% / 15%: {0.06 / 0.15:.2f}")
    print()
    hand_trace()
    print()
    mkt, a, b = synthetic()
    print(f"n = {len(mkt)} days, seed 0")
    rf, em = 0.05, 0.0005 * 252
    for name, r, true_b in (("A", a, 1.3), ("B", b, 0.5)):
        be, al = beta_alpha(r, mkt)
        resid = r - (al + be * mkt)
        r2 = 1 - resid.var() / r.var()
        se_b = resid.std(ddof=2) / (mkt.std() * len(r) ** .5)
        se_a = resid.std(ddof=2) / len(r) ** .5 * 252
        print(f"{name}: beta {be:.3f} +/- {se_b:.3f} (true {true_b})"
              f"  R^2 {r2:.2f}")
        print(f"   alpha {al * 252:+.3f} +/- {se_a:.3f} per year"
              f" (true 0)")
        print(f"   CAPM: {rf} + {true_b} x ({em:.3f} - {rf})"
              f" = {capm(true_b, rf, em):.4f}")
    print(f"sample Sharpe of mkt: "
          f"{sharpe(mkt.mean(), mkt.std(ddof=1)):.2f}")
    se = sharpe_se(0.05, len(mkt)) * math.sqrt(252)
    print(f"true market Sharpe 0.79 +/- {se:.2f} (1 se, "
          f"{len(mkt)} days, Lo 2002)")
