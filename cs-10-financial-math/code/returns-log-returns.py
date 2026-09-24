"""Simple vs log returns: compounding, the sigma^2/2 gap, and sqrt(252).

Run: python3 returns-log-returns.py   (Python 3.10, stdlib only)
"""
import math
import random
import statistics as st


def simple_returns(prices):
    """R_t = P_t / P_{t-1} - 1: what you report and what you earn."""
    return [b / a - 1 for a, b in zip(prices, prices[1:])]


def log_returns(prices):
    """r_t = ln(P_t / P_{t-1}) = ln(1 + R_t): these add up over time."""
    return [math.log(b / a) for a, b in zip(prices, prices[1:])]


def total_simple(rets):
    """Compound simple returns: multiply the growth factors."""
    return math.prod(1 + r for r in rets) - 1


def geometric_mean(rets):
    """The CAGR per period: (prod of 1 + R)^(1/n) - 1."""
    growth = math.prod(1 + r for r in rets)
    return growth ** (1 / len(rets)) - 1


def simulate_prices(n, mu, sigma, seed, p0=100.0):
    """Price path with i.i.d. normal daily log returns."""
    rng = random.Random(seed)
    prices = [p0]
    for _ in range(n):
        prices.append(prices[-1] * math.exp(rng.gauss(mu, sigma)))
    return prices


def annualize(daily_mean_log, daily_sd, days=252):
    """Means scale with time, standard deviations with its root."""
    return daily_mean_log * days, daily_sd * math.sqrt(days)


if __name__ == "__main__":
    # 1. The +50% / -50% path
    p = [100, 150, 75]
    R, r = simple_returns(p), log_returns(p)
    print("simple:", [round(x, 4) for x in R])
    print("log:   ", [round(x, 6) for x in r])
    print("sum of simple  :", round(sum(R), 4))
    print("compounded     :", round(total_simple(R), 4))
    print("sum of log     :", round(sum(r), 6),
          "= ln 0.75 =", round(math.log(0.75), 6))
    print("exp(sum log)-1 :", round(math.expm1(sum(r)), 4))
    g = geometric_mean(R)
    print("arith mean", st.mean(R), "geo mean", round(g, 4),
          "sigma^2/2", st.pvariance(R) / 2)

    # 2. The sigma^2/2 gap on a simulated path (seed 7)
    n, mu, sigma = 2520, 0.0003, 0.02
    prices = simulate_prices(n, mu, sigma, seed=7)
    R, r = simple_returns(prices), log_returns(prices)
    arith, geo = st.mean(R), geometric_mean(R)
    half_var = st.pvariance(R) / 2
    print(f"\n{n} days, seed 7: P_end = {prices[-1]:.2f}")
    print(f"arith mean   {arith * 1e4:8.3f} bp/day")
    print(f"geo mean     {geo * 1e4:8.3f} bp/day")
    print(f"gap          {(arith - geo) * 1e4:8.3f} bp/day")
    print(f"sigma^2/2    {half_var * 1e4:8.3f} bp/day")
    print(f"mean log r   {st.mean(r) * 1e4:8.3f} bp/day")
    print(f"ln(1+geo)    {math.log1p(geo) * 1e4:8.3f} bp/day")

    # 3. Annualize with 252 trading days
    m, s = annualize(st.mean(r), st.pstdev(r))
    print(f"\ndaily sd {st.pstdev(r):.5f} -> annual {s:.4f}")
    print(f"daily mean log {st.mean(r):.6f} -> annual {m:.4f}")
    print(f"annual drag 252 * sigma^2/2 = {252 * half_var:.4f}")
    print(f"1% daily sd -> {0.01 * math.sqrt(252):.4f} a year")

    # 4. Across assets, simple returns average; log returns don't
    w, a, b = 0.5, 0.5, -0.5
    port = w * a + (1 - w) * b
    wrong = math.expm1(w * math.log1p(a) + (1 - w) * math.log1p(b))
    print(f"\n50/50 portfolio of +50% and -50%: {port:.4f}")
    print(f"weighted log returns, converted: {wrong:.4f}")
