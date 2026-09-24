"""Random walks: gambler's ruin with a small edge, the sqrt(n) rule.

Run: python3 random-walks.py  (Python 3.10; last table needs numpy)
"""
import math
import random
from fractions import Fraction as F


def ruin(i, N, p):
    """Walk from i, +1 w.p. p else -1, stop at 0 or N.
    Returns (P(reach N first), E[steps]) from optional stopping."""
    q = 1 - p
    if p == q:                      # S_n and S_n^2 - n are martingales
        return F(i, N), i * (N - i)
    r = q / p                       # r**S_n is a martingale
    h = (1 - r**i) / (1 - r**N)
    return h, (i - N * h) / (q - p)  # S_n - (p - q) n is one too


def simulate_ruin(i, N, p, walks, seed=0):
    """Monte Carlo check: share of walks that hit N, mean steps."""
    rng = random.Random(seed)
    wins = steps = 0
    for _ in range(walks):
        x = i
        while 0 < x < N:
            x += 1 if rng.random() < p else -1
            steps += 1
        wins += x == N
    return wins / walks, steps / walks


def drift_and_spread(n, p):
    """E[S_n] = n(p - q) grows like n; sd = 2 sqrt(pqn) like sqrt n."""
    q = 1 - p
    return n * (p - q), 2 * math.sqrt(p * q * n)


def mean_abs_fair(n):
    """Exact E|S_n| for a fair walk, n even: n C(n, n/2) / 2^n,
    in logs so n = 10^6 doesn't overflow a float."""
    lg = math.lgamma
    return math.exp(math.log(n) + lg(n + 1) - 2 * lg(n / 2 + 1)
                    - n * math.log(2))


def simulate_endpoints(n, p, walks, seed=0):
    """S_n = 2 * Binomial(n, p) - n, sampled with numpy."""
    import numpy as np
    s = 2 * np.random.default_rng(seed).binomial(n, p, walks) - n
    return s.mean(), s.std(), np.abs(s).mean()


if __name__ == "__main__":
    print("ruin: P(reach N before 0) and E[steps]")
    for i, N in ((10, 20), (3, 10), (30, 100), (100, 200)):
        for p in (F(1, 2), F(49, 100)):
            h, t = ruin(i, N, p)
            print(f"  {i:>3} -> {N:<3} p={float(p):.2f}"
                  f"  P={float(h):.4f}  E[steps]={float(t):,.1f}")
    r = F(51, 49)
    print("  10 -> 20: r^10 =", f"{float(r**10):.4f}",
          " h = 1/(1 + r^10) =", f"{float(1 / (1 + r**10)):.4f}")

    print("monte carlo, 10 -> 20, 10^5 walks, seed 0")
    for p in (0.5, 0.49):
        w, t = simulate_ruin(10, 20, p, 100_000)
        se = math.sqrt(w * (1 - w) / 100_000)
        print(f"  p={p:.2f}  P={w:.4f} +/- {se:.4f}  steps={t:.1f}")

    print("drift vs spread, p = 0.49; E|S| fair vs sqrt(2n/pi)")
    for n in (100, 10_000, 1_000_000):
        d, sd = drift_and_spread(n, 0.49)
        print(f"  n={n:>9,} drift={d:>7,.0f} sd={sd:>5,.1f}"
              f" E|S| fair={mean_abs_fair(n):>6,.2f}"
              f" vs {math.sqrt(2 * n / math.pi):>6,.2f}")
    print("  drift = sd when n = 4pq / (p-q)^2 =",
          f"{4 * 0.49 * 0.51 / 0.02**2:,.0f}")

    print("simulated S_n, 10^5 walks, seed 0 (numpy)")
    for n, p in ((10_000, 0.5), (10_000, 0.49)):
        m, s, a = simulate_endpoints(n, p, 100_000)
        print(f"  n={n:,} p={p:.2f} mean={m:7.2f} sd={s:6.2f}"
              f" E|S|={a:6.2f}")
