"""The Kelly criterion: bet size that maximizes long-run growth.

Running example: a 60/40 even-money bet (p = 0.6, b = 1).
Run: python3 kelly-criterion.py   (Python 3.10, stdlib only)
"""
import math
import random
import statistics as st


def growth(f, p, b=1.0):
    """Expected log growth per bet when you stake a fraction f.

    Win (prob p): wealth * (1 + b f).  Lose: wealth * (1 - f).
    """
    return p * math.log1p(b * f) + (1 - p) * math.log1p(-f)


def kelly(p, b=1.0):
    """f* = p - q / b: set d growth / d f = 0 and solve."""
    return p - (1 - p) / b


def zero_growth(p, b=1.0):
    """The f above f* where growth falls back to 0 (bisection)."""
    lo, hi = kelly(p, b), 1 - 1e-12
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if growth(mid, p, b) > 0 else (lo, mid)
    return lo


def simulate(fs, p, n_bets, n_paths, seed, b=1.0):
    """Same coin flips for every f, so only the bet size differs."""
    rng = random.Random(seed)
    finals = {f: [] for f in fs}
    halved = {f: 0 for f in fs}
    for _ in range(n_paths):
        wins = [rng.random() < p for _ in range(n_bets)]
        for f in fs:
            w, low = 1.0, 1.0
            for win in wins:
                w *= 1 + b * f if win else 1 - f
                low = min(low, w)
            finals[f].append(w)
            halved[f] += low <= 0.5
    return finals, halved


def p_below_start(f, p, n, b=1.0):
    """Exact P(final wealth < 1): too few wins, from the binomial."""
    k0 = n * -math.log1p(-f) / (math.log1p(b * f) - math.log1p(-f))
    return sum(math.comb(n, k) * p**k * (1 - p)**(n - k)
               for k in range(n + 1) if k < k0)


if __name__ == "__main__":
    p = 0.6
    fs_star = kelly(p)
    print(f"f* = {p} - {1 - p:.1f} / 1 = {fs_star:.1f}")
    g_star = growth(fs_star, p)
    print(f"g(f*) = 0.6 ln 1.2 + 0.4 ln 0.8"
          f" = {0.6 * math.log(1.2):.5f} {0.4 * math.log(0.8):+.5f}"
          f" = {g_star:.5f}")
    print(f"growth factor per bet e^g = {math.exp(g_star):.5f}")

    print("\n  f    g(f)      % of max   e^(1000 g)")
    for f in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        g = growth(f, p)
        print(f"{f:4.2f}  {g:+.5f}  {g / g_star:8.1%}"
              f"   {math.exp(1000 * g):.4g}")
    print(f"growth hits 0 again at f = {zero_growth(p):.4f}")

    print(f"\n2:1 odds, p = 0.5: f* = {kelly(0.5, 2):.2f},"
          f" g = {growth(kelly(0.5, 2), 0.5, 2):.5f}")
    g2 = growth(kelly(0.5, 2), 0.5, 2)
    print(f"  half Kelly 0.125: g = {growth(0.125, 0.5, 2):.5f}"
          f" = {growth(0.125, 0.5, 2) / g2:.1%} of max")
    print(f"all-in 1000 times: E[W] = 1.2^1000 = 10^"
          f"{1000 * math.log10(1.2):.1f}, P(no loss) = 0.6^1000"
          f" = 10^{1000 * math.log10(0.6):.1f}")
    print("fixed $0.20 stake, 600 wins: W = 1 + 0.2 * 200 ="
          f" {1 + 0.2 * 200:.0f}")
    print("true p = 0.55 but you believe 0.6:")
    for f in (0.2, 0.1):
        print(f"  bet {f}: g = {growth(f, 0.55):+.5f}")

    fs, n, paths = (0.1, 0.2, 0.4), 1000, 2001
    finals, halved = simulate(fs, p, n, paths, seed=1)
    print(f"\n{paths} paths x {n} bets, seed 1:")
    print("  f    median wealth   exact median   P(end<1)"
          "  P(ever<=0.5)")
    for f in fs:
        med = st.median(finals[f])
        exact = math.exp(n * growth(f, p))
        print(f"{f:4.1f}  {med:13.4g}   {exact:12.4g}"
              f"   {p_below_start(f, p, n):8.2g}"
              f"  {halved[f] / paths:12.3f}")
    print("Thorp's continuous estimate of P(ever <= 0.5),"
          " 0.5^(2/c - 1):")
    for f in fs:
        c = f / fs_star
        print(f"  c = {c:.1f}: {0.5 ** (2 / c - 1):.3f}")
