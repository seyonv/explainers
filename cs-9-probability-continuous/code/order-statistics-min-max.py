"""Order statistics: the min and max of n iid exponentials.

Source problem: X1..X5 iid Exp(lam); find P(min <= a) and
P(max <= a). We take lam = 0.1 per minute (mean 10 min, as in the
pharmacy problem) to put numbers on it. Python 3.10, stdlib only.
"""
import math
import random


def cdf(lam, a):
    """F(a) = P(X <= a) for one X ~ Exp(rate lam)."""
    return 1 - math.exp(-lam * a)


def p_min_le(lam, a, n=5):
    """min <= a unless all n exceed a: 1 - (1 - F)^n."""
    return 1 - (1 - cdf(lam, a)) ** n      # = 1 - exp(-n*lam*a)


def p_max_le(lam, a, n=5):
    """max <= a only if all n are <= a: F^n."""
    return cdf(lam, a) ** n


def e_order_stat(lam, k, n=5):
    """E[X_(k)]: gaps are Exp(n*lam), Exp((n-1)*lam), ..."""
    return sum(1 / (j * lam) for j in range(n - k + 1, n + 1))


def simulate(lam, n=5, trials=200_000, seed=0):
    rng = random.Random(seed)
    lo, hi = [], []
    for _ in range(trials):
        xs = [rng.expovariate(lam) for _ in range(n)]
        lo.append(min(xs))
        hi.append(max(xs))
    return lo, hi


if __name__ == "__main__":
    lam, n = 0.1, 5
    print(" a     F(a)  P(min<=a)  P(max<=a)")
    for a in (1, 2, 5, 10, 20, 30, 50):
        print(f"{a:>2}  {cdf(lam, a):.4f}     {p_min_le(lam, a):.4f}"
              f"     {p_max_le(lam, a):.4f}")
    for k in range(1, n + 1):
        print(f"E[X_({k})] = {e_order_stat(lam, k):.4f}")
    print(f"H_5 = {sum(1 / j for j in range(1, 6)):.4f}")

    lo, hi = simulate(lam)
    t = len(lo)
    print(f"sim: E[min] {sum(lo) / t:.3f}  E[max] {sum(hi) / t:.3f}")
    print(f"sim: P(min<=10) {sum(x <= 10 for x in lo) / t:.4f}"
          f"  P(max<=10) {sum(x <= 10 for x in hi) / t:.4f}")
    med = -math.log(1 - 0.5 ** (1 / n)) / lam
    print(f"median of max = {med:.4f}")
    for m in (1, 5, 10, 100):
        h = sum(1 / j for j in range(1, m + 1))
        print(f"n={m:>3}: E[max] = {10 * h:.2f} min")
