"""Geometric distribution: the source's claims problem, the
reference strip, the memoryless check and a simulation.

Source: ljeng/cheat-sheet, probability/univariate-random-variables.md
#geometric (commit 5cedb05). Python 3.10, standard library only.
"""
import math
import random
from fractions import Fraction as F


def claims_pmf(ratio, n_max):
    """p_{n+1} = ratio * p_n for n >= 0. Normalise, return p_0..p_n."""
    p0 = 1 - ratio            # p0 * sum(ratio**n) = p0 / (1 - ratio)
    return [p0 * ratio**n for n in range(n_max + 1)]


def tail(q, k):
    """P(N >= k) for N = failures before the first success."""
    return q**k               # the first k trials all failed


def strip(p):
    """Mean, variance, MGF text for both conventions."""
    q = 1 - p
    return {
        "trials X=1,2,..": (1 / p, q / p**2, "p e^t / (1 - q e^t)"),
        "failures Y=0,1,..": (q / p, q / p**2, "p / (1 - q e^t)"),
    }


def simulate(p, trials=10**6, seed=0):
    """Count failures before the first success, many times."""
    rng = random.Random(seed)
    counts = []
    for _ in range(trials):
        n = 0
        while rng.random() >= p:      # failure with prob 1 - p
            n += 1
        counts.append(n)
    return counts


def memory_ratio(pmf):
    """P(N >= 2 | N >= 1) / P(N >= 1): 1 means memoryless."""
    ge1 = 1 - pmf(0)
    ge2 = ge1 - pmf(1)
    return ge2 / ge1, ge1


if __name__ == "__main__":
    q = F(1, 5)
    p = claims_pmf(q, 4)
    print("p_n:", [str(x) for x in p])
    print("p0 =", p[0], " P(N > 1) = 1 - p0 - p1 =",
          1 - p[0] - p[1], "=", tail(q, 2))

    for name, (m, v, mgf) in strip(F(4, 5)).items():
        print(f"{name:18} mean {m}  var {v}  MGF {mgf}")

    # memoryless: P(N >= m + n | N >= m) = q**n, for any m
    for m in range(4):
        print(f"P(N>={m}+2 | N>={m}) =",
              tail(q, m + 2) / tail(q, m))

    counts = simulate(0.8)
    t = len(counts)
    ge1 = sum(c >= 1 for c in counts)
    ge2 = sum(c >= 2 for c in counts)
    print(f"sim: P(N>1) = {ge2 / t:.6f}  (exact 0.040000)")
    print(f"sim: mean = {sum(counts) / t:.4f}  (exact 0.2500)")
    print(f"sim: P(N>=2 | N>=1) = {ge2 / ge1:.4f}  (exact 0.2000)")
    se = math.sqrt(0.04 * 0.96 / t)
    print(f"one standard error of P(N>1): {se:.5f}")

    # same mean 1/4, three other claim-count models
    lam = 0.25
    pois = lambda k: math.exp(-lam) * lam**k / math.factorial(k)
    nb = lambda k: (k + 1) * (8 / 9)**2 * (1 / 9)**k   # NB(2, 8/9)
    bn = lambda k: math.comb(2, k) * (1/8)**k * (7/8)**(2 - k)
    geo = lambda k: 0.8 * 0.2**k
    for name, f in [("geometric(0.8)", geo), ("Poisson(1/4)", pois),
                    ("NB(2, 8/9)", nb), ("Bin(2, 1/8)", bn)]:
        r, ge1 = memory_ratio(f)
        print(f"{name:15} P(N>=1) {ge1:.4f}"
              f"  P(N>=2 | N>=1) {r:.4f}")
