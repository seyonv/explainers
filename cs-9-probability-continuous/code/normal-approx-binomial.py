"""Normal approximation to the binomial, with a continuity correction.

Worked example: the source guide's pharmacy counter. Orders take an
exponential time with mean 10 min, so each of 100 customers waits
over 10 min with p = e^-1. P(at least 50 of them wait that long)?
Python 3.10, standard library only.
"""
import math
import random


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def binom_tail(n, k, p):
    """P(X >= k) for X ~ Bin(n, p), exact: sum the pmf."""
    return sum(math.comb(n, j) * p**j * (1 - p)**(n - j)
               for j in range(k, n + 1))


def normal_tail(n, k, p, cc=True):
    """P(X >= k) from N(np, npq); cc moves the cut to k - 0.5."""
    mu, sd = n * p, math.sqrt(n * p * (1 - p))
    x = k - 0.5 if cc else k
    return 1 - norm_cdf((x - mu) / sd)


def simulate(n, k, p, trials=10**5, seed=0):
    rng = random.Random(seed)
    hits = sum(sum(rng.random() < p for _ in range(n)) >= k
               for _ in range(trials))
    return hits / trials


if __name__ == "__main__":
    n, k, p = 100, 50, math.exp(-1)
    mu, var = n * p, n * p * (1 - p)
    sd = math.sqrt(var)
    print(f"p = {p:.6f}  mu = {mu:.4f}  var = {var:.4f}"
          f"  sd = {sd:.4f}")
    print(f"z no cc = {(k - mu) / sd:.4f}"
          f"  z cc = {(k - 0.5 - mu) / sd:.4f}")
    exact = binom_tail(n, k, p)
    for name, v in [("exact", exact),
                    ("normal, cc", normal_tail(n, k, p)),
                    ("normal, no cc", normal_tail(n, k, p, False))]:
        print(f"{name:14} {v:.6f}  x{v / exact:.3f}")
    print(f"sim (1e5 runs) {simulate(n, k, p):.5f}")
    print("k   exact     cc        no cc")
    for kk in (35, 40, 45, 50, 55):
        print(f"{kk}  {binom_tail(n, kk, p):.6f}"
              f"  {normal_tail(n, kk, p):.6f}"
              f"  {normal_tail(n, kk, p, False):.6f}")
