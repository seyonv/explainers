"""Hypergeometric likelihood and capture-recapture.

cheat-sheet > Univariate Random Variables > Hypergeometric: tag 20
animals, later catch 30 and find 7 tagged. Which population size N
makes that catch most likely? Exact likelihoods with Fractions, the
ratio test that finds the peak, the Lincoln-Petersen and Chapman
estimates, and a simulation check of P(7 tagged | N = 85).
"""
import random
from fractions import Fraction as F
from math import comb

K, n, k = 20, 30, 7            # tagged, recaptured, tagged in catch


def hyper_pmf(N, K, n, k):
    """P(k tagged in a catch of n | N animals, K tagged)."""
    return F(comb(K, k) * comb(N - K, n - k), comb(N, n))


def ratio(N):
    """L(N) / L(N-1) = (N-K)(N-n) / (N (N-K-n+k))."""
    return F((N - K) * (N - n), N * (N - K - n + k))


def mle(K, n, k):
    """Largest N with ratio(N) >= 1, i.e. floor(K n / k)."""
    return K * n // k


def simulate(N, trials=100_000, seed=0):
    rng = random.Random(seed)
    pond = [1] * K + [0] * (N - K)          # 1 = tagged
    hits = sum(sum(rng.sample(pond, n)) == k for _ in range(trials))
    return hits / trials


if __name__ == "__main__":
    lo = K + n - k                          # 43: fewer is impossible
    for N in (lo, 60, 84, 85, 86, 100, 150):
        L = hyper_pmf(N, K, n, k)
        r = f"{float(ratio(N)):.5f}" if N > lo else "  -  "
        print(f"N={N:>3}  L={float(L):.6f}  ratio={r}")
    best = max(range(lo, 1000), key=lambda N: hyper_pmf(N, K, n, k))
    print("scan argmax:", best, " floor(K*n/k):", mle(K, n, k))
    print("Lincoln-Petersen K*n/k =", F(K * n, k),
          f"= {K * n / k:.2f}")
    print("Chapman (K+1)(n+1)/(k+1) - 1 =",
          F((K + 1) * (n + 1), k + 1) - 1)
    L85 = hyper_pmf(85, K, n, k)
    print(f"L(85) exact {float(L85):.4f}  sim {simulate(85):.4f}")
    print(f"E[k | N=85] = n*K/N = {n * K / 85:.3f}")
