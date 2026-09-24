"""Quorums and quantifying 'eventual'.

1. P(stale read) when a write has reached k of n replicas and a read
   asks r of them at random (hypergeometric): C(n-k, r) / C(n, r).
2. The quorum overlap rule: w + r > n forces a shared replica.
3. A WARS Monte Carlo (Bailis et al., PBS): how often a read t ms
   after the write commits sees it. Latencies are illustrative.

Run: python3 quorums-staleness.py   (Python 3.10, stdlib only)
"""
import random
from fractions import Fraction
from math import comb


def p_stale(n, k, r):
    """Read r of n replicas at random; k have the new value."""
    return Fraction(comb(n - k, r), comb(n, r))


def min_overlap(n, w, r):
    """Replicas every w-set and r-set must share (pigeonhole)."""
    return max(0, w + r - n)


def wars_trial(n, w, r, ts, rng):
    """One write then one read (WARS). True per t if read is fresh."""
    W = [rng.expovariate(1 / 20) for _ in range(n)]  # write lands
    A = [rng.expovariate(1 / 5) for _ in range(n)]   # ack returns
    R = [rng.expovariate(1 / 5) for _ in range(n)]   # read request
    S = [rng.expovariate(1 / 5) for _ in range(n)]   # read response
    commit = sorted(W[i] + A[i] for i in range(n))[w - 1]
    first = sorted(range(n), key=lambda i: R[i] + S[i])[:r]
    return [any(W[i] <= commit + t + R[i] for i in first)
            for t in ts]


def wars(n, w, r, ts, trials=100_000, seed=1):
    rng = random.Random(seed)
    fresh = [0] * len(ts)
    for _ in range(trials):
        for j, ok in enumerate(wars_trial(n, w, r, ts, rng)):
            fresh[j] += ok
    return [f / trials for f in fresh]


if __name__ == "__main__":
    print(p_stale(3, 1, 1), p_stale(5, 3, 3), p_stale(5, 2, 3))
    print("1. P(stale) = C(n-k, r) / C(n, r), n = 3")
    for r in (1, 2, 3):
        row = [p_stale(3, k, r) for k in (1, 2, 3)]
        cells = "  ".join(f"{str(p):>4}" for p in row)
        print(f"  r={r}  k=1,2,3:", cells)
    print("  n=3 w=1 r=1, just acked (k=1):", p_stale(3, 1, 1),
          f"= {float(p_stale(3, 1, 1)):.3f}")
    ps = p_stale(3, 1, 1)
    for k in (1, 2, 3, 5):
        print(f"  newest-of-last-{k} versions: 1 - (2/3)^{k} "
              f"= {float(1 - ps ** k):.3f}")

    print("\n2. overlap = w + r - n")
    for n, w, r in [(3, 1, 1), (3, 2, 2), (5, 3, 3), (5, 2, 3)]:
        print(f"  n={n} w={w} r={r}: w+r={w + r} "
              f"{'>' if w + r > n else '<='} {n}, "
              f"overlap >= {min_overlap(n, w, r)}, "
              f"P(stale at k=w) = {p_stale(n, w, r)}")

    ts = [0, 5, 10, 20, 50]
    print("\n3. WARS, n=3, P(read sees write) t ms after commit")
    print("   W~Exp(mean 20 ms), A,R,S~Exp(mean 5 ms), 100k trials")
    print("   t (ms):     " + "".join(f"{t:>7}" for t in ts))
    for w, r in [(1, 1), (2, 1), (2, 2)]:
        ps = wars(3, w, r, ts)
        print(f"   w={w} r={r}:   " + "".join(f"{p:7.3f}" for p in ps))
