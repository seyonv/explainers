"""Exponential distribution: tail, memorylessness, min of exponentials.

Worked example: the source guide's pharmacy counter, where processing
times are exponential with mean 10 minutes. Python 3.10, stdlib only.
"""
import math
import random


def tail(lam, x):
    """P(X > x) for X ~ Exp(rate lam): no arrival in [0, x]."""
    return math.exp(-lam * x)


def cond_tail(lam, s, t):
    """P(X > s + t | X > s) as a ratio of tails."""
    return tail(lam, s + t) / tail(lam, s)


def simulate(lam, n=10**6, seed=0):
    rng = random.Random(seed)
    return [rng.expovariate(lam) for _ in range(n)]  # rate, not mean


def binom_tail(n, k, p):
    """P(Bin(n, p) >= k), exact."""
    return sum(math.comb(n, j) * p**j * (1 - p)**(n - j)
               for j in range(k, n + 1))


if __name__ == "__main__":
    lam = 1 / 10                       # mean 10 min -> rate 0.1/min
    print(f"P(X > 10) = {tail(lam, 10):.4f}")
    for s in (0, 5, 10, 30):
        c = cond_tail(lam, s, 10)
        print(f"P(X > {s} + 10 | X > {s}) = {c:.4f}")
    print(f"median = 10 ln 2 = {10 * math.log(2):.4f}")

    xs = simulate(lam)
    over = sum(x > 10 for x in xs) / len(xs)
    mean = sum(xs) / len(xs)
    print(f"sim: P(X > 10) = {over:.4f}, mean {mean:.3f}")
    s5 = [x for x in xs if x > 5]
    c = sum(x > 15 for x in s5) / len(s5)
    print(f"sim: P(X > 15 | X > 5) = {c:.4f}")

    # two counters, each mean 10: the first to finish is Exp(0.2)
    rng = random.Random(1)
    mins = [min(rng.expovariate(lam), rng.expovariate(lam))
            for _ in range(10**6)]
    m_over = sum(m > 10 for m in mins) / len(mins)
    print(f"min: exact mean 5, P(min > 10) = {tail(2 * lam, 10):.4f}")
    print(f"min: sim mean {sum(mins) / len(mins):.3f}, "
          f"P(min > 10) = {m_over:.4f}")

    # the source's follow-up: at least 50 of 100 wait over 10 min
    p50 = binom_tail(100, 50, tail(lam, 10))
    print(f"P(at least 50 of 100 wait > 10) = {p50:.4f}")
