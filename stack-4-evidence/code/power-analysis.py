"""How many tasks do you need to see a change in false-claim rate?

Paired design: every task runs under both conditions (say compaction
dose 0 and dose 1), m repeats per task per condition. Repeats of one
task are correlated (intra-class correlation rho), so we count tasks.
Standard library plus numpy. Python 3.10.
"""
import math
from statistics import NormalDist

import numpy as np

Z = NormalDist().inv_cdf


def tasks_needed(p0, delta, m, rho, alpha=0.05, power=0.8):
    """Tasks for a two-sided test of p1 - p0 = delta."""
    p1 = p0 + delta
    z = Z(1 - alpha / 2) + Z(power)
    deff = 1 + (m - 1) * rho          # design effect
    runs = z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / delta**2
    return runs * deff / m            # runs per arm -> tasks


def mde(n, p0, m, rho, alpha=0.05, power=0.8):
    """Smallest delta that n tasks can detect (bisection)."""
    lo, hi = 1e-4, 1 - p0 - 1e-4
    for _ in range(60):
        mid = (lo + hi) / 2
        if tasks_needed(p0, mid, m, rho, alpha, power) > n:
            lo = mid
        else:
            hi = mid
    return hi


def simulate(n, p0, delta, m, rho, reps=20000, seed=0):
    """Power of a paired z-test on per-task differences.

    Each task gets a true rate x ~ Beta with mean p and ICC rho
    (a + b = 1/rho - 1), then m Bernoulli runs at that rate.
    The two conditions are drawn independently per task.
    """
    rng = np.random.default_rng(seed)
    s = 1 / rho - 1

    def arm(p):
        x = rng.beta(p * s, (1 - p) * s, size=(reps, n))
        return rng.binomial(m, x) / m

    d = arm(p0 + delta) - arm(p0)
    se = d.std(axis=1, ddof=1) / math.sqrt(n)
    zs = np.divide(d.mean(axis=1), se,
                   out=np.zeros(reps), where=se > 0)
    return float(np.mean(np.abs(zs) > Z(0.975)))


if __name__ == "__main__":
    p0, m = 0.20, 3
    print(f"baseline {p0}, m = {m} repeats, alpha 0.05, power 0.8")
    print(f"z sum = {Z(0.975):.3f} + {Z(0.8):.3f}")
    for rho in (0.2, 0.5):
        print(f"\nrho = {rho}: design effect "
              f"{1 + (m - 1) * rho:.1f}")
        for delta in (0.10, 0.15, 0.20):
            n = math.ceil(tasks_needed(p0, delta, m, rho))
            pw = simulate(n, p0, delta, m, rho)
            print(f"  delta {delta:.2f}: {n:4d} tasks "
                  f"({n * m * 2:5d} runs)  sim power {pw:.3f}")
    print("\ncapstone: 30 tasks x 3 repeats per dose")
    for rho in (0.2, 0.5):
        d = mde(30, p0, m, rho)
        print(f"  rho {rho}: smallest delta {d:.3f}"
              f"  (0.20 -> {p0 + d:.2f})")
        for delta in (0.10, 0.15, 0.20):
            pw = simulate(30, p0, delta, m, rho)
            print(f"    sim power at delta {delta:.2f}: {pw:.3f}")
    print("pooled over 2 prompts x 2 models: m = 12 per dose")
    for rho in (0.2, 0.5):
        print(f"  rho {rho}: smallest delta "
              f"{mde(30, p0, 12, rho):.3f}")
    print("\nmore repeats vs more tasks (delta 0.10, rho 0.5)")
    for mm in (1, 3, 12, 1000):
        n = tasks_needed(p0, 0.10, mm, 0.5)
        print(f"  m = {mm:4d}: {n:6.1f} tasks")
