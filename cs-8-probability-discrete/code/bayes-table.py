"""Bayes' theorem as a table: prior x likelihood, then normalise.

Solves the three Bayes problems in the source guide > General Probability >
Basic Concepts of Probability (smokers, car model year, vaccine), then
checks the vaccine answer with a simulation of 10**6 shipments.
"""
from fractions import Fraction as F
from math import comb

import numpy as np


def bayes_table(rows):
    """rows: [(hypothesis, prior, likelihood)] -> table, P(E)."""
    joint = [p * like for _, p, like in rows]
    total = sum(joint)                       # P(E), the column total
    table = [(h, p, like, j, j / total)
             for (h, p, like), j in zip(rows, joint)]
    return table, total


def binom_pmf(n, k, p):
    return comb(n, k) * p**k * (1 - p)**(n - k)


def show(title, rows):
    table, total = bayes_table(rows)
    print(title)
    for h, p, like, j, post in table:
        print(f"  {h:<6} {float(p):.2f} x {float(like):.4f}"
              f" = {float(j):.5f} -> {float(post):.4f}")
    print(f"  P(E) = {float(total):.5f}")
    return table


def simulate_vaccine(trials=10**6, seed=0):
    rng = np.random.default_rng(seed)
    from_a = rng.random(trials) < 0.4
    bad = rng.binomial(25, np.where(from_a, 0.03, 0.02))
    hit = bad == 2                           # keep the evidence
    return from_a[hit].mean(), hit.sum()


if __name__ == "__main__":
    # Only ratios matter: light = 1, heavy = 2x light, non = light/2.
    smokers = show("smokers", [("heavy", F(2, 10), F(2)),
                               ("light", F(3, 10), F(1)),
                               ("non", F(5, 10), F(1, 2))])
    cars = show("car", [("1997", F(16, 100), F(5, 100)),
                        ("1998", F(18, 100), F(2, 100)),
                        ("1999", F(20, 100), F(3, 100)),
                        ("other", F(46, 100), F(4, 100))])
    vac = show("vaccine", [("A", F(2, 5), binom_pmf(25, 2, F(3, 100))),
                           ("B", F(3, 5), binom_pmf(25, 2, F(2, 100)))])
    print("exact:", smokers[0][4], cars[0][4],
          f"{float(vac[0][4]):.4f}")
    share, n = simulate_vaccine()
    print(f"sim: P(A | 2 bad) = {share:.4f} from {n:,} shipments")
