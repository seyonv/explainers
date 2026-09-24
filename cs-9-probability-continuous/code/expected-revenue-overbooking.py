"""Expected revenue: the overbooked bus.

Solves the tour-bus problem in cheat-sheet > Univariate Random
Variables > Applications exactly, then asks how many tickets the
operator should sell, and checks the answer by simulation.

Bus: 20 seats, fare 50 (kept even if the tourist does not show),
each tourist shows with p = 0.98 independently, and every tourist
who shows but finds no seat costs 100 (refund 50 + penalty 50).
"""
import random
from math import comb


def expected_revenue(n, seats=20, fare=50, bump=100, p=0.98):
    """E[revenue] when n tickets are sold. S ~ Binomial(n, p)."""
    over = sum((k - seats) * comb(n, k) * p**k * (1 - p)**(n - k)
               for k in range(seats + 1, n + 1))  # E[(S - seats)+]
    return fare * n - bump * over


def p_full(n, seats=20, p=0.98):
    """P(S >= seats): the next ticket would be bumped if it shows."""
    return sum(comb(n, k) * p**k * (1 - p)**(n - k)
               for k in range(seats, n + 1))


def simulate(n, trips, seats=20, fare=50, bump=100, p=0.98, seed=0):
    rng = random.Random(seed)
    total = 0
    for _ in range(trips):
        shows = sum(rng.random() < p for _ in range(n))
        total += fare * n - bump * max(shows - seats, 0)
    return total / trips


def best_n(p, lo=20, hi=40):
    return max(range(lo, hi + 1),
               key=lambda n: expected_revenue(n, p=p))


if __name__ == "__main__":
    # The source's question: 21 tickets.
    print(f"exact, 21 tickets: 1050 - 100 * 0.98**21"
          f" = {1050 - 100 * 0.98**21:.2f}")
    print(f"source's formula : 21*0.98*50 - 0.98**21*50"
          f" = {21 * 0.98 * 50 - 0.98**21 * 50:.2f}")
    print(f"plug in E[S]     : 1050 - 100 * (20.58 - 20)"
          f" = {1050 - 100 * (21 * 0.98 - 20):.2f}")
    print(f"simulated, 10^5 trips: {simulate(21, 10**5):.2f}")

    # How many tickets should he sell?
    print("\n n  E[bumped]  E[revenue]  next ticket adds")
    for n in range(20, 26):
        er = expected_revenue(n)
        over = (50 * n - er) / 100
        gain = 50 - 100 * 0.98 * p_full(n)
        print(f"{n:2d}  {over:9.4f}  {er:10.2f}  {gain:+8.2f}")
    print("best n for p = 0.98:", best_n(0.98))

    # When does overbooking start to pay?
    q_star = 1 - 0.5 ** (1 / 21)
    print(f"\nbreak-even no-show rate: 1 - 0.5**(1/21) = {q_star:.4f}")
    for q in (0.02, 0.04, 0.05, 0.10, 0.20):
        n = best_n(1 - q)
        print(f"no-show {q:.2f}: sell {n}, "
              f"E[revenue] = {expected_revenue(n, p=1 - q):.2f}")
