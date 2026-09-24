"""Choosing without conflicts, and bridge hands.

Solves three counting problems from cheat-sheet > General Probability:
  1. 3 cells of a 6-by-5 grid, no two in a row or column (Discrete
     Mathematics), counted two ways and by brute force;
  2. a 13-card hand holds the ace and king of at least one suit;
  3. a 13-card hand holds all 4 cards of at least one denomination
     (both Basic Concepts of Probability), by inclusion-exclusion
     with exact fractions, checked by simulating 10**5 deals.
"""
import random
from fractions import Fraction as F
from itertools import combinations
from math import comb, perm


def rook_ways(rows, cols, k):
    """Pick k cells, no two sharing a row or a column."""
    # choose the k rows, then give them distinct columns in order
    return comb(rows, k) * perm(cols, k)


def rook_brute(rows, cols, k):
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    return sum(len({r for r, _ in s}) == k
               and len({c for _, c in s}) == k
               for s in combinations(cells, k))


def at_least_one(groups, size, hand=13, deck=52):
    """P(hand holds every card of at least one block)."""
    total = F(0)
    for j in range(1, groups + 1):         # j blocks forced in
        if j * size > hand:
            break
        rest = comb(deck - j * size, hand - j * size)
        term = F(comb(groups, j) * rest, comb(deck, hand))
        total += (-1) ** (j + 1) * term    # +, -, +, -
    return total


def ie_terms(groups, size, hand=13, deck=52):
    return [(j, comb(groups, j), comb(deck - j * size, hand - j * size))
            for j in range(1, groups + 1) if j * size <= hand]


def simulate(trials=10**5, seed=0):
    rng = random.Random(seed)
    deck = [(rank, suit) for rank in range(13) for suit in range(4)]
    ak = quad = 0
    for _ in range(trials):
        hand = set(rng.sample(deck, 13))
        # rank 0 = ace, rank 12 = king
        ak += any((0, s) in hand and (12, s) in hand for s in range(4))
        counts = [0] * 13
        for rank, _ in hand:
            counts[rank] += 1
        quad += 4 in counts
    return ak / trials, quad / trials


if __name__ == "__main__":
    print("grid: C(6,3) * 5*4*3 =", comb(6, 3), "*", perm(5, 3), "=",
          rook_ways(6, 5, 3))
    print("      C(6,3)*C(5,3)*3! =", comb(6, 3) * comb(5, 3) * 6)
    print("      ordered 30*20*12/3! =", 30 * 20 * 12 // 6)
    print("      brute force =", rook_brute(6, 5, 3))

    n = comb(52, 13)
    print("C(52,13) =", n)
    for name, g, s in [("ace+king of a suit", 4, 2),
                       ("all 4 of a rank", 13, 4)]:
        print(name)
        for j, c, rest in ie_terms(g, s):
            sign = "+" if j % 2 else "-"
            print(f"  {sign} C({g},{j}) * {rest:,} / C(52,13)"
                  f" = {sign} {c * rest / n:.6f}")
        p = at_least_one(g, s)
        print(f"  = {float(p):.6f}")
        print(f"  exact = {p}")

    print(rook_ways(6, 5, 3))
    for g, s in [(4, 2), (13, 4)]:   # ace+king x 4 suits; quads x 13
        p = at_least_one(g, s)
        print(p, f"{float(p):.6f}")

    sim_ak, sim_quad = simulate()
    print(f"sim 10^5 deals (seed 0): {sim_ak:.4f}  {sim_quad:.4f}")
