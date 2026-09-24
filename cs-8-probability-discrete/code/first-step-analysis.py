"""First-step analysis: condition on the first move, solve for v.

Solves the craps problem from cheat-sheet > General Probability and
the Markov-thief problem from Univariate Random Variables >
Applications with exact Fractions, then checks both by simulation.
"""
import random
from fractions import Fraction as F
from itertools import permutations, product

# P(sum of two dice), exact
P = {s: F(0) for s in range(2, 13)}
for a, b in product(range(1, 7), repeat=2):
    P[a + b] += F(1, 36)

POINTS = (4, 5, 6, 8, 9, 10)


def point_win(pt):
    # w = P[pt]*1 + P[7]*0 + (1 - P[pt] - P[7]) * w
    return P[pt] / (P[pt] + P[7])


def craps():
    win = P[7] + P[11]                 # decided on the first roll
    for pt in POINTS:
        win += P[pt] * point_win(pt)   # first roll sets the point
    return win


def craps_rolls():
    # E = 1 + sum P[pt] * m_pt,  m = 1 + (1 - P[pt] - P[7]) * m
    return 1 + sum(P[pt] / (P[pt] + P[7]) for pt in POINTS)


def thief():
    # E = 1/3*6 + 1/3*(3 + E) + 1/3*(9 + E)   ->   E = a + b*E
    a, b = F(6 + 3 + 9, 3), F(2, 3)
    return a / (1 - b)


def thief_with_memory():
    # a thief who never retries a bad door: average over door orders
    times = []
    for order in permutations([6, 3, 9]):
        t = 0
        for d in order:
            t += d
            if d == 6:
                break
        times.append(t)
    return F(sum(times), len(times))


def sim_craps(games=10**6, seed=0):
    rng = random.Random(seed)
    roll = lambda: rng.randint(1, 6) + rng.randint(1, 6)
    wins = 0
    for _ in range(games):
        s = roll()
        if s in (7, 11):
            wins += 1
        elif s in POINTS:
            while (r := roll()) not in (s, 7):
                pass
            wins += r == s
    return wins / games


def sim_thief(runs=10**5, seed=0):
    rng = random.Random(seed)
    total = 0
    for _ in range(runs):
        while (d := rng.choice((6, 3, 9))) != 6:
            total += d
        total += 6
    return total / runs


if __name__ == "__main__":
    for pt in POINTS:
        print(f"point {pt:>2}: P={P[pt]}  win from point "
              f"{point_win(pt)}  contributes {P[pt] * point_win(pt)}")
    w = craps()
    print("craps:", w, f"{float(w):.4f}")
    r = craps_rolls()
    print("craps rolls per game:", r, f"{float(r):.3f}")
    print("thief:", thief(), "hours")
    print("thief with memory:", thief_with_memory(), "hours")
    print(f"sim craps, 10**6 games: {sim_craps():.4f}")
    print(f"sim thief, 10**5 runs: {sim_thief():.3f} hours")
