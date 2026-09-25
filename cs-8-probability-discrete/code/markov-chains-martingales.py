"""Markov chains and martingales: absorption by linear equations.

1. The Markov thief (source guide > Univariate Random Variables >
   Applications): a two-state chain with hours attached to each step;
   the fundamental matrix gives E[time to escape] = 18 h.
2. Gambler's ruin on 0..10 from 3: solve the linear equations exactly
   with Fractions, check with the martingale (optional stopping)
   closed forms, then with a simulation of 10**5 walks per game.
"""
import random
from fractions import Fraction as F


def solve(A, b):
    """Gauss-Jordan elimination on exact Fractions: A x = b."""
    n = len(A)
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[-1] for row in M]


def ruin(N, p, start):
    """Walk on 0..N, up w.p. p. Returns P(hit N), E[steps]."""
    q = 1 - p
    ids = range(1, N)                     # transient states 1..N-1
    A = [[F(int(i == j)) - (p if j == i + 1 else 0)
          - (q if j == i - 1 else 0) for j in ids] for i in ids]
    hit = solve(A, [p if i == N - 1 else F(0) for i in ids])
    steps = solve(A, [F(1)] * (N - 1))    # (I - Q) t = 1
    return hit[start - 1], steps[start - 1]


def ruin_formula(N, p, i):
    """Closed forms from optional stopping on a martingale."""
    if p == F(1, 2):                      # S_n and S_n^2 - n
        return F(i, N), F(i * (N - i))
    r = (1 - p) / p                       # r**S_n is a martingale
    h = (1 - r**i) / (1 - r**N)
    return h, (i - N * h) / (1 - 2 * p)   # S_n - (2p - 1) n


def simulate(N, p, start, walks=10**5, seed=0):
    rng = random.Random(seed)
    wins = steps = 0
    for _ in range(walks):
        s = start
        while 0 < s < N:
            s += 1 if rng.random() < p else -1
            steps += 1
        wins += s == N
    return wins / walks, steps / walks


def thief():
    """States: dungeon (transient), free (absorbing)."""
    Q = F(2, 3)                           # doors 3 h and 9 h: back in
    hours = F(6 + 3 + 9, 3)               # mean hours per door choice
    visits = 1 / (1 - Q)                  # fundamental matrix (1x1)
    return visits, hours, visits * hours


if __name__ == "__main__":
    v, h, t = thief()
    print(f"thief: {v} visits x {h} h = {t} h")
    for p in (F(1, 2), F(49, 100)):
        hit, steps = ruin(10, p, 3)
        fh, fs = ruin_formula(10, p, 3)
        assert (hit, steps) == (fh, fs)
        sh, ss = simulate(10, float(p), 3)
        print(f"p={float(p):.2f} P(10 before 0)={float(hit):.4f}"
              f" E[steps]={float(steps):.2f}"
              f"  sim {sh:.4f} {ss:.2f}")
    h, s = ruin_formula(100, F(49, 100), 30)
    print(f"p=0.49, 30 -> 100: P(win)={float(h):.4f}"
          f" E[steps]={float(s):.0f}")
