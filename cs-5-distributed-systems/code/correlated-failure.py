"""Why independent-failure math lies: P^N vs a common-cause floor.

p = chance one replica fails on its own in some window
c = chance of one shared event (bad config, same bug) that takes
    every replica down at once. Numbers are illustrative.
"""
import math
import numpy as np


def p_all_down(p, n, c=0.0):
    """Chance all n replicas are down: shared event, or n own ones."""
    return c + (1 - c) * p**n


def nines(prob):
    return -math.log10(prob)


def simulate(p, n, c, trials=10**6, seed=0):
    rng = np.random.default_rng(seed)
    shared = rng.random(trials) < c              # one event, all die
    own = (rng.random((trials, n)) < p).all(axis=1)
    return int((shared | own).sum())


def poison_trace(n):
    """One bad request; each crash makes the client retry elsewhere."""
    up = [True] * n
    for r in range(n):
        up[r] = False                            # parser crashes
        state = " ".join("up" if u else "XX" for u in up)
        print(f"  try replica {r + 1}: {state}  ({sum(up)} up)")


if __name__ == "__main__":
    p, c = 0.01, 0.01
    print(" N   independent p^N    with c = 1%   nines")
    for n in range(1, 6):
        a, b = p_all_down(p, n), p_all_down(p, n, c)
        print(f" {n}   {a:.0e}  {b:.8f}"
              f"   {nines(a):4.1f} vs {nines(b):.3f}")
    b3 = p_all_down(p, 3, c)
    print(f"shared event's share of risk at N=3: {c / b3:.4%}")
    R = 1e-9
    ok = [n for n in range(1, 21) if p_all_down(p, n) < R]
    okc = [n for n in range(1, 21) if p_all_down(p, n, c) < R]
    print(f"smallest N with P < {R:g}: independent {ok[0]},"
          f" correlated {okc[0] if okc else 'none up to 20'}")
    print("simulated 10^6 windows, N=3:")
    print("  independent:", simulate(p, 3, 0.0), "all-down")
    print("  with c=1%:  ", simulate(p, 3, c), "all-down")
    print("poison request, 3 replicas:")
    poison_trace(3)
