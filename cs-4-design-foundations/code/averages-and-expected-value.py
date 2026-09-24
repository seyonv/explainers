"""Averages, expected value and chance (cs-4-design-foundations).

Mean vs median vs mode on a skewed neighbourhood, expected tests for
pooled screening, and the exact expected value of chuck-a-luck.
Runs under python3 (3.10), standard library only.
"""
import random
from fractions import Fraction as F
from itertools import product
from statistics import mean, median, mode

# Illustrative incomes ($K): 12 farmers + 3 weekenders, chosen so the
# three averages match the source's $65K / $25K / $16K.
INCOMES = [16, 16, 16, 16, 18, 20, 22, 25, 26, 28, 30, 32,
           200, 230, 280]


def pooled_tests(n, p):
    """Expected tests: 1 pooled test, plus n more if it is positive."""
    if n == 1:
        return 1                       # a pool of one is just a test
    clean = (1 - p) ** n               # P(pool negative)
    return 1 * clean + (1 + n) * (1 - clean)


def pooled_per_person(k, p):
    """Expected tests per person with pools of k people."""
    return pooled_tests(k, p) / k


def chuck_a_luck():
    """Bet $1 on a number: win $m if m of 3 dice show it, else -$1."""
    total = F(0)
    for dice in product(range(1, 7), repeat=3):   # all 216 rolls
        m = dice.count(1)                          # we bet on 1
        total += m if m else -1
    return total / 216


def simulate_chuck(plays, seed=0):
    rng = random.Random(seed)
    won = 0
    for _ in range(plays):
        m = sum(rng.randint(1, 6) == 1 for _ in range(3))
        won += m if m else -1
    return won / plays


def secretary(n, trials, seed=0):
    """Skip the first n/e, then take the first one better than all."""
    rng = random.Random(seed)
    cut = round(n / 2.718281828459045)
    hits = 0
    for _ in range(trials):
        xs = rng.sample(range(n), n)
        best_seen = max(xs[:cut])
        pick = next((x for x in xs[cut:] if x > best_seen), xs[-1])
        hits += pick == n - 1
    return cut, hits / trials


if __name__ == "__main__":
    print(f"mean {mean(INCOMES)}  median {median(INCOMES)}  "
          f"mode {mode(INCOMES)}  ($K, illustrative)")
    print(f"without the 3 weekenders: mean "
          f"{mean(INCOMES[:12]):.1f}  median {median(INCOMES[:12])}")

    p = 0.01
    print(f"P(pool of 50 negative) = 0.99**50 = {0.99 ** 50:.4f}")
    print(f"E[tests], pool of 50 = {pooled_tests(50, p):.2f}")
    for k in (1, 5, 10, 25, 50):
        print(f"  pools of {k:>2}: {50 * pooled_per_person(k, p):6.2f}"
              f" tests for 50 people")
    best = min(range(1, 51), key=lambda k: pooled_per_person(k, p))
    print(f"best pool size {best}: "
          f"{pooled_per_person(best, p):.4f} tests per person")

    ev = chuck_a_luck()
    print(f"chuck-a-luck EV = {ev} = {float(ev):.4f} dollars per $1")
    print(f"source's formula 1/2 - 125/216 = {F(1, 2) - F(125, 216)}")
    print(f"P(win anything) = {1 - F(125, 216)}")
    sd = (float(F(269, 216) - ev ** 2) / 10**6) ** 0.5
    print(f"simulated, 10**6 plays: {simulate_chuck(10**6):.4f}"
          f" (standard error {sd:.4f})")

    cut, rate = secretary(100, 20_000)
    exact = cut / 100 * sum(1 / (i - 1) for i in range(cut + 1, 101))
    print(f"secretary n=100: skip {cut}, best chosen {rate:.3f}"
          f" of the time (exact {exact:.4f}, 1/e = 0.3679)")
    print(f"20 tries at alpha 0.05: P(>=1 false alarm) = "
          f"{1 - 0.95 ** 20:.3f}")
