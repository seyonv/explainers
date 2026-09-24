"""Conditional independence: the 3-judge panel (cheat-sheet, exam P).

Each judge votes guilty with prob 0.7 if the defendant is guilty and
0.2 if innocent, independently GIVEN the truth. 70% are guilty.
"""
import random
from fractions import Fraction as F

PRIOR = {"G": F(7, 10), "I": F(3, 10)}   # guilty / innocent
VOTE = {"G": F(7, 10), "I": F(2, 10)}    # P(one guilty vote | truth)


def p_votes(votes, truth):
    """P(vote pattern | truth): a product, by cond. independence."""
    q = VOTE[truth]
    out = F(1)
    for v in votes:
        out *= q if v else 1 - q
    return out


def p(votes):
    """Unconditional P(votes): mix the two truths by their priors."""
    return sum(PRIOR[t] * p_votes(votes, t) for t in PRIOR)


def p_judge3(votes12):
    """P(E3 | judges 1 and 2 voted votes12)."""
    return p(votes12 + (1,)) / p(votes12)


def simulate(trials=10**6, seed=0):
    rng = random.Random(seed)
    seen = {k: [0, 0] for k in ((1, 1), (1, 0), (0, 0))}
    for _ in range(trials):
        q = 0.7 if rng.random() < 0.7 else 0.2
        v1, v2, v3 = (rng.random() < q for _ in range(3))
        key = (1, 1) if v1 and v2 else (0, 0) if not (v1 or v2) \
            else (1, 0)
        seen[key][0] += 1
        seen[key][1] += v3
    return {k: b / a for k, (a, b) in seen.items()}


if __name__ == "__main__":
    e1, both = p((1,)), p((1, 1))
    print(f"P(E1) = {float(e1):.2f}   P(E1 E2) = {float(both):.4f}")
    print(f"P(E1)P(E2) = {float(e1 * e1):.4f}  -> not independent")
    g = p_votes((1, 1), "G")
    print(f"P(E1 E2 | G) = {float(g):.2f} = 0.7 * 0.7"
          "  -> independent given G")
    print(f"P(E2 | E1) = {float(both / e1):.4f}")
    cases = {"a both guilty": ((1, 1), 1),
             "b one of each": ((1, 0), 2),
             "c both not": ((0, 0), 1)}
    sim = simulate()
    for name, (v, ways) in cases.items():
        post_g = PRIOR["G"] * p_votes(v, "G") / p(v)
        ans = p_judge3(v)
        print(f"{name:13} P={float(ways * p(v)):.3f}"
              f" G|.={float(post_g):.4f}"
              f" E3|.={ans}={float(ans):.4f}"
              f" sim {sim[v]:.4f}")
