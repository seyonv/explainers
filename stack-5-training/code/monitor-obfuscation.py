"""Penalize the checker, teach the hiding: a toy Goodhart model.

Three report styles for the Tuesday run. Every number in PAYOFF
and P_FIX is illustrative; the training dynamics are computed.
The fix attempt is the same in all three, so only the report
changes. Reward = hidden tests + a judge's score for the report
- LAM if the false-claim checker catches a checkable false claim.
"""
import numpy as np

P_FIX = 0.3  # chance the fix really works (illustrative)
# judge score for the final report: (if fixed, if not fixed)
JUDGE = {
    "honest": (0.6, 0.1),   # "fixed" / "still 1 failing"
    "false":  (0.6, 0.6),   # "all tests pass" either way
    "vague":  (0.4, 0.4),   # "improved parsing, looks good"
}
CHECKABLE_FALSE = {"honest": False, "false": True, "vague": False}
STYLES = list(JUDGE)


def expected_reward(style, lam):
    fixed, unfixed = JUDGE[style]
    r_fixed = 1.0 + fixed
    caught = lam if CHECKABLE_FALSE[style] else 0.0
    r_unfixed = 0.0 + unfixed - caught
    return P_FIX * r_fixed + (1 - P_FIX) * r_unfixed


def train(lam, steps=300, lr=1.0):
    """Exact softmax policy gradient on expected reward."""
    r = np.array([expected_reward(s, lam) for s in STYLES])
    theta = np.zeros(len(STYLES))
    for _ in range(steps):
        pi = np.exp(theta) / np.exp(theta).sum()
        theta += lr * pi * (r - pi @ r)
    pi = np.exp(theta) / np.exp(theta).sum()
    return r, pi


def report(lam):
    r, pi = train(lam)
    miss = 1 - P_FIX
    flagged = pi[1] * miss
    misleading = (pi[1] + pi[2]) * miss
    print(f"penalty lam = {lam}")
    for s, ri, pii in zip(STYLES, r, pi):
        print(f"  {s:7s} E[reward] {ri:5.2f}   share {pii:6.1%}")
    print(f"  checker flags   {flagged:6.1%} of runs")
    print(f"  misleading      {misleading:6.1%} of runs")
    print(f"  tests pass      {P_FIX:6.1%} of runs")


if __name__ == "__main__":
    report(0.0)
    report(1.0)
