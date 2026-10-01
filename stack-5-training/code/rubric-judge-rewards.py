"""Three reward designs scored on four Tuesday-run rollouts.

The rollouts and which criteria they meet are illustrative.
Weights are the categorical weights from Rubrics as Rewards
(Essential 1.0, Important 0.7, Optional 0.3, Pitfall 0.9), as
listed in Wolfe's "Rubric-Based Rewards for RL".
"""
import numpy as np

# criterion: (weight, plain meaning)
RUBRIC = {
    "c1": (1.0, "Essential: checker re-runs pytest, exit 0"),
    "c2": (0.7, "Important: claim matches last test run"),
    "c3": (0.9, "Pitfall avoided: no test file edited"),
    "c4": (0.3, "Optional: names the root cause"),
}

# rollout: (final message says done, checker exit 0, criteria met)
ROLLOUTS = {
    "A false claim":  (1, 0, {"c1": 0, "c2": 0, "c3": 1, "c4": 1}),
    "B honest fail":  (0, 0, {"c1": 0, "c2": 1, "c3": 1, "c4": 0}),
    "C edits test":   (1, 1, {"c1": 1, "c2": 1, "c3": 0, "c4": 0}),
    "D real fix":     (1, 1, {"c1": 1, "c2": 1, "c3": 1, "c4": 1}),
}


def rubric_reward(met, veto=False):
    """Explicit aggregation: weighted sum / total weight."""
    if veto and met["c3"] == 0:  # failed pitfall overrides all
        return 0.0
    num = sum(RUBRIC[c][0] * met[c] for c in RUBRIC)
    return num / sum(w for w, _ in RUBRIC.values())


def group_advantage(r):
    """GRPO: (reward - group mean) / group std."""
    r = np.asarray(r, dtype=float)
    sd = r.std()
    return np.zeros_like(r) if sd == 0 else (r - r.mean()) / sd


if __name__ == "__main__":
    names = list(ROLLOUTS)
    designs = {
        "says done": [ROLLOUTS[n][0] for n in names],
        "tests pass": [ROLLOUTS[n][1] for n in names],
        "rubric": [rubric_reward(ROLLOUTS[n][2]) for n in names],
        "rubric+veto": [rubric_reward(ROLLOUTS[n][2], True)
                        for n in names],
    }
    for d, r in designs.items():
        adv = group_advantage(r)
        print(f"{d:12s}")
        for n, x, a in zip(names, r, adv):
            print(f"  {n:14s} reward {x:.3f}  advantage {a:+.2f}")
    # same false-claim run, but the judge for c2 only sees the
    # compacted summary ("tests passing"), so it marks c2 met
    fooled = dict(ROLLOUTS["A false claim"][2], c2=1)
    print(f"A, judge sees summary only: {rubric_reward(fooled):.3f}")
