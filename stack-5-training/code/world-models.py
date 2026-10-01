"""Mask or model the observations? A toy GRPO group, two objectives.

Token weights follow the hybrid loss in Wolfe's "Agentic World
Models": action tokens get the GRPO advantage, observation tokens
get a small constant (ECHO's lambda). Token counts are illustrative.
"""
import numpy as np

LAM = 0.05  # top of ECHO's reported best range [0.01, 0.05]


def grpo_adv(rewards):
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + 1e-8)


def weights(rewards, n_act, n_obs, lam):
    """Per-token loss weights for one group of rollouts."""
    rows = []
    for a in grpo_adv(rewards):
        rows.append(np.r_[np.full(n_act, a), np.full(n_obs, lam)])
    return np.array(rows)


def live(w):
    return int(np.count_nonzero(np.abs(w) > 1e-6))


def report(name, rewards, n_act=120, n_obs=480):
    masked = weights(rewards, n_act, n_obs, 0.0)
    echo = weights(rewards, n_act, n_obs, LAM)
    total = masked.size
    print(f"{name:<14} adv={np.round(grpo_adv(rewards), 2)}")
    print(f"  masked: {live(masked):>4} / {total} tokens carry"
          " a gradient")
    print(f"  ECHO:   {live(echo):>4} / {total} tokens carry"
          " a gradient")


def surprise(probs):
    """Cross-entropy, in nats, of the tokens that actually came."""
    return float(-np.log(np.asarray(probs)).sum())


if __name__ == "__main__":
    report("all fail", [0, 0, 0, 0])
    report("one passes", [1, 0, 0, 0])
    print()
    # Tuesday run, step 6: pytest prints "1 failed".
    # p = probability the model gave each real token (illustrative)
    for who, p in [("expects pass", [0.05, 0.10]),
                   ("expects fail", [0.60, 0.70])]:
        print(f"{who:<13} loss on '1 failed' = "
              f"{surprise(p):.2f} nats")
    print()
    print("gradient size vs token probability p")
    for p in [0.01, 0.1, 0.2, 0.5]:
        ce = 1 / p  # |d(-log p)/dp|
        mae = 1.0   # |d(1 - p)/dp|
        keep = "kept" if p <= 0.2 else "dropped (p > 0.2)"
        print(f"  p={p:<4}  CE {ce:>5.0f}   MAE {mae:.0f}"
              f"   clipped MAE: {keep}")
