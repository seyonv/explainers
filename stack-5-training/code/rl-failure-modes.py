"""Two ways agentic RL loses its learning signal, in numbers.

1. Dead groups. GRPO scores each rollout against its own group
   (advantage = reward - group mean, over group std). If all G
   rollouts of a task get the same 0/1 reward, every advantage is
   0 and the task teaches nothing. A task solved with chance p
   gives a dead group with chance p**G + (1-p)**G. G = 8 rollouts
   per prompt is Olmo 3's setting (Wolfe, GRPO++). As the policy
   gets sure of itself (collapse), p heads to 0 or 1 and the batch
   quietly empties. A sampled check confirms the closed form.

2. Entropy vs template collapse (RAGEN-2's split). Total output
   diversity H(Z) = H(Z|X) + I(X;Z): within-input entropy plus
   cross-input mutual information. Entropy alone cannot tell a
   healthy policy from one that writes the same templates for
   every input. The three toy policies are illustrative.

3. Why clipping starves rare tokens (DAPO's example, eps = 0.2,
   and DAPO's eps_high = 0.28).
Python 3.10 + numpy.
"""
import numpy as np

G = 8           # rollouts per prompt (Olmo 3: 64 prompts x 8)
SEED = 5


def dead_share(p, g=G):
    """Chance that all g rollouts get the same 0/1 reward."""
    return p ** g + (1 - p) ** g


def sampled_dead_share(p, rng, tasks=200_000, g=G):
    r = rng.random((tasks, g)) < p
    same = r.all(1) | ~r.any(1)
    return same.mean()


def bits(dist):
    d = np.asarray(dist, float)
    d = d[d > 0]
    return float(-(d * np.log2(d)).sum())


def diversity(policy):
    """policy: rows = inputs (equally likely), cols = templates."""
    pz_x = np.asarray(policy, float)
    h_cond = np.mean([bits(row) for row in pz_x])   # H(Z|X)
    h_total = bits(pz_x.mean(0))                    # H(Z)
    return h_cond, h_total - h_cond, h_total        # MI = H - H|X


POLICIES = {   # 2 inputs x 4 reasoning templates, illustrative
    "healthy": [[.45, .45, .05, .05], [.05, .05, .45, .45]],
    "entropy collapse": [[.97, .01, .01, .01],
                         [.01, .01, .01, .97]],
    "template collapse": [[.25, .25, .25, .25],
                          [.25, .25, .25, .25]],
}


def main():
    rng = np.random.default_rng(SEED)
    print(f"1. Dead groups with G = {G} rollouts per task")
    print("   p(solve)  closed form  sampled")
    for p in (0.5, 0.8, 0.9, 0.95, 0.99):
        print(f"   {p:<8}  {dead_share(p):>10.1%}"
              f"  {sampled_dead_share(p, rng):>7.1%}")

    print("\n2. Diversity in bits: H(Z|X) + I(X;Z) = H(Z)")
    for name, pol in POLICIES.items():
        h, mi, tot = diversity(pol)
        print(f"   {name:<18} entropy {h:.2f}  MI {mi:.2f}"
              f"  total {tot:.2f}")

    print("\n3. Max new probability after one clipped update")
    for old in (0.01, 0.9):
        print(f"   old {old:<5} eps 0.20 -> {old * 1.2:.4g}"
              f"   eps_high 0.28 -> {old * 1.28:.4g}")


if __name__ == "__main__":
    main()
