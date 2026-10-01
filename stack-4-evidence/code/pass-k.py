"""pass@k vs pass^k: the formulas and the unbiased estimators.

pass@k (Chen et al., 2021): chance at least one of k trials passes.
pass^k (Yao et al., 2024, tau-bench): chance all k trials pass.

From n trials of one task with c passes, both are estimated by
counting k-subsets of the n trials, then averaged over tasks.
The Tuesday-run trial counts below are illustrative.
"""
from math import comb


def pass_at_k(n, c, k):
    """Unbiased pass@k: 1 - C(n-c, k) / C(n, k)."""
    return 1 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n, c, k):
    """Unbiased pass^k: C(c, k) / C(n, k)."""
    return comb(c, k) / comb(n, k)


def expected_plug_in(p, n, k):
    """Mean of (c/n)**k over c ~ Binomial(n, p): the naive way."""
    return sum(comb(n, c) * p**c * (1 - p)**(n - c) * (c / n)**k
               for c in range(n + 1))


if __name__ == "__main__":
    print("p     k  pass@k   pass^k")
    for p in (0.5, 0.7, 0.9, 0.95):
        for k in (1, 3, 8):
            print(f"{p:<5} {k}  {1 - (1 - p)**k:.4f}   {p**k:.4f}")
    print(f"0.95**10 = {0.95**10:.4f}")

    n, c = 8, 5  # Tuesday task run 8 times, 5 passed
    print(f"\nTuesday task, n={n}, c={c}")
    for k in (1, 3, 8):
        print(f"k={k}: pass@k {pass_at_k(n, c, k):.3f}"
              f"  pass^k {pass_hat_k(n, c, k):.3f}"
              f"  plug-in (c/n)^k {(c / n)**k:.3f}")

    p, k = 0.7, 3
    print(f"\ntrue p={p}: pass^{k} = {p**k:.4f}")
    print(f"mean of plug-in (c/{n})^{k}: "
          f"{expected_plug_in(p, n, k):.4f}  (biased up)")
    mean_est = sum(comb(n, c) * p**c * (1 - p)**(n - c)
                   * pass_hat_k(n, c, k) for c in range(n + 1))
    print(f"mean of C(c,{k})/C({n},{k}): {mean_est:.4f}  (exact)")

    print("\nsame 70% average, two agents, 4 tasks, k=8")
    for name, rates in (("even", [0.7] * 4),
                        ("split", [1.0, 1.0, 0.4, 0.4])):
        hat = sum(r**8 for r in rates) / len(rates)
        at = sum(1 - (1 - r)**8 for r in rates) / len(rates)
        print(f"{name:<6} pass^8 {hat:.4f}  pass@8 {at:.4f}")
