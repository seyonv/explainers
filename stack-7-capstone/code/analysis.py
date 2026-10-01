"""Holm correction and the exact sign test, stdlib only.

The demo feeds in the three paired tampering p-values from the
baseline pilot (bootstrap, two-sided; exploratory contrasts, not a
registered family) and shows what Holm does to them.
"""
import math


def holm(pvals):
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvals, key=pvals.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def sign_test(n_pos, n_neg):
    """Exact two-sided sign test on the non-zero differences."""
    n, k = n_pos + n_neg, min(n_pos, n_neg)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2 ** n)


if __name__ == "__main__":
    pilot = {"d1-codex - d0": 0.015, "d1-opencode - d0": 0.235,
             "d1-opencode - d1-codex": 0.001}
    adj = holm(pilot)
    for k in sorted(pilot, key=pilot.get):
        verdict = "reject" if adj[k] < 0.05 else "keep"
        print(f"{k:24} raw {pilot[k]:.3f}  Holm {adj[k]:.3f}"
              f"  {verdict}")
    # d1-codex - d0: 12 prefixes went up, 4 down, 24 unchanged
    print(f"sign test 12 up, 4 down: p = {sign_test(12, 4):.4f}")
