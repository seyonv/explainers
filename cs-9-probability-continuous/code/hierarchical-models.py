"""Hierarchical (two-level) models: draw a parent, then a child.

1. Signal: S ~ N(mu, sigma^2), R | S = s ~ N(s, 1).
   Tower rule and total variance give E[R], Var(R), Cov(R, S).
   mu = 10, sigma = 2 are illustrative (the source keeps symbols).
2. Integers: X uniform on {0..4}, Y | X = x uniform on {0..x}.
   P(X + Y > 4) exactly, plus E and Var of Y by the same rules.
Both are checked by simulation (seed 0).
"""
import math
import random
from fractions import Fraction as F

MU, SIGMA = 10.0, 2.0


def signal_moments(mu, sigma):
    """Tower rule + law of total variance for R = S + noise."""
    e_r = mu                       # E[E[R|S]] = E[S]
    var_r = 1 + sigma ** 2         # E[Var(R|S)] + Var(E[R|S])
    cov = sigma ** 2               # E[S * E[R|S]] - mu^2
    return e_r, var_r, cov, cov / math.sqrt(sigma ** 2 * var_r)


def shrink(r, mu, sigma):
    """E[S | R = r]: pull the reading back toward mu."""
    w = sigma ** 2 / (sigma ** 2 + 1)
    return mu + w * (r - mu)


def two_dice():
    """X ~ U{0..4}; Y | x ~ U{0..x}. Joint pmf, exact."""
    return {(x, y): F(1, 5) * F(1, x + 1)
            for x in range(5) for y in range(x + 1)}


def ey_vary(joint):
    """E[Y] and Var(Y) two ways: directly and by conditioning."""
    ey = sum(p * y for (_, y), p in joint.items())
    ey2 = sum(p * y * y for (_, y), p in joint.items())
    within = sum(F(1, 5) * F(x * (x + 2), 12) for x in range(5))
    between = sum(F(1, 5) * (F(x, 2) - 1) ** 2 for x in range(5))
    return ey, ey2 - ey ** 2, within, between


if __name__ == "__main__":
    e_r, var_r, cov, rho = signal_moments(MU, SIGMA)
    print(f"E[R] = {e_r}  Var(R) = {var_r}  Cov(R,S) = {cov}"
          f"  corr = {rho:.4f}")
    print(f"E[S | R = 13] = {shrink(13, MU, SIGMA):.2f}")

    rng = random.Random(0)
    n = 10 ** 6
    ss = [rng.gauss(MU, SIGMA) for _ in range(n)]
    rs = [s + rng.gauss(0, 1) for s in ss]
    mr, ms = sum(rs) / n, sum(ss) / n
    vr = sum((r - mr) ** 2 for r in rs) / n
    cv = sum((r - mr) * (s - ms) for r, s in zip(rs, ss)) / n
    k = sum((r - mr) ** 4 for r in rs) / n / vr ** 2
    print(f"sim: E[R] {mr:.3f}  Var {vr:.3f}  Cov {cv:.3f}"
          f"  kurtosis {k:.3f}")

    # random variance instead of random mean: not normal
    v = [1, 9]
    kurt = 3 * (sum(x * x for x in v) / 2) / (sum(v) / 2) ** 2
    print(f"N(0, V), V = 1 or 9: kurtosis 3*41/25 = {kurt:.2f}")

    joint = two_dice()
    hit = {k: p for k, p in joint.items() if sum(k) > 4}
    for (x, y), p in sorted(hit.items()):
        print(f"  x={x} y={y}  1/5 * 1/{x + 1} = {p}")
    ans = sum(hit.values())
    print(f"P(X + Y > 4) = {ans} = {float(ans)}")
    ey, vy, within, between = ey_vary(joint)
    print(f"E[Y] = {ey}  Var(Y) = {vy} = {within} + {between}")

    rng = random.Random(0)
    m, hits, ys = 10 ** 6, 0, 0
    for _ in range(m):
        x = rng.randint(0, 4)
        y = rng.randint(0, x)
        hits += x + y > 4
        ys += y
    print(f"sim: P = {hits / m:.4f}  E[Y] = {ys / m:.4f}")
