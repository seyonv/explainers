"""Joint and conditional distributions from a table.

Source: ljeng/cheat-sheet, Multivariate Random Variables >
Multivariate Distributions (basketball shoe size S, height H).
Run: python3 joint-table-conditional.py
"""
import math
import random
from fractions import Fraction as F

HEIGHTS = [68, 70, 73]
SHOES = [F(17, 2), F(12)]                 # 8.5 and 12
JOINT = {                                 # (s, h) -> P(S=s, H=h)
    (F(17, 2), 68): F(25, 100), (F(17, 2), 70): F(20, 100),
    (F(17, 2), 73): F(15, 100), (F(12), 68): F(5, 100),
    (F(12), 70): F(12, 100),
}
JOINT[(F(12), 73)] = 1 - sum(JOINT.values())  # the "?" cell


def marginals(joint):
    ps, ph = {}, {}
    for (s, h), p in joint.items():
        ps[s] = ps.get(s, 0) + p          # sum across the row
        ph[h] = ph.get(h, 0) + p          # sum down the column
    return ps, ph


def mean(dist):
    return sum(x * p for x, p in dist.items())


def cond_s(joint, h):
    """P(S = s | H = h): one column, divided by its total."""
    col = {s: p for (s, hh), p in joint.items() if hh == h}
    tot = sum(col.values())
    return {s: p / tot for s, p in col.items()}


def cv(dist):
    m = mean(dist)
    var = sum((x - m) ** 2 * p for x, p in dist.items())
    return math.sqrt(var) / m, var


def simulate(n=10**6, seed=0):
    rng = random.Random(seed)
    cells = list(JOINT)
    draws = rng.choices(cells, [float(JOINT[c]) for c in cells], k=n)
    s73 = [float(s) for s, h in draws if h == 73]
    s68 = [float(s) for s, h in draws if h == 68]
    m68 = sum(s68) / len(s68)
    sd68 = math.sqrt(sum((x - m68) ** 2 for x in s68) / len(s68))
    return sum(s73) / len(s73), len(s73), sd68 / m68, len(s68)


if __name__ == "__main__":
    ps, ph = marginals(JOINT)
    print("missing cell:", JOINT[(F(12), 73)],
          "| table sums to", sum(JOINT.values()))
    print("P(S):", {float(s): float(p) for s, p in ps.items()})
    print("P(H):", {h: float(p) for h, p in ph.items()})
    print("E[H] =", float(mean(ph)), " E[S] =", float(mean(ps)))
    for h in HEIGHTS:
        c = cond_s(JOINT, h)
        r, v = cv(c)
        print(f"H={h}: P(S=8.5|H)={float(c[SHOES[0]]):.4f}")
        print(f"  E[S|H]={float(mean(c)):.4f}"
              f" ({mean(c)})  Var={float(v):.4f}  CV={r:.4f}")
    tower = sum(ph[h] * mean(cond_s(JOINT, h)) for h in HEIGHTS)
    print("tower: sum P(h) E[S|h] =", float(tower))
    esh = sum(s * h * p for (s, h), p in JOINT.items())
    cov = esh - mean(ps) * mean(ph)
    print("E[SH] =", float(esh), " Cov(S,H) =", float(cov))
    print("P(S=8.5)P(H=68) =", float(ps[SHOES[0]] * ph[68]),
          "vs P(8.5, 68) =", float(JOINT[(SHOES[0], 68)]))
    e73, n73, cv68, n68 = simulate()
    print(f"sim: E[S|H=73] = {e73:.4f} from {n73:,} draws;"
          f" CV|H=68 = {cv68:.4f} from {n68:,}")
