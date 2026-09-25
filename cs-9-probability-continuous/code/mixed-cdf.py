"""Mixed distributions: split a CDF into jumps plus a density.

Solves the two mixed-CDF problems in the source guide > Univariate Random
Variables > Continuous Univariate Distributions exactly (Fractions),
then checks the moments by simulating with the generalised inverse.
"""
import random
from fractions import Fraction as Fr


def peval(c, x):
    """Polynomial with coefficients c (c[k] * x**k) at x."""
    return sum(ck * x**k for k, ck in enumerate(c))


def split(pieces):
    """pieces: [(a, b, c)] with F(x) = peval(c, x) on [a, b),
    contiguous; F = 0 before the first a, 1 from the last b.
    Returns atoms {x: jump} and densities [(a, b, F' coeffs)]."""
    atoms, dens, left = {}, [], Fr(0)    # left = F(x-)
    for a, b, c in pieces:
        if peval(c, a) != left:          # F jumps at a
            atoms[a] = peval(c, a) - left
        dens.append((a, b, [k * ck for k, ck in enumerate(c)][1:]))
        left = peval(c, b)
    if left != 1:                        # final jump up to 1
        atoms[pieces[-1][1]] = 1 - left
    return atoms, dens


def moment(atoms, dens, n):
    """E[X**n] = sum x**n p over atoms + integral x**n f(x) dx."""
    total = sum(x**n * p for x, p in atoms.items())
    for a, b, f in dens:
        for k, fk in enumerate(f):       # integral of fk x**(k+n)
            m = k + n + 1
            total += fk * (Fr(b)**m - Fr(a)**m) / m
    return total


def cdf(pieces, x):
    if x < pieces[0][0]:
        return 0.0
    for a, b, c in pieces:
        if a <= x < b:
            return float(peval(c, x))
    return 1.0


def sample(pieces, u):
    """Generalised inverse: smallest x with F(x) >= u."""
    lo, hi = float(pieces[0][0]), float(pieces[-1][1])
    if cdf(pieces, lo) >= u:
        return lo
    for _ in range(30):                  # F(lo) < u <= F(hi)
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if cdf(pieces, mid) >= u else (mid, hi)
    return hi


h = Fr(1, 2)
Y = [(0, h, [Fr(1, 10), 0, 1]),          # y**2 + 0.1 on [0, 0.5)
     (h, 1, [0, 1])]                     # y on [0.5, 1)
X = [(1, 2, [1, -1, h])]                 # (x**2 - 2x + 2) / 2


def report(name, pieces, n=100_000, seed=0):
    atoms, dens = split(pieces)
    c1 = sum(atoms.values())
    m1, m2 = moment(atoms, dens, 1), moment(atoms, dens, 2)
    var = m2 - m1**2
    jumps = ", ".join(f"{x}: {p}" for x, p in atoms.items())
    print(f"{name}: jumps {{{jumps}}}  c1 = {c1}, c2 = {1 - c1}")
    for a, b, f in dens:
        print(f"  density on [{a}, {b}): coeffs {[str(v) for v in f]}")
    print(f"  E = {m1} = {float(m1):.4f}   E[{name}^2] = {m2}"
          f"   Var = {var} = {float(var):.5f}")
    fp = [(float(a), float(b), [float(v) for v in c])
          for a, b, c in pieces]         # floats: fast sampling
    rng = random.Random(seed)
    xs = [sample(fp, rng.random()) for _ in range(n)]
    mean = sum(xs) / n
    v = sum((x - mean) ** 2 for x in xs) / (n - 1)
    print(f"  sim n={n:,}: mean {mean:.4f}, var {v:.4f}")
    return atoms, dens, m1, var


if __name__ == "__main__":
    report("Y", Y)
    atoms, dens, m1, var = report("X", X)
    ay, _ = split(Y)                     # F = c1 F1 + c2 F2
    c1 = sum(ay.values())
    print("Y: F1 masses", {str(x): str(p / c1) for x, p in ay.items()},
          " F2(0.5) =", (Fr(35, 100) - Fr(1, 10)) / (1 - c1))
    # tail formula for X >= 0: E[X] = integral of 1 - F(x) dx
    tail = 1 + moment({}, [(1, 2, [0, 1, -h])], 0)
    print("X: tail-formula E =", tail)
    # the common slip: differentiate F, forget the jump
    print("X: density only -> mass", moment({}, dens, 0),
          " E-part", moment({}, dens, 1))
