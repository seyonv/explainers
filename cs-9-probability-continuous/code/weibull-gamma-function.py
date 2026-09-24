"""Rat survival: Weibull(k=3, lam=120) and the gamma function.

f(x) = 3x^2 / 120^3 * exp(-(x/120)^3), x > 0 (weeks).
P(X > 100) and E[X] = 120 * Gamma(4/3), checked by Simpson's
rule and by simulation. Ends with a Gamma(alpha, theta) strip:
a sum of alpha exponentials.
"""
import math
import random

K, LAM = 3, 120


def pdf(x, k=K, lam=LAM):
    return k / lam * (x / lam) ** (k - 1) * math.exp(-(x / lam) ** k)


def sf(x, k=K, lam=LAM):
    """P(X > x): substitute y = (x/lam)^k, integrand -> e^-y."""
    return math.exp(-(x / lam) ** k)


def moment(n, k=K, lam=LAM):
    """E[X^n] = lam^n * Gamma(1 + n/k), same substitution."""
    return lam ** n * math.gamma(1 + n / k)


def simpson(f, a, b, n=2000):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


def sample(rng, k=K, lam=LAM):
    """Inverse transform: solve 1 - F(x) = U for x."""
    return lam * (-math.log(1 - rng.random())) ** (1 / k)


if __name__ == "__main__":
    y0 = (100 / 120) ** 3
    print(f"y0 = (100/120)^3 = 125/216 = {y0:.6f}")
    print(f"P(X > 100) = e^-y0 = {sf(100):.6f}")
    print(f"  Simpson 100..1000: {simpson(pdf, 100, 1000):.6f}")
    g = math.gamma(4 / 3)
    print(f"Gamma(4/3) = Gamma(1/3)/3 = {g:.6f}")
    mean = moment(1)
    print(f"E[X] = 120 * Gamma(4/3) = {mean:.4f}")
    xf = lambda x: x * pdf(x)
    print(f"  Simpson 0..1000:   {simpson(xf, 0, 1000):.4f}")
    var = moment(2) - mean ** 2
    print(f"Var = 120^2 (Gamma(5/3) - Gamma(4/3)^2) = {var:.2f}")
    print(f"sd = {math.sqrt(var):.2f}")
    med = 120 * math.log(2) ** (1 / 3)
    print(f"median = 120 (ln 2)^(1/3) = {med:.2f}")
    for x in (50, 100, 150):
        print(f"hazard h({x}) = 3x^2/120^3 = {3 * x**2 / 120**3:.5f}")

    rng = random.Random(0)
    n = 10 ** 6
    xs = [sample(rng) for _ in range(n)]
    p = sum(x > 100 for x in xs) / n
    m = sum(xs) / n
    print(f"sim (n=10^6, seed 0): P(X>100) = {p:.4f}"
          f" (se {math.sqrt(p * (1 - p) / n):.4f}),"
          f" mean = {m:.2f} (se {math.sqrt(var / n):.2f})")

    print("same scale, shape k=1 (exponential, mean 120):")
    print(f"  P(X > 100) = e^(-100/120) = {sf(100, 1):.4f}")

    print("Gamma(alpha=3, theta=40) as a sum of 3 Exp(mean 40):")
    a, th = 3, 40
    s = [sum(-th * math.log(1 - rng.random()) for _ in range(a))
         for _ in range(10 ** 5)]
    ms = sum(s) / len(s)
    vs = sum((v - ms) ** 2 for v in s) / (len(s) - 1)
    print(f"  mean a*th = {a * th}, var a*th^2 = {a * th * th}")
    print(f"  sim 10^5: mean {ms:.1f}, var {vs:.0f}")
    print(f"  Gamma(n) = (n-1)!: Gamma(4) = {math.gamma(4):.0f},"
          f" Gamma(1/2)^2 = {math.gamma(0.5) ** 2:.6f} = pi")
