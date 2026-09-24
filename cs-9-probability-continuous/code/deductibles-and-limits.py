"""Deductibles, limits and coinsurance: two exam-P problems.

1. Auto: car worth 15, deductible 1 (thousands). P(partial) = 0.04,
   damage X ~ c*e^(-x/2) on (0, 15); P(total loss) = 0.02.
   Expected claim payment per policy year.
2. Health: cost X ~ Exp(mean 100), deductible 20, 100% paid up
   to 120, 50% above. G(115) = P(Y <= 115 | Y > 0).
"""
import math
import random

C = 1 / (2 * (1 - math.exp(-7.5)))   # makes the density integrate to 1


def auto_sf(x):
    """P(X > x) for the partial-damage density on (0, 15)."""
    return C * 2 * (math.exp(-x / 2) - math.exp(-7.5))


def simpson(f, a, b, n=1000):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


def auto_expected(d=1, value=15):
    # E[(X - d)+] = integral of the survival function above d
    layer = simpson(auto_sf, d, value)
    return 0.04 * layer + 0.02 * (value - d), layer


def health_pay(x, d=20, top=120):
    """Reimbursement for a cost of x."""
    if x <= d:
        return 0.0
    if x <= top:
        return x - d
    return (top - d) + 0.5 * (x - top)


def health_cost_for(y, d=20, top=120):
    """Invert health_pay for y > 0: which cost pays exactly y?"""
    return d + y if y <= top - d else top + (y - (top - d)) / 0.5


def health_G(y, mean=100, d=20):
    x = health_cost_for(y)
    F = lambda t: 1 - math.exp(-t / mean)
    return (F(x) - F(d)) / (1 - F(d))   # condition on Y > 0


def sim_auto(n=10**6, seed=0):
    rng, total, sq = random.Random(seed), 0.0, 0.0
    for _ in range(n):
        pay = 0.0
        u = rng.random()
        if u < 0.02:
            pay = 14                     # total loss: 15 - 1
        elif u < 0.06:
            # inverse CDF of the truncated exponential on (0, 15)
            v = rng.random() * (1 - math.exp(-7.5))
            pay = max(-2 * math.log(1 - v) - 1, 0)
        total, sq = total + pay, sq + pay * pay
    mean = total / n
    return mean, math.sqrt((sq / n - mean**2) / n)


def sim_health(n=10**6, seed=0):
    rng, pos, hit = random.Random(seed), 0, 0
    for _ in range(n):
        y = health_pay(rng.expovariate(1 / 100))
        if y > 0:
            pos += 1
            hit += y <= 115
    return hit / pos, pos


if __name__ == "__main__":
    print(f"c = {C:.6f}  (source prints 0.5003)")
    pay, layer = auto_expected()
    print(f"E[(X-1)+] = {layer:.4f}  closed form "
          f"{2 * C * (2 * math.exp(-0.5) - 16 * math.exp(-7.5)):.4f}")
    print(f"E[payment] = 0.04*{layer:.4f} + 0.02*14"
          f" = {pay:.4f} thousand")
    m, se = sim_auto()
    print(f"sim auto: {m:.4f} thousand (se {se:.4f})")
    print(f"cost for Y=115: {health_cost_for(115):g}")
    print(f"G(115) = {health_G(115):.4f}"
          f"  = 1 - e^-1.3 = {1 - math.exp(-1.3):.4f}")
    print(f"unconditional P(0 < Y <= 115) = "
          f"{math.exp(-0.2) - math.exp(-1.5):.4f}  (source's 0.596)")
    g, pos = sim_health()
    se = math.sqrt(g * (1 - g) / pos)
    print(f"sim health: {g:.4f} from {pos:,} payments (se {se:.4f})")
    for y in (50, 100, 115, 150):
        x = health_cost_for(y)
        print(f"  G({y}) = {health_G(y):.4f}  (cost {x:g})")
