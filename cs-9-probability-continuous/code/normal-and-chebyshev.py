"""The normal distribution and Chebyshev's bound.

Blood pressure data (n = 15) from the source guide: range rule,
mean and sample sd, the Chebyshev k = 2 interval and how many
readings it actually holds; then Chebyshev vs the normal model for
k = 1, 2, 3, a distribution where Chebyshev is exact, and a
simulation check of the 68-95-99.7 rule.
"""
import math
import random
import statistics as st

BP = [172, 140, 123, 130, 115, 148, 108, 129,
      137, 161, 123, 152, 133, 128, 142]


def norm_cdf(z):
    """Standard normal CDF via the error function."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def chebyshev(k):
    """Lower bound on the share within k sd, any distribution."""
    return max(0.0, 1 - 1 / k**2)


def inside(xs, lo, hi):
    return sum(lo <= x <= hi for x in xs)


def worst_case(k):
    """-k, 0, +k with P(+-k) = 1/(2k^2): sd = 1, bound is exact."""
    p = 1 / (2 * k * k)
    var = 2 * p * k * k
    share_strictly_inside = 1 - 2 * p
    return math.sqrt(var), share_strictly_inside


def simulate(n=10**6, seed=0):
    rng = random.Random(seed)
    counts = [0, 0, 0]
    for _ in range(n):
        z = abs(rng.gauss(0, 1))
        for k in (1, 2, 3):
            counts[k - 1] += z < k
    return [c / n for c in counts]


if __name__ == "__main__":
    n = len(BP)
    print("range rule: s ~", (max(BP) - min(BP)) / 4)
    ybar, s = st.mean(BP), st.stdev(BP)       # stdev uses n - 1
    print(f"ybar = {ybar:.2f}, s = {s:.2f}")
    k = 2
    a, b = ybar - k * s, ybar + k * s
    print(f"k = 2: ({a:.2f}, {b:.2f}) holds "
          f"{inside(BP, a, b)} of {n}; source's (102, 170) holds "
          f"{inside(BP, 102, 170)}")
    print("outside:", [x for x in BP if not a <= x <= b])
    for k in (1, 2, 3):
        c = sum(abs(x - ybar) <= k * s for x in BP)
        normal = norm_cdf(k) - norm_cdf(-k)
        print(k, f"{c}/15", f"{chebyshev(k):.3f}", f"{normal:.4f}")
    sd, share = worst_case(2)
    print(f"worst case k=2: sd = {sd:.0f}, strictly inside = {share}")
    print("simulated normal:",
          " ".join(f"{p:.4f}" for p in simulate()))
