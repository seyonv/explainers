"""Binomial option pricing: one step, then a CRR tree of N steps.

Run: python3 binomial-pricing.py   (Python 3.10, stdlib only)
"""
import math


def one_step(s, up, down, k, r=0.0, t=1.0):
    """Price a call over one step by replication and by q."""
    cu, cd = max(up - k, 0), max(down - k, 0)
    delta = (cu - cd) / (up - down)          # shares to hold
    bond = (delta * down - cd) * math.exp(-r * t)  # borrowed
    replicate = delta * s - bond
    growth = math.exp(r * t)
    q = (s * growth - down) / (up - down)    # risk-neutral prob
    risk_neutral = (q * cu + (1 - q) * cd) / growth
    return delta, bond, q, replicate, risk_neutral


def crr(s, k, r, sigma, t, n, call=True, american=False):
    """Cox-Ross-Rubinstein tree, rolled back one layer at a time."""
    dt = t / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    sign = 1 if call else -1

    def payoff(x):
        return max(sign * (x - k), 0.0)

    # layer n: j up-moves, n - j down-moves
    v = [payoff(s * u**j * d**(n - j)) for j in range(n + 1)]
    for i in range(n - 1, -1, -1):           # step back to t = 0
        v = [disc * (q * v[j + 1] + (1 - q) * v[j])
             for j in range(i + 1)]
        if american:                          # exercise early?
            v = [max(x, payoff(s * u**j * d**(i - j)))
                 for j, x in enumerate(v)]
    return v[0]


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes(s, k, r, sigma, t, call=True):
    d1 = (math.log(s / k) + (r + sigma**2 / 2) * t)
    d1 /= sigma * math.sqrt(t)
    d2 = d1 - sigma * math.sqrt(t)
    c = s * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)
    return c if call else c - s + k * math.exp(-r * t)


def two_step_trace(s, k, r, sigma, t):
    """Print every node of a 2-step tree: stock, call value."""
    dt = t / 2
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    print(f"u={u:.4f} d={d:.4f} q={q:.4f} disc={disc:.4f}")
    v2 = [max(s * u**j * d**(2 - j) - k, 0) for j in range(3)]
    for j in (2, 1, 0):
        print(f"  t=1.0 j={j}: S={s * u**j * d**(2 - j):8.4f}"
              f"  C={v2[j]:.4f}")
    v1 = [disc * (q * v2[j + 1] + (1 - q) * v2[j]) for j in range(2)]
    for j in (1, 0):
        print(f"  t=0.5 j={j}: S={s * u**j * d**(1 - j):8.4f}"
              f"  C={v1[j]:.4f}")
    v0 = disc * (q * v1[1] + (1 - q) * v1[0])
    print(f"  t=0.0     : S={s:8.4f}  C={v0:.4f}")


if __name__ == "__main__":
    delta, bond, q, rep, rn = one_step(100, 110, 90, 100)
    print("one step 100 -> 110/90, K=100, r=0")
    print(f"  delta={delta} borrow={bond} q={q}")
    print(f"  replicate={rep} risk-neutral={rn}")

    S, K, R, SIG, T = 100, 100, 0.05, 0.20, 1.0
    print("\ntwo-step CRR, shared contract")
    two_step_trace(S, K, R, SIG, T)

    bs = black_scholes(S, K, R, SIG, T)
    print(f"\nBlack-Scholes call = {bs:.4f}")
    for n in (1, 2, 10, 11, 100, 101, 1000):
        c = crr(S, K, R, SIG, T, n)
        print(f"  N={n:5d}  CRR={c:.4f}  error={c - bs:+.4f}")

    eu = crr(S, K, R, SIG, T, 1000, call=False)
    am = crr(S, K, R, SIG, T, 1000, call=False, american=True)
    amc = crr(S, K, R, SIG, T, 1000, american=True)
    bsp = black_scholes(S, K, R, SIG, T, call=False)
    print(f"\nput, N=1000: European {eu:.4f} (BS {bsp:.4f})"
          f"  American {am:.4f}")
    print(f"American call, N=1000: {amc:.4f} (no dividends:"
          f" same as European)")
