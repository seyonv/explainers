"""Black-Scholes prices and Greeks, checked by finite differences.

Run: python3 black-scholes-greeks.py   (Python 3.10, stdlib only)
"""
import math


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def d1_d2(s, k, r, sig, t):
    d1 = (math.log(s / k) + (r + sig**2 / 2) * t) / (sig * math.sqrt(t))
    return d1, d1 - sig * math.sqrt(t)


def price(s, k, r, sig, t, call=True):
    d1, d2 = d1_d2(s, k, r, sig, t)
    disc_k = k * math.exp(-r * t)            # K e^(-rT)
    if call:
        return s * norm_cdf(d1) - disc_k * norm_cdf(d2)
    return disc_k * norm_cdf(-d2) - s * norm_cdf(-d1)


def greeks(s, k, r, sig, t):
    """Call Greeks in closed form (vega per 1.00 of sigma)."""
    d1, d2 = d1_d2(s, k, r, sig, t)
    return {
        "delta": norm_cdf(d1),
        "gamma": norm_pdf(d1) / (s * sig * math.sqrt(t)),
        "vega": s * norm_pdf(d1) * math.sqrt(t),
    }


def bumped(s, k, r, sig, t, h=1e-3):
    """The same Greeks by re-pricing with small bumps."""
    c = lambda s_=s, g=sig: price(s_, k, r, g, t)
    return {
        "delta": (c(s + h) - c(s - h)) / (2 * h),
        "gamma": (c(s + h) - 2 * c() + c(s - h)) / h**2,
        "vega": (c(g=sig + h) - c(g=sig - h)) / (2 * h),
    }


def theta_rho(s, k, r, sig, t):
    """Call theta (per year) and rho (per 1.00 of r)."""
    d1, d2 = d1_d2(s, k, r, sig, t)
    disc_k = k * math.exp(-r * t)
    theta = (-s * norm_pdf(d1) * sig / (2 * math.sqrt(t))
             - r * disc_k * norm_cdf(d2))
    rho = t * disc_k * norm_cdf(d2)
    return theta, rho


if __name__ == "__main__":
    S, K, R, SIG, T = 100, 100, 0.05, 0.20, 1.0
    d1, d2 = d1_d2(S, K, R, SIG, T)
    print(f"d1 = {d1:.4f}  d2 = {d2:.4f}")
    print(f"N(d1) = {norm_cdf(d1):.4f}  N(d2) = {norm_cdf(d2):.4f}")
    c, p = price(S, K, R, SIG, T), price(S, K, R, SIG, T, False)
    print(f"call = {c:.4f}  put = {p:.4f}")
    rhs = S - K * math.exp(-R * T)
    print(f"parity: C - P = {c - p:.4f}  S - K e^-rT = {rhs:.4f}")

    g, b = greeks(S, K, R, SIG, T), bumped(S, K, R, SIG, T)
    print("greek   closed form   bump h=0.001   difference")
    for name in g:
        print(f"{name:6s}  {g[name]:11.6f}  {b[name]:13.6f}"
              f"   {g[name] - b[name]:+.1e}")
    for h in (1e-3, 1e-7):
        err = bumped(S, K, R, SIG, T, h)["gamma"] - g["gamma"]
        print(f"gamma bump h={h:g}: error {err:+.1e}")
    th, rho = theta_rho(S, K, R, SIG, T)
    print(f"theta = {th:.4f}/yr = {th / 365:.4f}/day"
          f"  rho = {rho:.4f}")

    pde = th + SIG**2 * S**2 * g["gamma"] / 2 + R * S * g["delta"]
    print(f"PDE check: theta + sig^2 S^2 gamma/2 + r S delta"
          f" = {pde:.4f} = r C = {R * c:.4f}")
    up = price(S, K, R, SIG + 0.01, T) - c
    print(f"sigma 20% -> 21%: call +{up:.4f}"
          f" (vega/100 = {g['vega'] / 100:.4f})")

    print("\ndelta hedge: sell 1 call, hold delta shares")
    for s1 in (99, 101, 105):
        dc = price(s1, K, R, SIG, T) - c
        pred = g["delta"] * (s1 - S)
        pred2 = pred + g["gamma"] * (s1 - S) ** 2 / 2
        print(f"  S={s1}: dC = {dc:+.4f}  delta says {pred:+.4f}"
              f"  +gamma {pred2:+.4f}")

    print("\ndelta and gamma across spot")
    for s1 in (80, 90, 100, 110, 120):
        g1 = greeks(s1, K, R, SIG, T)
        print(f"  S={s1:3d}  call={price(s1, K, R, SIG, T):7.4f}"
              f"  delta={g1['delta']:.4f}  gamma={g1['gamma']:.4f}")
