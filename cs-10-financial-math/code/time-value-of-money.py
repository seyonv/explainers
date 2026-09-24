"""Time value of money: compounding, annuities, NPV and IRR.

Run: python3 time-value-of-money.py   (Python 3.10, stdlib only)
"""
import math


def fv(pv, rate, years, m=1):
    """Future value with m compounding periods per year."""
    return pv * (1 + rate / m) ** (m * years)


def fv_cont(pv, rate, years):
    """Future value with continuous compounding."""
    return pv * math.exp(rate * years)


def annuity_pv(pmt, rate, n):
    """PV of n payments of pmt at the end of each period."""
    return pmt * (1 - (1 + rate) ** -n) / rate


def npv(rate, flows):
    """flows[t] arrives at the end of period t (t = 0 is now)."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(flows))


def irr(flows, lo=-0.99, hi=1.0, tol=1e-10, trace=None):
    """Bisection: the rate where NPV crosses zero."""
    f_lo = npv(lo, flows)
    if f_lo * npv(hi, flows) > 0:
        raise ValueError("NPV has the same sign at both ends")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        f_mid = npv(mid, flows)
        if trace is not None:
            trace.append((lo, hi, mid, f_mid))
        if (f_mid > 0) == (f_lo > 0):   # root is above mid
            lo, f_lo = mid, f_mid
        else:                           # root is below mid
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    print("$1,000 at 5% for 10 years")
    for name, m in [("annual", 1), ("quarterly", 4),
                    ("monthly", 12), ("daily", 365)]:
        eff = (1 + 0.05 / m) ** m - 1
        print(f"  {name:<10} {fv(1000, 0.05, 10, m):9,.2f}"
              f"   effective {eff:.4%}")
    print(f"  {'continuous':<10} {fv_cont(1000, 0.05, 10):9,.2f}"
          f"   effective {math.expm1(0.05):.4%}")
    print(f"  PV of 1,000 due in 10y: {1000 / 1.05 ** 10:,.2f}")

    print("30-year $1,000/month annuity at 6% (0.5%/month)")
    pv = annuity_pv(1000, 0.06 / 12, 360)
    print(f"  PV = {pv:,.2f}  (paid in: {360 * 1000:,})")
    first = 1000 / 1.005
    last = 1000 / 1.005 ** 360
    print(f"  month 1 worth {first:.2f} today, "
          f"month 360 worth {last:.2f}")

    flows = [-1000, 300, 400, 500]
    print(f"cash flows {flows}")
    for r in (0.0, 0.05, 0.10):
        print(f"  NPV at {r:4.0%}: {npv(r, flows):8.2f}")
    steps = []
    rate = irr(flows, trace=steps)
    for i, (lo, hi, mid, f) in enumerate(steps[:8], 1):
        print(f"  step {i}: [{lo:7.3%}, {hi:7.3%}] "
              f"mid {mid:7.3%} NPV {f:8.2f}")
    print(f"  IRR = {rate:.4%} after {len(steps)} bisection steps")
    print(f"  NPV at IRR: {npv(rate, flows):.1e}")
