"""Bond pricing, Macaulay and modified duration, convexity.

Run: python3 bonds-duration.py   (Python 3.10, stdlib only)
Annual coupons, face 100, yield compounded once a year.
"""


def cash_flows(face, coupon, years):
    """(t, cash) for t = 1..years; the last year repays face."""
    c = face * coupon
    return [(t, c + (face if t == years else 0))
            for t in range(1, years + 1)]


def price(flows, y):
    """Discount every cash flow at yield y."""
    return sum(cf / (1 + y) ** t for t, cf in flows)


def macaulay(flows, y):
    """PV-weighted average time until the cash arrives."""
    p = price(flows, y)
    return sum(t * cf / (1 + y) ** t for t, cf in flows) / p


def modified(flows, y):
    """-(dP/dy)/P: % price drop per unit rise in yield."""
    return macaulay(flows, y) / (1 + y)


def convexity(flows, y):
    """(d2P/dy2)/P: how much the price curve bends."""
    p = price(flows, y)
    s = sum(t * (t + 1) * cf / (1 + y) ** (t + 2)
            for t, cf in flows)
    return s / p


if __name__ == "__main__":
    flows = cash_flows(100, 0.04, 5)
    y = 0.05
    p = price(flows, y)
    print("5-year 4% annual-coupon bond, face 100, yield 5%")
    print("  t   cash      PV   PV/P   t*PV/P")
    for t, cf in flows:
        pv = cf / (1 + y) ** t
        print(f"  {t} {cf:6.2f} {pv:7.4f} {pv / p:6.4f}"
              f" {t * pv / p:8.4f}")
    d_mac = macaulay(flows, y)
    d_mod = modified(flows, y)
    cx = convexity(flows, y)
    print(f"  price     {p:.4f}")
    print(f"  Macaulay  {d_mac:.4f} years")
    print(f"  modified  {d_mod:.4f}")
    print(f"  convexity {cx:.4f}")
    print(f"  DV01      {d_mod * p * 1e-4:.4f} per 100 face")

    print("yield move: actual vs duration vs duration+convexity")
    for dy in (0.0001, 0.01, -0.01, 0.03):
        actual = price(flows, y + dy) - p
        lin = -d_mod * p * dy
        quad = lin + 0.5 * cx * p * dy ** 2
        print(f"  {dy * 1e4:+5.0f} bp  actual {actual:+8.4f}"
              f"  dur {lin:+8.4f}  +conv {quad:+8.4f}")

    zero = cash_flows(100, 0.0, 5)
    print(f"5y zero-coupon: Macaulay {macaulay(zero, y):.4f}")
    par = cash_flows(100, 0.05, 5)
    print(f"5y 5% coupon at 5%: price {price(par, y):.4f}")
