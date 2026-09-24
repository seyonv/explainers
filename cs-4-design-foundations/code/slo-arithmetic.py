"""Availability and SLO arithmetic.

Source: ljeng/cheat-sheet, large-scale-design/system-design/
distributed-systems.md. Its image-resize service wants 99.9%
availability, sets the API tier at 99.99%, runs at least three
API servers at 5K RPS each for 10K RPS, calls "P^N < R" hogwash
because failures are correlated, and says manual failover is too
slow for a 99.9% SLA. It never converts any of these to minutes;
this file does.

Illustrative (not from the source): 99% per server, a 99.95%
common-mode layer (bad config push, shared power), a 60-minute
manual failover and a 2-minute automatic one.
"""

MONTH_MIN = 30 * 24 * 60      # 43,200 minutes in a 30-day month
YEAR_MIN = 365 * 24 * 60      # 525,600 minutes in a 365-day year


def downtime(a, minutes=MONTH_MIN):
    """Allowed downtime (the error budget) for availability a."""
    return (1 - a) * minutes


def serial(*parts):
    """Every part must be up: availabilities multiply."""
    out = 1.0
    for a in parts:
        out *= a
    return out


def parallel(p, k):
    """Any 1 of k independent copies is enough."""
    return 1 - (1 - p) ** k


def k_of_n(p, k, n):
    """At least k of n independent copies must be up."""
    from math import comb
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def mttr_budget(a, mttr):
    """Minimum mean time to failure so MTTF/(MTTF+MTTR) >= a."""
    return a / (1 - a) * mttr


if __name__ == "__main__":
    print("nines   per month      per year")
    for a in (0.99, 0.999, 0.9995, 0.9999, 0.99999):
        print(f"{a:<7} {downtime(a):7.2f} min  "
              f"{downtime(a, YEAR_MIN):8.2f} min")

    print("\nserial chain of n parts at 99.99% each")
    for n in (1, 5, 20):
        a = serial(*[0.9999] * n)
        print(f"n={n:<3} {a:.6f}  {downtime(a):6.2f} min/month")

    print("\nsource: 99.9% overall, API tier 99.99%")
    rest = 0.999 / 0.9999
    print(f"rest of the path must reach {rest:.6f}")

    print("\nredundancy, 99% per server (illustrative)")
    rows = [("1 server", 0.99),
            ("1 of 2", parallel(0.99, 2)),
            ("1 of 3", parallel(0.99, 3)),
            ("2 of 3 (source API tier)", k_of_n(0.99, 2, 3)),
            ("1 of 3 + 99.95% shared",
             serial(0.9995, parallel(0.99, 3)))]
    for name, a in rows:
        print(f"{name:<25} {a:.6f}  {downtime(a):6.2f} min")

    lo, hi = 0.99, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if k_of_n(mid, 2, 3) >= 0.9999 \
            else (mid, hi)
    print(f"2 of 3 reaches 99.99% at p >= {hi:.5f}")

    print("\nerror budget at 99.9%")
    print(f"{downtime(0.999):.1f} min/month; "
          f"{10_000 * 30 * 86_400 * 0.001:,.0f} of "
          f"{10_000 * 30 * 86_400:,} requests at 10K RPS")
    for mttr in (60, 2):
        m = mttr_budget(0.999, mttr) / 1440
        print(f"MTTR {mttr:>2} min -> need MTTF >= {m:.2f} days")
