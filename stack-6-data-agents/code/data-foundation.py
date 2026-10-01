"""Three tables could answer "What was Q3 net revenue for EMEA?".

Each one runs, returns a plausible number, and disagrees with the
others. A name-matching picker chooses the deprecated table; a
picker that reads each table's asset contract (status, freshness,
grain) keeps only the canonical one.

All tables, rows and amounts (EUR thousands) are illustrative.
"""

# EMEA orders in Q3: (order, customer, month, gross, refund, test?)
ORDERS = [
    ("o1", "c1", 7, 1200, 0, False),
    ("o2", "c2", 7, 800, 100, False),
    ("o3", "c3", 8, 1500, 0, False),
    ("o4", "c4", 8, 600, 600, False),   # fully refunded
    ("o5", "c5", 9, 2000, 200, False),
    ("o6", "c6", 9, 900, 0, False),
    ("o7", "t1", 8, 1000, 0, True),     # internal test account
]

# Region history: c3 and c5 changed region record, so 2 rows each.
REGION_ROWS = {"c1": 1, "c2": 1, "c3": 2, "c4": 1,
               "c5": 2, "c6": 1, "t1": 1}


def fct_net_revenue():
    """Canonical: net of refunds, test accounts excluded."""
    return sum(g - r for _, _, _, g, r, t in ORDERS if not t)


def emea_net_revenue_v1():
    """Deprecated: gross, keeps test accounts, last load Aug 31."""
    return sum(g for _, _, m, g, _, _ in ORDERS if m <= 8)


def orders_enriched():
    """Scratch: net, but joined to region history per customer,
    so a customer with 2 region rows is counted twice."""
    return sum((g - r) * REGION_ROWS[c]
               for _, c, _, g, r, t in ORDERS if not t)


TABLES = {  # name: (function, asset contract)
    "fct_net_revenue": (fct_net_revenue, dict(
        status="canonical", fresh_through=9, grain="order")),
    "emea_net_revenue_v1": (emea_net_revenue_v1, dict(
        status="deprecated", fresh_through=8, grain="order")),
    "orders_enriched": (orders_enriched, dict(
        status="scratch", fresh_through=9, grain="unknown")),
}

QUESTION = {"q3", "net", "revenue", "emea"}


def name_score(name):
    return len(QUESTION & set(name.split("_")))


def pick_by_name():
    return max(TABLES, key=name_score)


def pick_by_contract(last_month=9):
    return [n for n, (_, c) in TABLES.items()
            if c["status"] == "canonical"
            and c["fresh_through"] >= last_month
            and c["grain"] != "unknown"]


if __name__ == "__main__":
    truth = fct_net_revenue()
    for name, (fn, c) in TABLES.items():
        v = fn()
        print(f"{name:<20} {c['status']:<10} name={name_score(name)}"
              f"  {v:>5}k  {100 * (v - truth) / truth:+.1f}%")
    print("pick by name:    ", pick_by_name())
    print("pick by contract:", pick_by_contract())
    mined, kept = 40, 0.125 * 40   # 12.5%: Spotify's acceptance rate
    print(f"of {mined} mined pairs at 12.5%: {kept:.0f} kept,"
          f" {mined - kept:.0f} rejected")
