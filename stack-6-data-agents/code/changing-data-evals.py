"""Grading one golden case while the warehouse moves underneath it.

The case: "What was Q3 net revenue for EMEA?" (fiscal Q3, Apr-Jun).
Three agents answer it in July and again in October. All revenue
figures are illustrative ($M). Five grading strategies score the
answers in each month; we count how many verdicts are right.
"""

# Ledger rows: (kind, amount). Net = sales - refunds,
# excluding intercompany transfers (the metric contract says so).
JULY = [
    ("sales", 48.20), ("refund", 1.90),
]
OCTOBER_EXTRA = [
    ("refund", 0.85),         # late refunds posted back to Q3
    ("sales", 0.60),          # backfill of a missed Irish feed
    ("intercompany", 1.40),   # new EMEA entity, first transfer
]
GOLDEN = 46.30                # expected value written in July
TOL = 0.01                    # +-1% tolerance, illustrative


def net(rows, exclude_ic=True):
    total = 0.0
    for kind, amt in rows:
        if kind == "sales":
            total += amt
        elif kind == "refund":
            total -= amt
        elif kind == "intercompany" and not exclude_ic:
            total += amt
    return round(total, 2)


def canon(month):
    return JULY + (OCTOBER_EXTRA if month == "oct" else [])


def answers(month):
    rows = canon(month)
    legacy = JULY  # deprecated table stopped loading in August
    return {
        "healthy": dict(value=net(rows), table="fct_revenue",
                        excl_ic=True),
        "legacy table": dict(value=net(legacy),
                             table="rev_daily_legacy", excl_ic=True),
        "no IC filter": dict(value=net(rows, exclude_ic=False),
                             table="fct_revenue", excl_ic=False),
    }


TRULY_OK = {"healthy": True, "legacy table": False,
            "no IC filter": False}


def strategies(month):
    ref_now = net(canon(month))  # canonical SQL, run today
    snap = answers("jul")        # every agent on July data
    return {
        "fixed golden": lambda n, a: a["value"] == GOLDEN,
        "golden +-1%": lambda n, a:
            abs(a["value"] - GOLDEN) <= TOL * GOLDEN,
        "July snapshot": lambda n, a: snap[n]["value"] == GOLDEN,
        "reference now": lambda n, a: a["value"] == ref_now,
        "path check": lambda n, a:
            a["table"] == "fct_revenue" and a["excl_ic"],
    }


def score(month):
    ans = answers(month)
    out = {}
    for name, grade in strategies(month).items():
        verdicts = {n: grade(n, a) for n, a in ans.items()}
        right = sum(v == TRULY_OK[n] for n, v in verdicts.items())
        out[name] = (verdicts, right)
    return ans, out


def slices():
    # (cases, pass before, pass after) per slice, illustrative
    s = {"Product": (180, 144, 162), "Finance": (20, 15, 0)}
    n = sum(c for c, _, _ in s.values())
    before = sum(b for _, b, _ in s.values()) / n
    after = sum(a for _, _, a in s.values()) / n
    return s, before, after


if __name__ == "__main__":
    for month in ("jul", "oct"):
        ans, out = score(month)
        vals = ", ".join(f"{n} {a['value']:.2f}"
                         for n, a in ans.items())
        print(f"{month}: {vals}")
        for name, (v, right) in out.items():
            marks = " ".join("P" if x else "F" for x in v.values())
            print(f"  {name:14s} {marks}   right {right}/3")
    s, before, after = slices()
    print(f"overall {before:.1%} -> {after:.1%}")
    for k, (c, b, a) in s.items():
        print(f"  {k:8s} {b}/{c} = {b/c:.0%} -> {a}/{c} = {a/c:.0%}")
