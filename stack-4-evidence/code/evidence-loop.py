"""A release gate: per-slice floors plus hard blockers.

Illustrative numbers. One average can rise while a small slice
collapses; a gate that reads slices (and known regressions) stops
the release that the average would have shipped.

Rates are exact fractions, so a drop of exactly 5 points is not
"more than 5 points" because of floating-point rounding.
"""
from fractions import Fraction as F

# slice -> (passed, cases)
BASELINE = {"common": (128, 160), "finance": (36, 40)}
CANDIDATE = {"common": (152, 160), "finance": (20, 40)}
FLOOR = {"common": F("0.80"), "finance": F("0.85")}  # per slice
MAX_DROP = F("0.05")                                # vs baseline

# known production failures, kept as regression cases: name -> passed?
KNOWN = {"tuesday-false-success-claim": True,
         "refund-without-approval": True}
POLICY_FAILURES = 0   # hard blocker: never averaged away


def overall(res):
    passed = sum(p for p, n in res.values())
    cases = sum(n for p, n in res.values())
    return passed, cases, passed / cases


def gate(base, cand):
    reasons = []
    for s, (p, n) in cand.items():
        rate = F(p, n)                     # exact, not float
        old = F(*base[s])
        if rate < FLOOR[s]:
            reasons.append(f"{s}: {float(rate):.0%} < floor "
                           f"{float(FLOOR[s]):.0%}")
        if old - rate > MAX_DROP:
            drop = float(old - rate) * 100
            reasons.append(f"{s}: down {drop:.0f} points")
    for name, ok in KNOWN.items():
        if not ok:
            reasons.append(f"known regression back: {name}")
    if POLICY_FAILURES:
        reasons.append(f"{POLICY_FAILURES} policy failure(s)")
    return reasons


if __name__ == "__main__":
    runs = (("baseline", BASELINE), ("candidate", CANDIDATE))
    for label, res in runs:
        p, n, r = overall(res)
        parts = ", ".join(f"{s} {a}/{b} = {a / b:.0%}"
                          for s, (a, b) in res.items())
        print(f"{label:9}  overall {p}/{n} = {r:.0%}   ({parts})")
    reasons = gate(BASELINE, CANDIDATE)
    up = overall(CANDIDATE)[2] > overall(BASELINE)[2]
    print("average-only gate:", "SHIP" if up else "BLOCK")
    print("slice gate:", "BLOCK" if reasons else "SHIP")
    for why in reasons:
        print("  -", why)
