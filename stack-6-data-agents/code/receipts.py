"""A tiny receipt checker for data-agent answers.

A receipt is a dict the runtime writes while the analyst loop runs.
check() reads it and lists what the receipt cannot vouch for. Each
rule maps to a failure mode in Govindarajan's Data Agent Stack
Part 5 (tools) or Part 6 (trust).

What it never checks: whether the number is right. A clean receipt
makes a run explainable; only an eval says it was correct.

All receipts below are illustrative, built for this card.
"""


def check(r):
    flags = []
    # Part 5, failure 7: a number with no SQL, tables or result link.
    if not (r.get("sql") and r.get("tables") and r.get("result_ref")):
        flags.append("no evidence: a sentence with a number")
    # Part 6, failure 1: the actor reads what the requester can't.
    extra = set(r.get("actor_reads", [])) - set(r.get("may_read", []))
    if extra and not r.get("policy_as_requester"):
        flags.append("privilege bridge: " + ", ".join(sorted(extra)))
    # Part 6, failure 8: a fallback source that nobody was told about.
    if r.get("fallback") and not r.get("fallback_disclosed"):
        flags.append("silent fallback: " + r["fallback"])
    # Part 5, failure 5: defaults applied but not surfaced.
    hidden = set(r.get("defaults", [])) - set(r.get("assumptions", []))
    if hidden:
        flags.append("hidden default: " + ", ".join(sorted(hidden)))
    # Part 6, failure 7: query authority is not publish authority.
    if r.get("destination") not in r.get("may_publish_to", []):
        flags.append("unapproved destination: "
                     + str(r.get("destination")))
    return flags or ["explainable (not proof it is right)"]


EMEA = {  # "What was Q3 net revenue for EMEA?"
    "sql": "select sum(net_revenue) from fct_revenue where ...",
    "tables": ["fct_revenue"],
    "result_ref": "q_7f3a",
    "may_read": ["fct_revenue"],
    "actor_reads": ["fct_revenue", "fct_invoices_customer"],
    "policy_as_requester": True,
    "fallback": None,
    "defaults": ["fiscal quarter", "is_test = false"],
    "assumptions": ["fiscal quarter", "is_test = false"],
    "destination": "reply to requester",
    "may_publish_to": ["reply to requester"],
}

NUMBER_ONLY = dict(EMEA, sql=None, result_ref=None, assumptions=[])

BRIDGE = dict(EMEA, policy_as_requester=False,
              destination="#finance-all")

SWAPPED = dict(EMEA, fallback="weekly summary",
               fallback_disclosed=False)

if __name__ == "__main__":
    for name, r in [("EMEA, full receipt", EMEA),
                    ("same answer, number only", NUMBER_ONLY),
                    ("service identity, posted", BRIDGE),
                    ("denied, source swapped", SWAPPED)]:
        print(name)
        for f in check(r):
            print("  -", f)
