"""One question, eight valid queries, eight answers.

"What was Q3 net revenue for EMEA?" has (at least) three hidden
choices. Each choice is a legal reading of the words:

  metric  net = amount minus refunds | the net_amount column as-is
  time    calendar Q3 (Jul-Sep)      | fiscal Q3, FY from Feb (Aug-Oct)
  region  billing country in EMEA    | account owned by the EMEA team

All eight combinations run as SQL against the same toy warehouse
(sqlite3, in memory). Every query succeeds. None of them fails a
test, because there is no test: the check is a definition.
The data is illustrative, generated from a fixed seed.
Python 3.10, standard library only.
"""
import itertools
import random
import sqlite3
from datetime import date, timedelta

SEED = 6
EMEA = {"DE", "FR", "GB", "NL", "AE", "ZA"}
OTHER = {"US", "CA", "JP", "AU", "BR"}


def build(seed=SEED):
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE accounts(id, country, owner_region)")
    db.execute("CREATE TABLE orders(account_id, day, net_amount,"
               " refund)")
    countries = sorted(EMEA) + sorted(OTHER)
    for i in range(80):
        c = rng.choice(countries)
        home = "EMEA" if c in EMEA else "OTHER"
        # 1 in 5 accounts is owned by a team in another region
        owner = home
        if rng.random() < 0.2:
            owner = "OTHER" if home == "EMEA" else "EMEA"
        db.execute("INSERT INTO accounts VALUES (?,?,?)",
                   (i, c, owner))
    start = date(2026, 6, 1)
    for _ in range(1200):
        day = start + timedelta(days=rng.randrange(183))
        amt = round(rng.lognormvariate(8.0, 0.9), 2)
        refund = amt if rng.random() < 0.08 else 0.0
        db.execute("INSERT INTO orders VALUES (?,?,?,?)",
                   (rng.randrange(80), day.isoformat(), amt, refund))
    return db


METRIC = {"minus refunds": "o.net_amount - o.refund",
          "column as-is": "o.net_amount"}
TIME = {"calendar Q3": ("2026-07-01", "2026-09-30"),
        "fiscal Q3": ("2026-08-01", "2026-10-31")}
REGION = {"billing country": "a.country IN ({})".format(
              ",".join(f"'{c}'" for c in sorted(EMEA))),
          "owner team": "a.owner_region = 'EMEA'"}


def readings(db):
    out = []
    for m, t, r in itertools.product(METRIC, TIME, REGION):
        lo, hi = TIME[t]
        sql = (f"SELECT COUNT(*), SUM({METRIC[m]}) FROM orders o"
               f" JOIN accounts a ON a.id = o.account_id"
               f" WHERE o.day BETWEEN '{lo}' AND '{hi}'"
               f" AND {REGION[r]}")
        rows, total = db.execute(sql).fetchone()  # never raises
        out.append((m, t, r, rows, total))
    return out


if __name__ == "__main__":
    res = readings(build())
    print(f"{'metric':14} {'time':12} {'region':16} rows"
          f"  status  net revenue")
    for m, t, r, n, v in res:
        print(f"{m:14} {t:12} {r:16} {n:4}  ok      "
              f"${v / 1000:8.1f}k")
    vals = [v for *_, v in res]
    lo, hi = min(vals), max(vals)
    print(f"\n8 of 8 queries succeeded; answers span "
          f"${lo / 1000:.1f}k to ${hi / 1000:.1f}k")
    print(f"largest / smallest = {hi / lo:.2f}x")
