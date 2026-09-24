"""Money transfer without a distributed transaction.

The request log is the only place a transfer is committed: one
append of (request_id, from, to, amount). A processor reads the log
in order and sends a debit to the payer's partition and, if that
succeeds, a credit to the payee's. Every instruction carries the
request_id, and each account partition remembers the ids it has
already applied, with their result. So any replay (a crash, a
client retry, reprocessing the whole log) changes nothing.

Accounts and amounts are illustrative.
"""


class Account:
    """One account partition: a balance plus the ids it has seen."""

    def __init__(self, balance, dedup=True):
        self.balance, self.dedup, self.seen = balance, dedup, {}

    def apply(self, rid, delta):
        if self.dedup and rid in self.seen:
            return self.seen[rid]           # replay: same answer
        if self.balance + delta < 0:
            result = "rejected"             # debit: funds too low
        else:
            self.balance += delta
            result = "ok"
        self.seen[rid] = result             # same partition, same
        return result                       # write as the balance


def process(log, accts, start=0, crash_at=None):
    """Replay log[start:]; crash_at=rid stops after its debit."""
    status = {}
    for rid, src, dst, amt in log[start:]:
        status[rid] = accts[src].apply(rid, -amt)   # debit first
        if rid == crash_at:
            return status                   # crash: no credit yet
        if status[rid] == "ok":
            accts[dst].apply(rid, +amt)     # then credit
    return status


LOG = [("r1", "alice", "bob", 30),
       ("r2", "bob", "carol", 20),
       ("r3", "carol", "alice", 25),
       ("r4", "alice", "carol", 10)]
START = {"alice": 100, "bob": 50, "carol": 0}


def fresh(dedup):
    return {k: Account(v, dedup) for k, v in START.items()}


def show(label, accts):
    b = {k: a.balance for k, a in accts.items()}
    print(f"{label:<28}", b, "total", sum(b.values()))


def scenario(dedup):
    tag = "dedup" if dedup else "naive"
    a = fresh(dedup)
    process(LOG, a, crash_at="r2")          # crash mid-r2
    show(f"{tag}: crash after r2 debit", a)
    process(LOG, a, start=1)                # restart from offset 1
    show(f"{tag}: restart from r2", a)
    process(LOG + [LOG[3]], a, start=4)     # client retries r4
    show(f"{tag}: client retries r4", a)
    st = process(LOG, a)                    # replay whole log...
    process(LOG, a)                         # ...twice
    show(f"{tag}: replay whole log x2", a)
    print(f"{tag}: 1st replay status", st)


if __name__ == "__main__":
    ok = fresh(True)
    print("status", process(LOG, ok))
    show("one clean pass", ok)
    print()
    scenario(dedup=False)
    print()
    scenario(dedup=True)
