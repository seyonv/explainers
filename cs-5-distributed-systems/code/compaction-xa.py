"""Compaction, deletion and XA: two small simulations.

1. An append-only key-value log. A delete is a tombstone record;
   the old value stays on disk until compaction rewrites the
   segments, and a backup taken earlier keeps it forever.
2. Two-phase commit (2PC) with n participants. Counts messages
   (4n) and shows participants stuck "in doubt" when the
   coordinator crashes after collecting the YES votes.

Run: python3 compaction-xa.py   (Python 3.10, stdlib only)
"""

TOMB = None  # a tombstone: "this key was deleted"


def compact(segments, oldest_included):
    """Merge segments (oldest first); keep the newest record per key.

    A tombstone can be dropped only if no older segment is left out
    of this compaction, or the old value would resurface.
    """
    latest = {}
    for seg in segments:
        for key, value in seg:
            latest[key] = value
    return [(k, v) for k, v in latest.items()
            if not (v is TOMB and oldest_included)]


def show(label, records):
    body = ", ".join(f"{k}={'TOMB' if v is TOMB else v}"
                     for k, v in records)
    print(f"  {label:<22}{len(records)} records: {body}")


def demo_compaction():
    print("1. append-only log: delete u1, then compact")
    seg1 = [("u1", "ann@x"), ("u2", "bo@x"), ("u1", "ann@y")]
    seg2 = [("u2", "bo@y"), ("u1", TOMB), ("u3", "cy@x")]
    backup = list(seg1)                  # nightly backup of seg1
    show("seg1 (old)", seg1)
    show("seg2 (new)", seg2)
    show("compact seg2 only", compact([seg2], False))
    show("compact seg1+seg2", compact([seg1, seg2], True))
    show("backup of seg1", backup)
    live = compact([seg1, seg2], True)
    on_disk = len(seg1) + len(seg2)
    print(f"  space before/after: {on_disk} -> {len(live)} records")
    leaked = any("ann" in str(v) for _, v in backup)
    print(f"  'ann' still in backup? {leaked}")


class Participant:
    def __init__(self, name):
        self.name, self.state = name, "working"

    def prepare(self):           # write YES to disk, keep locks
        self.state = "prepared"
        return "YES"

    def finish(self, decision):  # COMMIT or ABORT, release locks
        self.state = {"COMMIT": "committed",
                      "ABORT": "aborted"}[decision]
        return "ACK"


def two_phase_commit(parts, crash_after_votes=False):
    msgs = 0
    votes = []
    for p in parts:                      # phase 1
        msgs += 1                        # PREPARE ->
        votes.append(p.prepare())
        msgs += 1                        # <- YES / NO
    decision = "COMMIT" if all(v == "YES" for v in votes) \
        else "ABORT"                     # logged: commit point
    if crash_after_votes:
        return msgs, None                # coordinator is down
    for p in parts:                      # phase 2
        msgs += 1                        # COMMIT ->
        p.finish(decision)
        msgs += 1                        # <- ACK
    return msgs, decision


def demo_2pc():
    print("2. two-phase commit")
    for n in (2, 3):
        parts = [Participant(f"P{i}") for i in range(1, n + 1)]
        msgs, d = two_phase_commit(parts)
        print(f"  n={n}: {msgs} messages, {d}, "
              f"{[p.state for p in parts]}")
    parts = [Participant("db"), Participant("queue")]
    msgs, d = two_phase_commit(parts, crash_after_votes=True)
    print(f"  crash after votes: {msgs} messages, decision {d}, "
          f"{[p.state for p in parts]}")
    print("  prepared = in doubt: can't commit or abort alone,")
    print("  holds its locks until the coordinator comes back")


def demo_numbers():
    print("3. the arithmetic on the card")
    a = 0.999                            # each part: 99.9% up
    month_min = 30 * 24 * 60
    for n in (1, 2, 3, 5):
        up = a ** (n + 1)                # n participants + coord
        print(f"  n={n}: msgs {4 * n:>2}, all up {up:.4%}, "
              f"down {(1 - up) * month_min:.1f} min/month")
    rtt, rate, restart = 0.008, 1000, 30   # s, tx/s, s
    print(f"  normal lock hold ~1 RTT = {rtt * 1000:.0f} ms")
    print(f"  in doubt at crash = {rate} tx/s x {rtt} s "
          f"= {rate * rtt:.0f} tx")
    print(f"  hold during restart = {restart} s = "
          f"{restart / rtt:,.0f}x the normal hold")
    print(f"  hot row, {rate} tx/s x {restart} s = "
          f"{rate * restart:,} tx blocked")


if __name__ == "__main__":
    demo_compaction()
    print()
    demo_2pc()
    print()
    demo_numbers()
