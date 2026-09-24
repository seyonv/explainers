"""Log-structured memory, RAMCloud style, in miniature.

Memory is a log of fixed-size segments. Writes append at the head;
a hash table maps each key to its record. Deletes and overwrites
append a tombstone. The cleaner copies a segment's live records to
the head, repoints the hash table, and frees the segment.

Segments hold 4 equal-size records here (illustrative; RAMCloud's
segments are 8 MB and records vary in size).
"""

SEG = 4  # records per segment (illustrative)


class Log:
    def __init__(self):
        self.segs = {}      # segment id -> list of records
        self.table = {}     # key -> (segment id, slot): the hash table
        self.head = None
        self.next_id = 0

    def _append(self, rec):
        if self.head is None or len(self.segs[self.head]) == SEG:
            self.head, self.next_id = self.next_id, self.next_id + 1
            self.segs[self.head] = []
        self.segs[self.head].append(rec)
        return self.head, len(self.segs[self.head]) - 1

    def put(self, key, val):
        old = self.table.get(key)
        self.table[key] = self._append(("obj", key, val))
        if old is not None:          # overwrite: old copy is dead
            self._append(("tomb", key, old[0]))

    def delete(self, key):
        seg, _ = self.table.pop(key)
        self._append(("tomb", key, seg))  # names the dead segment

    def live(self, sid, i):
        kind, key, x = self.segs[sid][i]
        if kind == "obj":            # live iff the table points here
            return self.table.get(key) == (sid, i)
        return x in self.segs        # tombstone: target seg exists

    def util(self, sid):
        recs = self.segs[sid]
        return sum(self.live(sid, i) for i in range(len(recs))) / SEG

    def clean(self, sid):
        keep = [r for i, r in enumerate(self.segs[sid])
                if self.live(sid, i)]
        for rec in keep:                 # copy live records to the head
            where = self._append(rec)
            if rec[0] == "obj":
                self.table[rec[1]] = where   # the one pointer to swap
        del self.segs[sid]               # free the whole segment
        return len(keep), SEG - len(keep)    # copied, freed


def cost(u):
    """Bytes copied per byte freed when cleaning at utilization u."""
    return u / (1 - u)


def show(log, msg):
    def name(sid, i):
        kind, key, x = log.segs[sid][i]
        s = key if kind == "obj" else "t" + key + str(x)
        return s if log.live(sid, i) else "(" + s + ")"
    segs = "  ".join(
        f"s{sid}[{' '.join(name(sid, i) for i in range(len(r)))}]"
        for sid, r in log.segs.items())
    print(f"{msg:<14} {segs}")


if __name__ == "__main__":
    log = Log()
    for k in "ABCD":
        log.put(k, 1)
    show(log, "put A B C D")
    log.put("A", 2)
    show(log, "put A again")
    log.delete("B")
    log.delete("C")
    show(log, "delete B, C")
    print(f"utilization: s0 {log.util(0):.0%}, s1 {log.util(1):.0%}")
    c, f = log.clean(0)
    show(log, f"clean s0: {c}/{f}")
    print(f"utilization: s1 {log.util(1):.0%}  (tombstones died)")
    c, f = log.clean(1)
    show(log, f"clean s1: {c}/{f}")
    print("table:", log.table)
    print()
    for pct in (50, 60, 70, 75, 80, 85, 90, 95):
        u = pct / 100
        print(f"u = {pct}%  copy {cost(u):5.2f} bytes per byte freed")
