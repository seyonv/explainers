"""All O(1) data structure: a doubly linked list of count buckets.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md,
Linked Lists > All O(1) Data Structure (C++ std::list of buckets plus
a key -> iterator map). Rewritten in Python: Python has no std::list
iterator, so we build a small doubly linked list with two sentinels
and keep a dict key -> bucket node as the "iterator".

The source's dec() reads prev->count before checking whether the
bucket is the first one, so it dereferences nodes.end() (undefined
behaviour) when the key sits in the lowest bucket. The sentinels
(count 0) make that case an ordinary comparison.
"""


class Bucket:
    __slots__ = ("count", "keys", "prev", "next")

    def __init__(self, count):
        self.count = count
        self.keys = {}           # dict as an ordered set
        self.prev = self.next = None


class AllOne:
    def __init__(self):
        self.head, self.tail = Bucket(0), Bucket(0)  # sentinels
        self.head.next, self.tail.prev = self.tail, self.head
        self.where = {}          # key -> its bucket: the handle

    def _add_after(self, node, count):
        b = Bucket(count)        # 4 pointer writes, O(1)
        b.prev, b.next = node, node.next
        node.next.prev = b
        node.next = b
        return b

    def _drop(self, key, b):
        del b.keys[key]
        if not b.keys:           # unlink: 2 pointer writes, O(1)
            b.prev.next, b.next.prev = b.next, b.prev

    def inc(self, key):
        cur = self.where.get(key, self.head)   # new key: count 0
        nxt = cur.next           # sentinel counts are 0: never equal
        if nxt.count != cur.count + 1:
            nxt = self._add_after(cur, cur.count + 1)
        nxt.keys[key] = None
        self.where[key] = nxt
        if cur is not self.head:
            self._drop(key, cur)

    def dec(self, key):
        cur = self.where.pop(key)
        if cur.count > 1:
            prv = cur.prev
            if prv.count != cur.count - 1:
                prv = self._add_after(prv, cur.count - 1)
            prv.keys[key] = None
            self.where[key] = prv
        self._drop(key, cur)

    def get_max_key(self):
        b = self.tail.prev
        return "" if b is self.head else next(iter(b.keys))

    def get_min_key(self):
        b = self.head.next
        return "" if b is self.tail else next(iter(b.keys))

    def buckets(self):
        b, out = self.head.next, []
        while b is not self.tail:
            out.append((b.count, list(b.keys)))
            b = b.next
        return out


def brute(ops):
    """Dict of counts, scanned for max/min: the O(n) baseline."""
    c, out = {}, []
    for op, k in ops:
        if op == "inc":
            c[k] = c.get(k, 0) + 1
        elif op == "dec":
            c[k] -= 1
            if not c[k]:
                del c[k]
        elif op == "max":
            out.append(max(c.values()) if c else 0)
        else:
            out.append(min(c.values()) if c else 0)
    return out


if __name__ == "__main__":
    import random

    s = AllOne()
    for op, k in [("inc", "a"), ("inc", "b"), ("inc", "a"),
                  ("dec", "b")]:
        getattr(s, op)(k)
        print(f"{op} {k}: {s.buckets()}  "
              f"max={s.get_max_key()!r} min={s.get_min_key()!r}")

    rng = random.Random(0)
    for _ in range(1000):
        s, ops, got = AllOne(), [], []
        for _ in range(rng.randint(0, 40)):
            op = rng.choice(["inc", "inc", "dec", "max", "min"])
            if op == "dec":
                if not s.where:
                    continue
                k = rng.choice(sorted(s.where))
            else:
                k = rng.choice("abcde")
            ops.append((op, k))
            if op in ("inc", "dec"):
                getattr(s, op)(k)
            else:
                key = (s.get_max_key() if op == "max"
                       else s.get_min_key())
                got.append(s.where[key].count if key else 0)
        assert got == brute(ops), ops
    print("1000 random op sequences match brute force")
