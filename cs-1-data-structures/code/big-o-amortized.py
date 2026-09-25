"""Big-O and amortized O(1): growth, doubling, RandomizedCollection.

Source: ljeng/cheat-sheet, coding-algorithms/algorithms.md, Big-O
Notation > Insert, Delete, GetRandom O(1) - Duplicates Allowed.
The source stores (value, slot) pairs and index lists; this copy uses
index sets, and both are checked against a brute force below.
"""
import collections
import math
import random
import sys


class RandomizedCollection:
    def __init__(self):
        self.vals = []                           # multiset, any order
        self.idx = collections.defaultdict(set)  # value -> positions

    def insert(self, val):
        self.idx[val].add(len(self.vals))
        self.vals.append(val)                    # amortized O(1)
        return len(self.idx[val]) == 1

    def remove(self, val):
        if not self.idx[val]:
            return False
        i = self.idx[val].pop()                  # any copy of val
        last = self.vals[-1]
        self.vals[i] = last                      # fill the hole
        self.idx[last].add(i)
        self.idx[last].discard(len(self.vals) - 1)
        self.vals.pop()                          # O(1): drop end
        return True

    def get_random(self):
        return random.choice(self.vals)          # index a list


class SourceRandomizedCollection:
    """The source guide's version: (value, slot in its index list)."""

    def __init__(self):
        self.vals = []
        self.val_index = collections.defaultdict(list)

    def insert(self, val):
        self.vals.append((val, len(self.val_index[val])))
        self.val_index[val].append(len(self.vals) - 1)
        return len(self.val_index[val]) == 1

    def remove(self, val):
        if self.val_index[val]:
            i = self.val_index[val].pop()
            last = last_val, last_index = self.vals.pop()
            if i < len(self.vals):
                self.vals[i] = last
                self.val_index[last_val][last_index] = i
            return True
        return False


def growth_table(ns=(10**3, 10**6)):
    rows = [("O(1)", lambda n: 1), ("O(log n)", math.log2),
            ("O(n)", lambda n: n),
            ("O(n log n)", lambda n: n * math.log2(n)),
            ("O(n^2)", lambda n: n * n)]
    for name, f in rows:
        print(f"{name:<11}" + "".join(f"{f(n):>18,.0f}" for n in ns))


def doubling_copies(n, grow=lambda cap: 2 * cap):
    """Cost of each of n appends: 1 write + copies on a resize."""
    cap, size, costs = 1, 0, []
    for _ in range(n):
        copies = 0
        if size == cap:
            copies, cap = size, grow(cap)
        size += 1
        costs.append(1 + copies)
    return costs


def cpython_resizes(n):
    """Appends at which CPython's list got a bigger buffer."""
    lst, last, at = [], sys.getsizeof([]), []
    for k in range(1, n + 1):
        lst.append(None)
        s = sys.getsizeof(lst)
        if s != last:
            at.append(k)
            last = s
    return at


def brute_check(cls, seed=0, ops=20_000):
    rng = random.Random(seed)
    rc, bag = cls(), collections.Counter()
    for _ in range(ops):
        v = rng.randint(0, 5)
        if rng.random() < 0.55:
            assert rc.insert(v) == (bag[v] == 0)
            bag[v] += 1
        else:
            assert rc.remove(v) == (bag[v] > 0)
            if bag[v]:
                bag[v] -= 1
        got = collections.Counter(
            x[0] if isinstance(x, tuple) else x for x in rc.vals)
        assert got == +bag


if __name__ == "__main__":
    print("steps for n = 1,000 and n = 1,000,000")
    growth_table()

    c = doubling_copies(16)
    print("doubling, cost per append:", c)
    print("  total", sum(c), "= 16 writes +", sum(c) - 16, "copies")
    c1 = doubling_copies(16, grow=lambda cap: cap + 1)
    print("grow by 1: total", sum(c1), "=", 16, "writes +",
          sum(c1) - 16, "copies")
    for n in (10**3, 10**6):
        d = sum(doubling_copies(n)) - n
        print(f"n={n:,}: doubling copies {d:,} ({d / n:.3f} each)")

    r = cpython_resizes(10**6)
    print("CPython 3.10 resizes in the first 16 appends:",
          [k for k in r if k <= 16])
    print("CPython resizes in 10^6 appends:", len(r))

    random.seed(0)
    rc = RandomizedCollection()
    print(rc.insert(1), rc.insert(1), rc.insert(2), rc.vals,
          dict(rc.idx))
    draws = collections.Counter(rc.get_random() for _ in range(30_000))
    print("30,000 draws:", dict(sorted(draws.items())))
    print(rc.remove(1), rc.vals,
          {k: v for k, v in rc.idx.items() if v})
    print(rc.remove(1), rc.vals,
          {k: v for k, v in rc.idx.items() if v})
    print(rc.remove(1), rc.vals)

    brute_check(RandomizedCollection)
    brute_check(SourceRandomizedCollection)
    print("20,000 random ops: both versions match a Counter")
