"""LFU cache with O(1) get and put.

Source: ljeng/cheat-sheet, coding-algorithms/coding.md,
Object-oriented Design and Programming > LFU Cache (Java).
Rewritten in Python: key -> value, key -> count, and
count -> OrderedDict of keys (oldest use first, so ties are LRU).
The source's eviction calls key_count.remove(key) on the NEW key
instead of the evicted one; SourceLFU below reproduces that leak.
"""
from collections import OrderedDict, defaultdict


class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.val = {}                        # key -> value
        self.cnt = {}                        # key -> use count
        self.keys = defaultdict(OrderedDict)  # count -> keys, LRU first
        self.min = 0                         # smallest count in use

    def _touch(self, key):                   # count += 1, move bucket
        c = self.cnt[key]
        del self.keys[c][key]
        if not self.keys[c]:
            del self.keys[c]
            if self.min == c:
                self.min = c + 1             # key went to c + 1
        self.cnt[key] = c + 1
        self.keys[c + 1][key] = None         # newest in its bucket

    def get(self, key):
        if key not in self.val:
            return -1
        self._touch(key)
        return self.val[key]

    def put(self, key, value):
        if self.cap <= 0:
            return
        if key in self.val:
            self.val[key] = value
            self._touch(key)
            return
        if len(self.val) == self.cap:        # evict: min count, LRU
            old, _ = self.keys[self.min].popitem(last=False)
            if not self.keys[self.min]:
                del self.keys[self.min]
            del self.val[old], self.cnt[old]  # the evicted key
        self.val[key], self.cnt[key] = value, 1
        self.keys[1][key] = None
        self.min = 1


class SourceLFU:
    """Line-by-line port of the source's Java, bug included."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache, self.key_count = {}, {}
        self.count_keys = {1: OrderedDict()}
        self.min = 0

    def get(self, key):
        if key in self.cache:
            count = self.key_count[key]
            del self.count_keys[count][key]
            if count == self.min and not self.count_keys[count]:
                self.min += 1
            count += 1
            self.key_count[key] = count
            self.count_keys.setdefault(count, OrderedDict())[key] = 1
            return self.cache[key]
        return -1

    def put(self, key, value):
        if key in self.cache:
            self.get(key)
            self.cache[key] = value
            return
        elif len(self.cache) >= self.capacity:
            lfu = next(iter(self.count_keys[self.min]))
            del self.cache[lfu]
            self.key_count.pop(key, None)    # BUG: should be lfu
            del self.count_keys[self.min][lfu]
        self.cache[key] = value
        self.key_count[key] = 1
        self.count_keys[1][key] = 1
        self.min = 1


class BruteLFU:
    """O(n) reference: scan for (count, last use) on eviction."""

    def __init__(self, capacity):
        self.cap, self.d, self.t = capacity, {}, 0

    def get(self, key):
        if key not in self.d:
            return -1
        self.t += 1
        v, c, _ = self.d[key]
        self.d[key] = (v, c + 1, self.t)
        return v

    def put(self, key, value):
        if self.cap <= 0:
            return
        self.t += 1
        if key in self.d:
            _, c, _ = self.d[key]
            self.d[key] = (value, c + 1, self.t)
            return
        if len(self.d) == self.cap:
            old = min(self.d, key=lambda k: self.d[k][1:])
            del self.d[old]
        self.d[key] = (value, 1, self.t)


def show(c):
    """Buckets as {count: [keys, LRU first]}."""
    return {n: list(ks) for n, ks in sorted(c.keys.items())}


if __name__ == "__main__":
    import random

    c = LFUCache(2)
    ops = [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3),
           ("get", 2), ("get", 3), ("put", 4, 4), ("get", 1),
           ("get", 3), ("get", 4)]
    for op, *args in ops:
        out = getattr(c, op)(*args)
        call = f"{op}({', '.join(map(str, args))})"
        res = "" if out is None else f"-> {out}"
        print(f"{call:<10} {res:<6} buckets={show(c)} min={c.min}")

    rng = random.Random(0)
    for trial in range(1000):
        cap = rng.randint(1, 4)
        a, b = LFUCache(cap), BruteLFU(cap)
        for _ in range(60):
            k = rng.randint(0, 6)
            if rng.random() < 0.5:
                assert a.get(k) == b.get(k), trial
            else:
                v = rng.randint(0, 99)
                a.put(k, v)
                b.put(k, v)
    print("1000 random op sequences match brute force")

    src, fixed = SourceLFU(2), LFUCache(2)
    for k in range(1000):
        src.put(k, k)
        fixed.put(k, k)
    print("after 1000 puts, cap 2: source key_count size",
          len(src.key_count), "| fixed", len(fixed.cnt))
