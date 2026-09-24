"""How to test your code: example, edge cases, then random cases.

Card: cs-1-data-structures/testing-your-code.html
The worked example is an LFU cache (see lfu-cache.html). Three
versions are tested against the same 1,000 random op sequences:
  LFU        correct O(1) version
  BuggyLFU   evicts BEFORE checking whether the key already exists
  LeakyLFU   the cheat-sheet's own bug: forgets the evicted key's
             count (outputs stay right, internal state is wrong)
Run: python3 testing-your-code.py
Tests: python3 -m pytest -q --import-mode=importlib <this file>
"""
import random
from collections import OrderedDict, defaultdict


class LFU:
    def __init__(self, cap):
        self.cap, self.val, self.cnt = cap, {}, {}
        self.keys = defaultdict(OrderedDict)  # count -> keys, LRU 1st
        self.min = 0

    def _touch(self, k):                      # count += 1
        c = self.cnt[k]
        del self.keys[c][k]
        if not self.keys[c]:
            del self.keys[c]
            if self.min == c:
                self.min = c + 1
        self.cnt[k] = c + 1
        self.keys[c + 1][k] = None

    def _evict(self):                         # min count, then LRU
        old, _ = self.keys[self.min].popitem(last=False)
        if not self.keys[self.min]:
            del self.keys[self.min]
        del self.val[old]
        self._forget(old)

    def _forget(self, old):
        del self.cnt[old]

    def get(self, k):
        if k not in self.val:
            return -1
        self._touch(k)
        return self.val[k]

    def put(self, k, v):
        if self.cap <= 0:
            return
        if k in self.val:                     # update, never evict
            self.val[k] = v
            self._touch(k)
            return
        if len(self.val) == self.cap:
            self._evict()
        self.val[k], self.cnt[k] = v, 1
        self.keys[1][k] = None
        self.min = 1


class BuggyLFU(LFU):
    """Checks 'full?' before 'already here?': a classic slip."""

    def put(self, k, v):
        if self.cap <= 0:
            return
        if len(self.val) == self.cap:         # BUG: runs on updates
            self._evict()
        if k in self.val:
            self.val[k] = v
            self._touch(k)
            return
        self.val[k], self.cnt[k] = v, 1
        self.keys[1][k] = None
        self.min = 1


class LeakyLFU(LFU):
    """The source's key_count.remove(key): drops the NEW key's
    count instead of the evicted one's, so old counts pile up."""

    def _forget(self, old):
        pass


class Brute:
    """Obviously right, O(n) per op: evict min (count, last use)."""

    def __init__(self, cap):
        self.cap, self.d, self.t = cap, {}, 0

    def get(self, k):
        if k not in self.d:
            return -1
        self.t += 1
        v, c, _ = self.d[k]
        self.d[k] = (v, c + 1, self.t)
        return v

    def put(self, k, v):
        if self.cap <= 0:
            return
        self.t += 1
        if k in self.d:
            self.d[k] = (v, self.d[k][1] + 1, self.t)
            return
        if len(self.d) == self.cap:
            del self.d[min(self.d, key=lambda x: self.d[x][1:])]
        self.d[k] = (v, 1, self.t)


def run(cls, cap, ops):
    c = cls(cap)
    return [getattr(c, op)(*args) for op, *args in ops], c


def random_case(rng):
    cap = rng.randint(1, 3)
    ops = []
    for _ in range(rng.randint(1, 12)):
        k = rng.randint(0, 4)                 # few keys: many hits
        ops.append(("get", k) if rng.random() < 0.5
                   else ("put", k, rng.randint(0, 9)))
    return cap, ops


def fails(cls, cap, ops):
    got, c = run(cls, cap, ops)
    return got != run(Brute, cap, ops)[0] or set(c.cnt) != set(c.val)


def shrink(cls, cap, ops):
    """Drop one op at a time while the case still fails."""
    i = 0
    while i < len(ops):
        smaller = ops[:i] + ops[i + 1:]
        if fails(cls, cap, smaller):
            ops = smaller
        else:
            i += 1
    return ops


def check(cls, cases=1000, seed=0):
    rng = random.Random(seed)
    wrong_out = bad_state = 0
    first = None
    for _ in range(cases):
        cap, ops = random_case(rng)
        got, c = run(cls, cap, ops)
        out_bad = got != run(Brute, cap, ops)[0]
        state_bad = set(c.cnt) != set(c.val)  # invariant
        wrong_out += out_bad
        bad_state += state_bad
        if (out_bad or state_bad) and first is None:
            first = (cap, ops)
    return wrong_out, bad_state, first


EXAMPLE = [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3),
           ("get", 2), ("get", 3), ("put", 4, 4), ("get", 1),
           ("get", 3), ("get", 4)]
EXPECTED = [None, None, 1, None, -1, 3, None, -1, 3, 4]


def fmt(ops):
    return ", ".join(f"{op}({', '.join(map(str, a))})"
                     for op, *a in ops)


# pytest shape: run with  python3 -m pytest -q --import-mode=importlib
def test_leetcode_example():
    assert run(LFU, 2, EXAMPLE)[0] == EXPECTED


def test_update_does_not_evict():             # the shrunk failure
    ops = [("put", 0, 2), ("put", 4, 7), ("put", 4, 7), ("get", 0)]
    assert run(LFU, 2, ops)[0] == [None, None, None, 2]


def test_matches_brute_force():
    assert check(LFU)[:2] == (0, 0)


def largest_rectangle(heights):               # monotonic-stack card
    h, stack, best = heights + [0], [-1], 0
    for i, x in enumerate(h):
        while x < h[stack[-1]]:
            top = stack.pop()
            best = max(best, h[top] * (i - stack[-1] - 1))
        stack.append(i)
    return best


def atoi32(s):                                # edge-case-parsing card
    s = s.lstrip(" ")
    sign = -1 if s[:1] == "-" else 1
    s = s[1:] if s[:1] in "+-" else s
    n = 0
    for ch in s:
        if not ch.isdigit():
            break
        n = n * 10 + int(ch)
    return max(-2**31, min(2**31 - 1, sign * n))


if __name__ == "__main__":
    print("1. hand example, capacity 2:", fmt(EXAMPLE))
    for cls in (LFU, BuggyLFU, LeakyLFU):
        out = run(cls, 2, EXAMPLE)[0]
        print(f"   {cls.__name__:<9}", out, "PASS" if out == EXPECTED
              else "FAIL")

    print("2. edge cases")
    print("   largest_rectangle([])  =", largest_rectangle([]))
    print("   largest_rectangle([4]) =", largest_rectangle([4]))
    print("   atoi32('-91283472332') =", atoi32("-91283472332"))
    n = 10**5
    print("   pairs for n = 10**5    =", f"{n * (n - 1) // 2:,}")
    c = LFU(1)
    c.put(1, 1)
    c.put(2, 2)
    print("   LFU(1): put(1,1), put(2,2), get(1) =", c.get(1))
    for cls in (LFU, BuggyLFU, LeakyLFU):
        c = cls(2)
        c.put(1, 1)
        c.put(2, 2)
        c.put(1, 10)
        print(f"   {cls.__name__:<9} put 1,2 then put(1,10):",
              "get(1) =", c.get(1), " get(2) =", c.get(2))

    print("3. 1,000 random cases vs Brute (seed 0)")
    for cls in (LFU, BuggyLFU, LeakyLFU):
        wrong, bad, first = check(cls)
        print(f"   {cls.__name__:<9} wrong output {wrong:>4}"
              f"   broken invariant {bad:>4}")
        if first:
            cap, ops = first
            small = shrink(cls, cap, ops)
            print(f"      first failure: cap {cap}, {len(ops)} ops;"
                  f" shrunk to {len(small)}: {fmt(small)}")
            got = run(cls, cap, small)
            want = run(Brute, cap, small)[0]
            print("      got ", got[0], " cnt keys",
                  sorted(got[1].cnt), "\n      want", want)
