"""Sliding window median: two heaps with lazy deletion.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md,
Lists > Arrays > Sliding Window Median. The source is C++ (a multiset
plus an iterator parked on the median). Python has no multiset, so
this is the standard Python answer: a max-heap for the low half, a
min-heap for the high half, and deletions deferred until a value
reaches the top of its heap.
"""
import bisect
import heapq
from collections import defaultdict


def median_sliding_window(nums, k):
    lo, hi = [], []            # lo: max-heap (negated), hi: min-heap
    gone = defaultdict(int)    # value -> copies waiting to be deleted
    for x in nums[:k]:
        heapq.heappush(lo, -x)
    for _ in range(k // 2):    # lo keeps the extra one when k is odd
        heapq.heappush(hi, -heapq.heappop(lo))
    def median():
        return float(-lo[0]) if k % 2 else (-lo[0] + hi[0]) / 2
    out = [median()]
    for i in range(k, len(nums)):
        x, old = nums[i], nums[i - k]
        gone[old] += 1         # lazy: just remember it left
        bal = -1 if old <= -lo[0] else 1
        if x <= -lo[0]:
            heapq.heappush(lo, -x); bal += 1
        else:
            heapq.heappush(hi, x); bal -= 1
        if bal < 0:            # lo is one short: borrow hi's min
            heapq.heappush(lo, -heapq.heappop(hi))
        elif bal > 0:          # lo is one over: give hi its max
            heapq.heappush(hi, -heapq.heappop(lo))
        while lo and gone[-lo[0]]:     # delete only at the tops
            gone[-heapq.heappop(lo)] -= 1
        while hi and gone[hi[0]]:
            gone[heapq.heappop(hi)] -= 1
        out.append(median())
    return out


def median_bisect(nums, k):
    """Baseline: keep the window sorted with bisect. O(k) per step."""
    w = sorted(nums[:k])
    out = []
    for i in range(k, len(nums) + 1):
        out.append((w[k // 2] + w[(k - 1) // 2]) / 2)
        if i == len(nums):
            break
        del w[bisect.bisect_left(w, nums[i - k])]
        bisect.insort(w, nums[i])
    return out


def brute_force(nums, k):
    out = []
    for i in range(len(nums) - k + 1):
        w = sorted(nums[i:i + k])
        out.append((w[k // 2] + w[(k - 1) // 2]) / 2)
    return out


def trace(nums, k):
    """Same algorithm, printing both heaps after every step."""
    lo, hi, gone = [], [], defaultdict(int)
    for x in nums[:k]:
        heapq.heappush(lo, -x)
    for _ in range(k // 2):
        heapq.heappush(hi, -heapq.heappop(lo))

    def show(step, note):
        low = sorted((-v for v in lo), reverse=True)
        med = -lo[0] if k % 2 else (-lo[0] + hi[0]) / 2
        print(f"{step:<5}{note:<26} lo={low!s:<14}"
              f" hi={sorted(hi)!s:<10} gone={dict(gone)} -> {med}")

    show("init", f"window {nums[:k]}")
    for i in range(k, len(nums)):
        x, old = nums[i], nums[i - k]
        gone[old] += 1
        bal = -1 if old <= -lo[0] else 1
        side = "lo" if x <= -lo[0] else "hi"
        if side == "lo":
            heapq.heappush(lo, -x)
            bal += 1
        else:
            heapq.heappush(hi, x)
            bal -= 1
        move = ""
        if bal < 0:
            v = heapq.heappop(hi)
            heapq.heappush(lo, -v)
            move = f" {v}:hi->lo"
        elif bal > 0:
            v = -heapq.heappop(lo)
            heapq.heappush(hi, v)
            move = f" {v}:lo->hi"
        pruned = []
        while lo and gone[-lo[0]]:
            v = -heapq.heappop(lo)
            gone[v] -= 1
            pruned.append(v)
        while hi and gone[hi[0]]:
            v = heapq.heappop(hi)
            gone[v] -= 1
            pruned.append(v)
        gone = defaultdict(int, {a: b for a, b in gone.items() if b})
        note = f"out {old} in {x}>{side}{move}"
        if pruned:
            note += f" drop{pruned}"
        show(f"i={i}", note)


if __name__ == "__main__":
    import random

    nums, k = [1, 3, -1, -3, 5, 3, 6, 7], 3
    print(median_sliding_window(nums, k))
    trace(nums, k)
    print(median_sliding_window([1, 2, 3, 4, 2, 3, 1, 4, 2], 4))

    rng = random.Random(0)
    for _ in range(1000):
        n = rng.randint(1, 30)
        a = [rng.randint(-5, 5) for _ in range(n)]
        k = rng.randint(1, n)
        want = brute_force(a, k)
        assert median_sliding_window(a, k) == want
        assert median_bisect(a, k) == want
    print("1000 random cases match brute force")
