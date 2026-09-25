"""Bottom-up merge sort, like the source guide's C++ version.

Merge runs of width 1, then 2, then 4, ... until one run covers the
whole list. Each pass copies every element once into a buffer, and
there are ceil(log2 n) passes, so the work is O(n log n) always.
"""
import math
import random


def merge_sort(a, key=lambda x: x, log=None):
    a = list(a)
    n = len(a)
    width = 1
    while width < n:
        b = []
        for lo in range(0, n, 2 * width):
            mid, hi = min(lo + width, n), min(lo + 2 * width, n)
            i, j = lo, mid
            while i < mid and j < hi:
                if key(a[j]) < key(a[i]):  # strict: ties take left
                    b.append(a[j]); j += 1
                else:
                    b.append(a[i]); i += 1
            b += a[i:mid] + a[j:hi]         # one side ran out
        a = b
        if log is not None:
            log.append((width, a))
        width *= 2
    return a


def count_compares(a):
    """Same passes, returning compares made in each pass."""
    a, n, width, per_pass = list(a), len(a), 1, []
    while width < n:
        b, c = [], 0
        for lo in range(0, n, 2 * width):
            mid, hi = min(lo + width, n), min(lo + 2 * width, n)
            i, j = lo, mid
            while i < mid and j < hi:
                c += 1
                if a[j] < a[i]:
                    b.append(a[j]); j += 1
                else:
                    b.append(a[i]); i += 1
            b += a[i:mid] + a[j:hi]
        a, width = b, width * 2
        per_pass.append(c)
    return per_pass


if __name__ == "__main__":
    a = [38, 27, 43, 3, 9, 82, 10]
    log = []
    print("start      ", a)
    merge_sort(a, log=log)
    for width, arr in log:
        print(f"width {width} -> ", arr)
    cs = count_compares(a)
    print("compares per pass:", cs, "total:", sum(cs))
    n = len(a)
    print("passes = ceil(log2 7) =", math.ceil(math.log2(n)))
    print("lower bound log2(7!) = %.1f" % math.log2(math.factorial(n)))

    cards = [("A", 2), ("B", 1), ("C", 2), ("D", 1)]
    print(merge_sort(cards, key=lambda t: t[1]))

    rng = random.Random(1)
    for _ in range(1000):
        xs = [(rng.randint(0, 9), k) for k in range(rng.randint(0, 40))]
        assert merge_sort(xs, key=lambda t: t[0]) == sorted(
            xs, key=lambda t: t[0])
    print("1,000 random lists: sorted and stable (matches sorted())")
    for m in (1_000, 10**6):
        worst = m * math.ceil(math.log2(m))
        print(f"n={m:,}: passes={math.ceil(math.log2(m))}, "
              f"<= {worst:,} compares")
