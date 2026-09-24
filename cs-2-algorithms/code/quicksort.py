"""Quicksort: iterative, explicit stack, Lomuto partition.

Python rewrite of the cheat-sheet's C++ version. One change: the
larger side is pushed first, so the smaller side is popped (and
finished) first and the stack never holds more than ~log2(n) ranges.
"""
import math
import random


def partition(a, lo, hi):          # Lomuto, pivot = a[hi]
    p, i = a[hi], lo - 1           # a[lo..i] holds values < p
    for j in range(lo, hi):
        if a[j] < p:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[hi] = a[hi], a[i + 1]   # pivot lands between
    return i + 1


def quicksort(a):
    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue
        m = partition(a, lo, hi)
        left, right = (lo, m - 1), (m + 1, hi)
        if m - lo < hi - m:        # push the larger side first,
            stack += [right, left]  # so the smaller is done first
        else:
            stack += [left, right]


# ---- instrumented versions for the numbers on the card ----

def counted(a, pick=None, smaller_first=True):
    """Same algorithm; returns (compares, peak stack size).
    pick(lo, hi) -> index of the pivot to swap into a[hi]."""
    stack, compares, peak = [(0, len(a) - 1)], 0, 1
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue
        if pick:
            k = pick(lo, hi)
            a[k], a[hi] = a[hi], a[k]
        m = partition(a, lo, hi)
        compares += hi - lo          # one a[j] < p per j
        left, right = (lo, m - 1), (m + 1, hi)
        if smaller_first and m - lo < hi - m:
            stack += [right, left]
        else:
            stack += [left, right]  # the source's order
        peak = max(peak, len(stack))
    return compares, peak


def median_of_3(a):
    def pick(lo, hi):
        mid = (lo + hi) // 2
        return sorted([lo, mid, hi], key=lambda k: a[k])[1]
    return pick


def trace(a):
    """Every partition in the source's order (right side popped
    first), as on the card."""
    a = list(a)
    stack = [(0, len(a) - 1)]
    print("start       ", a)
    while stack:
        lo, hi = stack.pop()
        if lo < hi:
            p = a[hi]
            m = partition(a, lo, hi)
            print(f"{lo}..{hi} pivot {p:<3}", a, f"-> index {m},",
                  f"{hi - lo} compares")
            stack += [(lo, m - 1), (m + 1, hi)]


def expected_compares(n):
    """Exact average for distinct keys in random order:
    C(n) = (n - 1) + (2/n) * sum(C(k) for k < n)
         = 2(n + 1)H_n - 4n."""
    c, s = [0.0] * (n + 1), 0.0
    for k in range(1, n + 1):
        s += c[k - 1]
        c[k] = (k - 1) + 2 * s / k
    return c[n]


if __name__ == "__main__":
    a = [38, 27, 43, 3, 9, 82, 10]
    trace(a)
    b = list(a)
    quicksort(b)
    print("quicksort(a):", b, "compares:", counted(list(a))[0])

    n = 8
    print("n=8 sorted, last pivot:", counted(list(range(n)))[0],
          "= n(n-1)/2 =", n * (n - 1) // 2)
    print("n=8 all equal, last pivot:", counted([5] * n)[0])
    s = list(range(n))
    print("n=8 sorted, median-of-3:", counted(s, median_of_3(s))[0],
          "(partition compares only)")
    print("n=7 random order, exact average:",
          round(expected_compares(7), 2))

    N = 10**6
    print(f"n=1e6 average compares: {expected_compares(N):,.0f}"
          f"  (2n ln n = {2 * N * math.log(N):,.0f})")
    print(f"n=1e6 worst case:       {N * (N - 1) // 2:,}")
    print(f"log2(n!) lower bound:   "
          f"{math.lgamma(N + 1) / math.log(2):,.0f}")

    rev = list(range(2000, 0, -1))
    print("peak stack, reversed n=2000, source order:",
          counted(list(rev), smaller_first=False)[1])
    print("peak stack, reversed n=2000, smaller first:",
          counted(list(rev))[1])

    random.seed(1)
    for _ in range(1000):
        n = random.randint(0, 30)
        x = [random.randint(0, 20) for _ in range(n)]
        y, z = list(x), list(x)
        quicksort(y)
        counted(z, lambda lo, hi: random.randint(lo, hi))
        assert y == z == sorted(x)
    print("1,000 random lists match sorted(): OK")
