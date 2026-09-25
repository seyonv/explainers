"""Heapsort, in place, following the source guide's C++ HeapSort.

Phase 1 (build): sift down every internal node, last one first, so
the array becomes a max-heap (children of i live at 2i+1 and 2i+2).
Phase 2 (extract): swap the max at a[0] to the end, shrink the heap
by one, and sift the new root down. Repeat until one item is left.
"""
import random


def sift_down(a, root, end):
    while (child := 2 * root + 1) < end:
        if child + 1 < end and a[child] < a[child + 1]:
            child += 1                    # pick the bigger child
        if a[root] >= a[child]:
            break                         # heap property holds
        a[root], a[child] = a[child], a[root]
        root = child                      # follow it down


def sift_logged(a, root, end):
    """Same as sift_down, but returns the (up, down) swaps it made."""
    log = []
    while (child := 2 * root + 1) < end:
        if child + 1 < end and a[child] < a[child + 1]:
            child += 1
        if a[root] >= a[child]:
            break
        a[root], a[child] = a[child], a[root]
        log.append((a[root], a[child]))
        root = child
    return log


def heapsort(a):
    n = len(a)
    for start in range(n // 2 - 1, -1, -1):   # build: O(n)
        sift_down(a, start, n)
    for end in range(n - 1, 0, -1):           # extract: O(n log n)
        a[0], a[end] = a[end], a[0]           # max goes to its place
        sift_down(a, 0, end)
    return a


def trace(a):
    a = list(a)
    n = len(a)
    print("build a max-heap")
    for start in range(n // 2 - 1, -1, -1):
        log = sift_logged(a, start, n)
        print(f"  sift i={start}  swaps={log}  {a}")
    print("extract the max, n-1 times")
    total = 0
    for end in range(n - 1, 0, -1):
        top = a[0]
        a[0], a[end] = a[end], a[0]
        log = sift_logged(a, 0, end)
        total += len(log)
        print(f"  {top:>2} -> a[{end}]  swaps={log}  "
              f"heap={a[:end]} sorted={a[end:]}")
    print("sift swaps in phase 2:", total)
    return a


def count_ops(a):
    """Comparisons between elements, per phase."""
    a = list(a)
    n = len(a)
    c = [0]

    def lt(x, y):
        c[0] += 1
        return x < y

    def sift(root, end):
        while (child := 2 * root + 1) < end:
            if child + 1 < end and lt(a[child], a[child + 1]):
                child += 1
            if not lt(a[root], a[child]):
                break
            a[root], a[child] = a[child], a[root]
            root = child

    for start in range(n // 2 - 1, -1, -1):
        sift(start, n)
    build = c[0]
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift(0, end)
    return build, c[0] - build


def sum_of_heights(n):
    """Most swaps build can do: each node sinks at most its height."""
    return sum(height(i, n) for i in range(n))


def height(i, n):
    h = 0
    while 2 * i + 1 < n:
        i = 2 * i + 1
        h += 1
    return h


class Rec:
    """A record compared by key only, to show heapsort is not stable."""

    def __init__(self, key, tag):
        self.key, self.tag = key, tag

    def __lt__(self, other):
        return self.key < other.key

    def __ge__(self, other):
        return self.key >= other.key

    def __repr__(self):
        return f"{self.key}{self.tag}"


if __name__ == "__main__":
    a = [38, 27, 43, 3, 9, 82, 10]
    print(heapsort(list(a)))
    out = trace(a)
    print("comparisons (build, extract):", count_ops(a))
    for n in (7, 1023, 10**6):
        s = sum_of_heights(n)
        print(f"n={n}: sum of heights = {s:,}"
              f"  (n - popcount(n) = {n - bin(n).count('1'):,})")
    rng = random.Random(1)
    xs = [rng.random() for _ in range(10**4)]
    print("n=10^4 random, comparisons (build, extract):",
          count_ops(xs))
    for keys in ([2, 2], [1, 1, 0]):
        recs = [Rec(k, t) for k, t in zip(keys, "abc")]
        print("not stable:", recs, "->", heapsort(list(recs)))
    for _ in range(1000):
        xs = [rng.randint(-50, 50) for _ in range(rng.randint(0, 40))]
        assert heapsort(list(xs)) == sorted(xs)
    print("1,000 random lists: matches sorted()")
