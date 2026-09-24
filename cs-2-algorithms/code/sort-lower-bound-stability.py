"""Why comparison sorts need about n log2 n compares, and stability.

A comparison sort learns about the input only by asking "is x < y?".
Each answer at best halves the orderings still possible, and n items
have n! orderings, so some input needs at least ceil(log2 n!) asks.
This file counts real compares against that bound, then shows why a
stable sort lets you sort by several keys in two passes.
"""
import itertools
import math
import random


class Counted:
    """Wraps a value and counts every < comparison made on it."""
    compares = 0

    def __init__(self, v):
        self.v = v

    def __lt__(self, other):
        Counted.compares += 1
        return self.v < other.v


def lower_bound(n):
    """log2(n!) via lgamma, so it works for n = 10**6 too."""
    return math.lgamma(n + 1) / math.log(2)


def compares(sort, a):
    Counted.compares = 0
    out = sort([Counted(x) for x in a])
    assert [c.v for c in out] == sorted(a)
    return Counted.compares


def insertion(a):
    for i in range(1, len(a)):
        j = i
        while j and a[j] < a[j - 1]:
            a[j - 1], a[j] = a[j], a[j - 1]
            j -= 1
    return a


def merge(a):
    n, width = len(a), 1
    while width < n:
        b = []
        for lo in range(0, n, 2 * width):
            mid, hi = min(lo + width, n), min(lo + 2 * width, n)
            i, j = lo, mid
            while i < mid and j < hi:
                if a[j] < a[i]:
                    b.append(a[j]); j += 1
                else:
                    b.append(a[i]); i += 1
            b += a[i:mid] + a[j:hi]
        a, width = b, 2 * width
    return a


def quick(a):
    """Lomuto, last element as pivot, explicit stack (the source's)."""
    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo < hi:
            i = lo
            for j in range(lo, hi):
                if a[j] < a[hi]:
                    a[i], a[j] = a[j], a[i]
                    i += 1
            a[i], a[hi] = a[hi], a[i]
            stack += [(lo, i - 1), (i + 1, hi)]
    return a


def heap(a):
    def sift(root, end):
        while (c := 2 * root + 1) < end:
            if c + 1 < end and a[c] < a[c + 1]:
                c += 1
            if not a[root] < a[c]:
                return
            a[root], a[c] = a[c], a[root]
            root = c
    n = len(a)
    for i in reversed(range(n // 2)):
        sift(i, n)
    for end in reversed(range(1, n)):
        a[0], a[end] = a[end], a[0]
        sift(0, end)
    return a


SORTS = [("insertion", insertion), ("merge (bottom-up)", merge),
         ("quicksort (Lomuto)", quick), ("heapsort", heap),
         ("sorted() Timsort", sorted)]


def decision_tree_n3():
    """Every order of 3 items and the compares insertion sort asks."""
    for p in sorted(set(itertools.permutations("abc"))):
        rank = {"a": 0, "b": 1, "c": 2}
        vals = [rank[ch] for ch in p]
        print("  input order", "".join(p), "->",
              compares(insertion, vals), "compares")


if __name__ == "__main__":
    a = [38, 27, 43, 3, 9, 82, 10]
    lb = lower_bound(len(a))
    print(f"n=7: 7! = {math.factorial(7):,} orderings,"
          f" log2(7!) = {lb:.2f} -> at least {math.ceil(lb)}")
    for name, s in SORTS:
        print(f"  {name:20} compares on a: {compares(s, a)}")
        worst = max(compares(s, list(p)) for p in
                    itertools.permutations(a))
        print(f"  {'':20} worst over all 5,040 orders: {worst}")

    print("n=3: 3! = 6 leaves, log2 6 =",
          f"{math.log2(6):.2f} -> at least 3")
    decision_tree_n3()

    for n in (3, 7, 10, 100, 10**4, 10**6):
        lb = lower_bound(n)
        print(f"n={n:>9,}: log2(n!) = {lb:>14,.1f}"
              f"   n*log2(n) = {n * math.log2(n):>14,.1f}"
              f"   ratio = {lb / (n * math.log2(n)):.3f}")

    random.seed(0)
    n = 10**5
    r = [random.random() for _ in range(n)]
    print(f"sorted() on {n:,} random floats: {compares(sorted, r):,}"
          f" compares; bound {lower_bound(n):,.0f}")
    print(f"sorted() on {n:,} already sorted: "
          f"{compares(sorted, sorted(r)):,} compares (n - 1)")

    # Stability: sort by the secondary key first, then the primary.
    people = [("Di", "dev"), ("Ana", "ops"), ("Ed", "ops"),
              ("Bo", "dev"), ("Cy", "ops")]
    by_name = sorted(people, key=lambda p: p[0])
    by_team = sorted(by_name, key=lambda p: p[1])
    print("by team, then name:", by_team)
    class ByTeam:
        def __init__(self, p):
            self.p = p

        def __lt__(self, other):
            return self.p[1] < other.p[1]
    for name, s in (("heapsort", heap), ("quicksort", quick)):
        out = [k.p for k in s([ByTeam(p) for p in by_name])]
        print(f"{name} (unstable) by team:", out)
    print("one pass, tuple key:",
          sorted(people, key=lambda p: (p[1], p[0])) == by_team)
