"""Quickselect: the k-th smallest element (k = 1 is the minimum).

Same three-way split as the cheat-sheet's version (smaller / equal /
larger than the pivot), but written as a loop, because the recursive
version with pivot arr[0] hits Python's recursion limit (1,000) on
sorted input of about 1,000 elements.
"""
import heapq
import math
import random
from fractions import Fraction


def quickselect(arr, k, pick=random.choice):
    while True:
        p = pick(arr)
        smaller = [x for x in arr if x < p]
        larger = [x for x in arr if x > p]
        n = len(arr) - len(larger)        # how many are <= p
        if k <= len(smaller):
            arr = smaller                 # answer is left of p
        elif k > n:
            arr, k = larger, k - n        # skip n items on the left
        else:
            return p                      # p itself is k-th


def first(a):
    return a[0]


def source_quickselect(arr, k):
    """The cheat-sheet's recursive version, pivot arr[0]."""
    smaller = [x for x in arr if x < arr[0]]
    larger = [x for x in arr if x > arr[0]]
    n = len(arr) - len(larger)
    if k <= len(smaller):
        return source_quickselect(smaller, k)
    elif k > n:
        return source_quickselect(larger, k - n)
    return arr[0]


def compares(arr, k, pick):
    """Compares against the pivot: m - 1 on a level of m items."""
    total = 0
    while True:
        total += len(arr) - 1
        p = pick(arr)
        smaller = [x for x in arr if x < p]
        larger = [x for x in arr if x > p]
        n = len(arr) - len(larger)
        if k <= len(smaller):
            arr = smaller
        elif k > n:
            arr, k = larger, k - n
        else:
            return total


def two_way_compares(arr, k):
    """Split into < p and >= p: equal keys all land on one side."""
    total = 0
    while True:
        total += len(arr) - 1
        p, rest = arr[0], arr[1:]
        left = [x for x in rest if x < p]
        right = [x for x in rest if x >= p]
        if k <= len(left):
            arr = left
        elif k == len(left) + 1:
            return total
        else:
            arr, k = right, k - len(left) - 1


def expected_exact(n_max):
    """E[compares] with a random pivot on distinct keys, exactly."""
    C = [[Fraction(0)] * (n_max + 1) for _ in range(n_max + 1)]
    for n in range(1, n_max + 1):
        for k in range(1, n + 1):
            s = Fraction()
            s += sum(C[p - 1][k] for p in range(k + 1, n + 1))
            s += sum(C[n - p][k - p] for p in range(1, k))
            C[n][k] = n - 1 + s / n
    return C


def knuth(n, k):
    """Knuth (1971): closed form of the same expectation."""
    def H(m):
        return sum(Fraction(1, i) for i in range(1, m + 1))
    return 2 * ((n + 1) * H(n) - (n + 3 - k) * H(n + 1 - k)
                - (k + 2) * H(k) + n + 3)


def trace(arr, k):
    total = 0
    while True:
        p = arr[0]
        smaller = [x for x in arr if x < p]
        larger = [x for x in arr if x > p]
        n = len(arr) - len(larger)
        total += len(arr) - 1
        if k <= len(smaller):
            step, nxt = f"k={k} <= {len(smaller)}: go left", smaller
        elif k > n:
            step, nxt = f"k={k} > {n}: go right, k={k - n}", larger
        else:
            step, nxt = f"{len(smaller)} < k <= {n}: answer {p}", None
        print(f"{str(arr):<28} pivot {p:<3} {smaller} | {larger}"
              f"  {step}")
        if nxt is None:
            print("compares against the pivot:", total)
            return p
        if k > n:
            k -= n
        arr = nxt


if __name__ == "__main__":
    a = [38, 27, 43, 3, 9, 82, 10]
    trace(a, 3)
    print("quickselect(a, 3) =", quickselect(a, 3))
    print("heapq.nsmallest(3, a)[-1] =", heapq.nsmallest(3, a)[-1])

    print("sorted a, k=7, pivot a[0]:",
          compares(sorted(a), 7, first), "compares (= 7*6/2)")
    try:
        source_quickselect(list(range(1000)), 1000)
    except RecursionError:
        print("source version, range(1000), k=1000: RecursionError")
    print("loop version:", quickselect(list(range(1000)), 1000, first))

    rng = random.Random(1)
    for _ in range(1000):
        xs = [rng.randint(-20, 20) for _ in range(rng.randint(1, 40))]
        k = rng.randint(1, len(xs))
        want = sorted(xs)[k - 1]
        assert quickselect(xs, k) == want == source_quickselect(xs, k)
    print("1,000 random lists with duplicates: all match sorted()")

    ones = [5] * 1000
    print("[5]*1000, k=1000: three-way", compares(ones, 1000, first),
          "compares, two-way", two_way_compares(ones, 1000))

    C = expected_exact(40)
    assert all(C[n][k] == knuth(n, k)
               for n in range(1, 41) for k in range(1, n + 1))
    print("Knuth's formula = exact recurrence for all n <= 40")
    print(f"n=7, k=3 expected: {float(C[7][3]):.2f} compares")
    med = float(knuth(10_000, 5_000)) / 1e4
    print(f"median, n=10,000: {med:.2f} n;"
          f" limit 2(1+ln 2) = {2 * (1 + math.log(2)):.2f} n")

    random.seed(7)
    n, runs = 10_000, 200
    xs = random.sample(range(10 * n), n)
    avg = sum(compares(xs, n // 2, random.choice)
              for _ in range(runs)) / runs
    print(f"median of n={n:,}, random pivot, {runs} runs: "
          f"{avg / n:.2f} n compares (theory 3.38 n)")
