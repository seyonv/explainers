"""Insertion sort, swap-based like the cheat-sheet's C++ version.

Each swap fixes exactly one inversion (a pair i < j with a[i] > a[j]),
so the number of swaps equals the number of inversions.
"""
import bisect
import random


def insertion_sort(a):
    a = list(a)
    swaps = 0
    for i in range(1, len(a)):
        j = i
        while j > 0 and a[j - 1] > a[j]:   # left neighbour too big
            a[j - 1], a[j] = a[j], a[j - 1]
            j -= 1                         # decrement after the swap
            swaps += 1
    return a, swaps


def inversions(a):
    n = len(a)
    return sum(a[i] > a[j] for i in range(n) for j in range(i + 1, n))


def binary_insertion_sort(a):
    """What Timsort does on short runs: binary search, then shift."""
    out = []
    for x in a:
        bisect.insort_right(out, x)       # stable: after equal keys
    return out


def trace(a):
    a = list(a)
    print(f"start      {a}")
    for i in range(1, len(a)):
        key, j, s = a[i], i, 0
        while j > 0 and a[j - 1] > a[j]:
            a[j - 1], a[j] = a[j], a[j - 1]
            j -= 1
            s += 1
        print(f"i={i} key={key:<3} swaps={s}  {a}")


if __name__ == "__main__":
    a = [38, 27, 43, 3, 9, 82, 10]
    trace(a)
    out, swaps = insertion_sort(a)
    print("sorted:", out)
    print("swaps:", swaps, " inversions:", inversions(a))
    n = len(a)
    print("worst case n(n-1)/2 =", n * (n - 1) // 2)
    print("sorted input swaps:", insertion_sort(sorted(a))[1])
    rng = random.Random(1)
    for _ in range(1000):
        xs = [rng.randint(-50, 50) for _ in range(rng.randint(0, 30))]
        got, s = insertion_sort(xs)
        assert got == sorted(xs) == binary_insertion_sort(xs)
        assert s == inversions(xs)
    print("1,000 random lists: sorted and swaps == inversions")
