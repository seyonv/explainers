"""Median of two sorted arrays by binary search on the cut.

Take i items from the shorter array a and j = k - i from b, where
k = (m + n - 1) // 2 is the size of the left half. Binary search for
the smallest i with a[i] >= b[j - 1]; the median is then the smallest
(odd total) or the mean of the two smallest (even total) items right
of the cut.
"""
import bisect
import random


def median_two(a, b):
    if len(a) > len(b):
        a, b = b, a                  # search the shorter array
    m, n = len(a), len(b)
    k = (m + n - 1) // 2             # items left of the cut
    lo, hi = 0, min(m, k)
    while lo < hi:
        i = (lo + hi) // 2           # take i from a, k - i from b
        if a[i] >= b[k - 1 - i]:     # enough from a: go left
            hi = i
        else:                        # b's last left item too big
            lo = i + 1
    i, j = lo, k - lo
    right = sorted(a[i:i + 2] + b[j:j + 2])   # 2 smallest are here
    if (m + n) % 2:
        return right[0]
    return (right[0] + right[1]) / 2


def source_median(nums1, nums2):
    """The source guide's version (needs Python 3.10 for key=)."""
    nums1, nums2 = sorted((nums1, nums2), key=len)
    m, n = len(nums1), len(nums2)
    j = (m + n - 1) // 2 - 1
    i = bisect.bisect_left(
        range(m), True,
        key=lambda x: x > j or nums1[x] >= nums2[j - x])
    j -= i
    median = sorted(nums1[i:i + 2] + nums2[j + 1:j + 3])
    return (median[0] + median[(m + n) & 1 ^ 1]) / 2


def trace(a, b):
    """Print each probe with its four boundary values."""
    inf = float("inf")

    def at(xs, t):
        return -inf if t < 0 else inf if t >= len(xs) else xs[t]

    m, n = len(a), len(b)
    k = (m + n - 1) // 2
    lo, hi = 0, min(m, k)
    print(f"k = ({m} + {n} - 1) // 2 = {k}")
    while lo < hi:
        i = (lo + hi) // 2
        j = k - i
        ok = a[i] >= b[j - 1]
        print(f"lo={lo} hi={hi} i={i} j={j}  "
              f"A[i-1]={at(a, i - 1)} A[i]={at(a, i)}  "
              f"B[j-1]={at(b, j - 1)} B[j]={at(b, j)}  "
              f"A[i]>=B[j-1]? {ok} -> "
              + ("hi = i" if ok else "lo = i + 1"))
        if ok:
            hi = i
        else:
            lo = i + 1
    i, j = lo, k - lo
    print(f"cut i={i} j={j}: left = A{a[:i]} + B{b[:j]}, "
          f"right starts min({at(a, i)}, {at(b, j)})")


def brute(a, b):
    s = sorted(a + b)
    t = len(s)
    return s[t // 2] if t % 2 else (s[t // 2 - 1] + s[t // 2]) / 2


def probes(width):
    """Most loop iterations when lo..hi starts `width` wide."""
    return width.bit_length()


if __name__ == "__main__":
    A = [1, 3, 8, 9, 15]
    B = [7, 11, 18, 19, 21, 25]
    trace(A, B)
    print("median_two:", median_two(A, B))
    print("source:    ", source_median(A, B))
    print("merged:    ", sorted(A + B))
    print("even case: ", median_two([1, 2], [3, 4]),
          source_median([1, 2], [3, 4]))

    rng = random.Random(0)
    for _ in range(10_000):
        a = sorted(rng.randint(-20, 20)
                   for _ in range(rng.randint(0, 8)))
        b = sorted(rng.randint(-20, 20)
                   for _ in range(rng.randint(1, 8)))
        want = brute(a, b)
        assert median_two(a, b) == want, (a, b)
        assert source_median(a, b) == want, (a, b)
    print("10,000 random pairs agree with sorted(a + b)")

    for m in (5, 1_000, 10 ** 6):
        k = (2 * m - 1) // 2             # m = n
        print(f"m = n = {m:,}: <= {probes(min(m, k))} probes vs "
              f"{k + 2:,} merge steps to reach the middle")
