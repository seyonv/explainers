"""Hashing and buckets: a chained hash set, then Contains Duplicate III.

Source: ljeng/cheat-sheet, coding-algorithms/sorting.md, "Contains
Duplicate" (LeetCode 220, Contains Duplicate III). Same bucket idea;
this copy tests `b in seen`, so neighbour checks add no placeholder
keys.
"""
import sys


def toy_buckets(keys, m=8):
    """Chained hash set with m buckets: bucket = hash(key) % m."""
    buckets = [[] for _ in range(m)]
    for k in keys:
        b = buckets[hash(k) % m]
        if k not in b:           # only this one bucket is searched
            b.append(k)
    return buckets


def contains_near_dup(nums, index_diff, value_diff):
    w = value_diff + 1           # bucket width: same bucket => close
    seen = {}                    # bucket id -> the one value in it
    for i, x in enumerate(nums):
        b = x // w
        if b in seen:
            return True          # same bucket: |x - y| <= value_diff
        for nb in (b - 1, b + 1):   # neighbours may also be close
            if nb in seen and abs(x - seen[nb]) <= value_diff:
                return True
        seen[b] = x
        if i >= index_diff:      # drop the value leaving the window
            del seen[nums[i - index_diff] // w]
    return False


def trace(nums, index_diff, value_diff):
    """Print the bucket map and each neighbour check, step by step."""
    w = value_diff + 1
    seen = {}
    for i, x in enumerate(nums):
        b = x // w
        checks = [f"b{nb}:|{x}-{seen[nb]}|={abs(x - seen[nb])}"
                  for nb in (b, b - 1, b + 1) if nb in seen]
        hit = b in seen or any(
            abs(x - seen[nb]) <= value_diff
            for nb in (b - 1, b + 1) if nb in seen)
        seen[b] = x
        gone = ""
        if i >= index_diff:
            old = nums[i - index_diff]
            del seen[old // w]
            gone = f"drop {old} (i={i - index_diff})"
        print(f"i={i} x={x} b={b}  {', '.join(checks) or '-':<22}"
              f" {gone:<16} buckets={dict(sorted(seen.items()))}")
        if hit:
            return True
    return False


def source_version(nums, indexDiff, valueDiff):
    """The cheat-sheet's code, verbatim apart from line wrapping."""
    import collections
    valueDiff += 1
    buckets = collections.defaultdict(lambda: sys.maxsize)
    for i, num in enumerate(nums):
        k = num // valueDiff
        if buckets[k] < sys.maxsize or min(
                abs(num - buckets[k - 1]),
                abs(num - buckets[k + 1])) < valueDiff:
            return True
        buckets[k] = num
        if i >= indexDiff:
            del buckets[nums[i - indexDiff] // valueDiff]
    return False, len(buckets)


def brute_force(nums, index_diff, value_diff):
    n = len(nums)
    return any(abs(nums[i] - nums[j]) <= value_diff
               for i in range(n)
               for j in range(i + 1, min(n, i + index_diff + 1)))


if __name__ == "__main__":
    import random

    a = [5, 2, 9, 1, 5, 6]
    for i, b in enumerate(toy_buckets(a)):
        print(f"bucket {i}: {b}")
    print("load factor", len(set(a)), "/ 8 =", len(set(a)) / 8)

    nums = [1, 5, 9, 1, 5, 9]
    print(contains_near_dup(nums, 2, 3))        # False
    trace(nums, 2, 3)
    print(contains_near_dup(nums, 3, 3))        # True: 1 and 1
    print("source version:", source_version(nums, 2, 3))

    # dict index sizes: 8 slots hold 5 keys, the 6th triggers a resize
    sizes, last = [], None
    for n in range(50):
        s = sys.getsizeof({i: i for i in range(n)})
        if s != last:
            sizes.append(n)
            last = s
    print("dict grows at", sizes[2:], "keys")

    rng = random.Random(0)
    for _ in range(1000):
        xs = [rng.randint(-20, 20) for _ in range(rng.randint(0, 12))]
        k, t = rng.randint(0, 5), rng.randint(0, 6)
        want = brute_force(xs, k, t)
        assert contains_near_dup(xs, k, t) == want
        r = source_version(xs, k, t)
        assert (r is True) == want      # the source is correct too
    print("1000 random cases match brute force (both versions)")
