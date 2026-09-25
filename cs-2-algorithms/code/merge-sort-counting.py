"""Counting during a merge: Count Smaller and Reverse Pairs.

Bottom-up merge sort (widths 1, 2, 4, ...) as in the source guide,
rewritten in Python. Plus the two usual alternatives: bisect.insort
(stdlib, O(n) per insert) and a Fenwick tree, checked against brute
force on random lists.
"""
import bisect
import heapq
import random


def count_smaller(nums, trace=False):
    n = len(nums)
    idx = list(range(n))           # indices, sorted by value per block
    counts = [0] * n
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid, hi = min(lo + width, n), min(lo + 2 * width, n)
            left, right = idx[lo:mid], idx[mid:hi]
            out, j = [], 0
            for i in left:
                while j < len(right) and nums[right[j]] < nums[i]:
                    out.append(right[j])   # smaller one jumps ahead
                    j += 1
                counts[i] += j             # j smaller ones from right
                out.append(i)
            idx[lo:hi] = out + right[j:]
            if trace and right:
                print(f"width {width}:",
                      [nums[k] for k in left], "+",
                      [nums[k] for k in right], "->",
                      [nums[k] for k in idx[lo:hi]],
                      "counts", counts)
        width *= 2
    return counts


def reverse_pairs(nums, trace=False):
    a, n, total = list(nums), len(nums), 0
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid, hi = min(lo + width, n), min(lo + 2 * width, n)
            j, found = mid, 0
            for i in range(lo, mid):       # count pass: both sorted
                while j < hi and a[i] > 2 * a[j]:
                    j += 1
                found += j - mid
            total += found
            if trace and mid < hi:
                print(f"width {width}:", a[lo:mid], "+", a[mid:hi],
                      "found", found)
            a[lo:hi] = heapq.merge(a[lo:mid], a[mid:hi])
        width *= 2
    return total


def count_smaller_insort(nums):
    seen, counts = [], [0] * len(nums)
    for i in reversed(range(len(nums))):
        counts[i] = bisect.bisect_left(seen, nums[i])
        bisect.insort(seen, nums[i])   # O(n) shift per insert
    return counts


def count_smaller_fenwick(nums):
    rank = {v: r for r, v in enumerate(sorted(set(nums)), 1)}
    tree, counts = [0] * (len(rank) + 1), [0] * len(nums)
    for i in reversed(range(len(nums))):
        r = rank[nums[i]] - 1          # count ranks < this one
        while r > 0:
            counts[i] += tree[r]
            r -= r & -r
        r = rank[nums[i]]
        while r < len(tree):
            tree[r] += 1
            r += r & -r
    return counts


def brute_smaller(nums):
    return [sum(y < x for y in nums[i + 1:])
            for i, x in enumerate(nums)]


def brute_reverse(nums):
    n = len(nums)
    return sum(nums[i] > 2 * nums[j]
               for i in range(n) for j in range(i + 1, n))


if __name__ == "__main__":
    print(count_smaller([5, 2, 6, 1], trace=True))
    print(reverse_pairs([1, 3, 2, 3, 1], trace=True))
    a = [38, 27, 43, 3, 9, 82, 10]
    cs = count_smaller(a)
    print("a:", cs, "inversions", sum(cs),
          "reverse pairs", reverse_pairs(a))
    rng = random.Random(1)
    for _ in range(1000):
        xs = [rng.randint(-20, 20) for _ in range(rng.randint(0, 30))]
        want = brute_smaller(xs)
        assert count_smaller(xs) == want
        assert count_smaller_insort(xs) == want
        assert count_smaller_fenwick(xs) == want
        assert reverse_pairs(xs) == brute_reverse(xs)
    print("1,000 random lists: all four agree with brute force")
