"""Greedy II: Candy and Patching Array.

Candy: two passes. Left to right, a child rated above its left
neighbour gets one more than that neighbour; right to left, the same
against the right neighbour. Each child takes the larger of the two.
The cheat-sheet's C++ counts up-runs and down-runs instead; the
run-counting version below is a Python rewrite and gives the same sum.

Patching Array: keep `miss`, the smallest sum not yet formable, so
every value in [1, miss) is formable. Use nums[i] if it is <= miss,
otherwise patch in `miss` itself, which doubles the reach.
"""
import itertools
import random


def candy(ratings):
    n = len(ratings)
    left = [1] * n
    for i in range(1, n):                    # beat the left neighbour
        if ratings[i] > ratings[i - 1]:
            left[i] = left[i - 1] + 1
    right = [1] * n
    for i in range(n - 2, -1, -1):           # beat the right neighbour
        if ratings[i] > ratings[i + 1]:
            right[i] = right[i + 1] + 1
    return sum(max(a, b) for a, b in zip(left, right))


def candy_passes(ratings):
    n = len(ratings)
    left, right = [1] * n, [1] * n
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            left[i] = left[i - 1] + 1
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            right[i] = right[i + 1] + 1
    return left, right, [max(a, b) for a, b in zip(left, right)]


def candy_runs(ratings):
    """The source's up/down run counting, rewritten in Python."""
    n, i, total = len(ratings), 1, len(ratings)
    while i < n:
        if ratings[i] == ratings[i - 1]:
            i += 1
            continue
        up = down = 0
        while i < n and ratings[i] > ratings[i - 1]:
            up += 1
            total += up                      # 1 + 2 + ... + up
            i += 1
        while i < n and ratings[i] < ratings[i - 1]:
            down += 1
            total += down                    # 1 + 2 + ... + down
            i += 1
        total -= min(up, down)           # peak counted twice
    return total


def candy_brute(ratings, cap=6):
    """Smallest valid assignment by trying every one up to `cap`."""
    n, best = len(ratings), None
    for c in itertools.product(range(1, cap + 1), repeat=n):
        ok = all(
            (ratings[i] <= ratings[i - 1] or c[i] > c[i - 1])
            and (ratings[i - 1] <= ratings[i] or c[i - 1] > c[i])
            for i in range(1, n))
        if ok and (best is None or sum(c) < best):
            best = sum(c)
    return best


def min_patches(nums, n):
    miss, i, patches = 1, 0, 0               # [1, miss) is formable
    while miss <= n:
        if i < len(nums) and nums[i] <= miss:
            miss += nums[i]                  # use it: reach grows
            i += 1
        else:
            miss += miss                     # patch miss: reach doubles
            patches += 1
    return patches


def patches_trace(nums, n):
    miss, i, patches = 1, 0, 0
    while miss <= n:
        if i < len(nums) and nums[i] <= miss:
            step = f"use nums[{i}] = {nums[i]}"
            miss, i = miss + nums[i], i + 1
        else:
            step = f"patch {miss}"
            miss, patches = miss + miss, patches + 1
        print(f"  {step:<16} -> covers [1, {miss - 1}]")
    return patches


def patches_brute(nums, n):
    """Fewest patches from 1..n such that all of 1..n are sums."""
    def covers(xs):
        sums = {0}
        for x in xs:
            sums |= {s + x for s in sums}
        return all(v in sums for v in range(1, n + 1))
    for k in range(n + 1):
        for extra in itertools.combinations_with_replacement(
                range(1, n + 1), k):
            if covers(list(nums) + list(extra)):
                return k


if __name__ == "__main__":
    for r in ([1, 0, 2], [1, 2, 87, 87, 87, 2, 1], [1, 2, 3, 2, 1]):
        left, right, c = candy_passes(r)
        print(f"ratings {r}")
        print(f"  left  {left}\n  right {right}\n  max   {c}")
        print(f"  two passes {candy(r)}, source runs {candy_runs(r)}")
    rng = random.Random(0)
    for _ in range(1000):
        r = [rng.randint(0, 4) for _ in range(rng.randint(0, 9))]
        assert candy(r) == candy_runs(r)
    print("1,000 random ratings: two passes == source's run count")
    for _ in range(200):
        r = [rng.randint(0, 3) for _ in range(rng.randint(1, 5))]
        assert candy(r) == candy_brute(r)
    print("200 random ratings (n <= 5) match exhaustive search")

    print("nums [1, 5, 10], n = 20")
    print("  patches:", patches_trace([1, 5, 10], 20))
    for nums, n in (([1, 3], 6), ([1, 2, 2], 5), ([], 7),
                    ([], 2**31 - 1)):
        print(f"nums {nums}, n = {n} -> {min_patches(nums, n)}")
    for _ in range(300):
        n = rng.randint(1, 12)
        k = rng.randint(0, 4)
        nums = sorted(rng.randint(1, n) for _ in range(k))
        assert min_patches(nums, n) == patches_brute(nums, n)
    print("300 random cases (n <= 12) match exhaustive search")
