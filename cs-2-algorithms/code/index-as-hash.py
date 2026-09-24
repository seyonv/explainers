"""First Missing Positive: the array as its own hash table.

Cyclic placement, as in the cheat-sheet's C++ version: send every
value v in 1..n to its home slot v - 1 by swapping, then scan for the
first slot that does not hold its own number. O(n) time, O(1) extra.
"""
import random


def first_missing_positive(nums):
    n = len(nums)
    for i in range(n):
        # v in range and its home slot doesn't hold v yet: send it home
        while 0 < nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            j = nums[i] - 1              # read the home index first
            nums[i], nums[j] = nums[j], nums[i]
    for i in range(n):
        if nums[i] != i + 1:             # slot i should hold i + 1
            return i + 1
    return n + 1                         # 1..n all present


def trace(nums):
    """Print every swap, return (answer, swaps, condition checks)."""
    nums = list(nums)
    n, swaps, checks = len(nums), 0, 0
    print(f"start          {nums}")
    for i in range(n):
        while True:
            checks += 1
            v = nums[i]
            if not (0 < v <= n and nums[v - 1] != v):
                break
            j = v - 1
            nums[i], nums[j] = nums[j], nums[i]
            swaps += 1
            print(f"i={i} send {v} to slot {j}  {nums}")
    ans = next((i + 1 for i in range(n) if nums[i] != i + 1), n + 1)
    print(f"scan: first slot with the wrong value -> answer {ans}")
    return ans, swaps, checks


def brute(nums):
    s = set(nums)
    k = 1
    while k in s:
        k += 1
    return k


def swaps_only(nums):
    nums, n, swaps = list(nums), len(nums), 0
    for i in range(n):
        while 0 < nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            j = nums[i] - 1
            nums[i], nums[j] = nums[j], nums[i]
            swaps += 1
    return swaps


if __name__ == "__main__":
    ans, swaps, checks = trace([3, 4, -1, 1])
    print(f"swaps={swaps}  while-checks={checks}  (n=4)")
    for xs in ([1, 2, 0], [7, 8, 9, 11, 12], [1, 1], [1, 2, 3]):
        print(xs, "->", first_missing_positive(list(xs)))
    rng = random.Random(0)
    for _ in range(1000):
        xs = [rng.randint(-5, 12) for _ in range(rng.randint(0, 10))]
        assert first_missing_positive(list(xs)) == brute(xs)
    print("1,000 random lists agree with the set-based brute force")
    n = 10**5
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    print(f"shuffled 1..{n:,}: {swaps_only(perm):,} swaps (< n)")
