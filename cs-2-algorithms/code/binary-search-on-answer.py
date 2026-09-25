"""Binary search on the answer.

One template, first_true, finds the smallest x in [lo, hi] for which a
monotone predicate ok(x) is True (False ... False True ... True).
The source guide's two C++ binary searches, in Python:
  - k-th smallest pair distance: search the distance d itself
  - minimum of a rotated sorted array that may contain duplicates
"""
import bisect
import itertools
import random


def first_true(lo, hi, ok):
    """Smallest x in [lo, hi] with ok(x). Assumes ok(hi) is True."""
    while lo < hi:
        mid = (lo + hi) // 2      # floor: mid < hi, so hi = mid shrinks
        if ok(mid):
            hi = mid              # mid works: answer is mid or left
        else:
            lo = mid + 1          # mid fails: answer is right of mid
    return lo


def pairs_within(a, d):
    """Pairs i < j of sorted a with a[j] - a[i] <= d. Two pointers."""
    count = j = 0
    for i in range(len(a)):
        while j < len(a) and a[j] <= a[i] + d:
            j += 1
        count += j - i - 1        # partners of i: i+1 .. j-1
    return count


def smallest_distance_pair(nums, k):
    a = sorted(nums)
    return first_true(0, a[-1] - a[0],
                      lambda d: pairs_within(a, d) >= k)


def find_min(nums):
    """Min of a rotated sorted array, duplicates allowed."""
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1          # drop in (mid, hi]: min is right
        elif nums[mid] < nums[hi]:
            hi = mid              # mid..hi ascends: min is mid or left
        else:
            hi -= 1               # tie: can't tell, shrink by one
    return nums[lo]


# ---------- traces and checks for the card ----------

def trace_pair(nums, k):
    a = sorted(nums)
    lo, hi = 0, a[-1] - a[0]
    print(f"nums={nums} k={k} sorted={a} lo={lo} hi={hi}")
    while lo < hi:
        mid = (lo + hi) // 2
        c = pairs_within(a, mid)
        ok = c >= k
        new = (lo, mid) if ok else (mid + 1, hi)
        print(f"  lo={lo} hi={hi} mid={mid} count={c} "
              f"ok={ok} -> [{new[0]}, {new[1]}]")
        lo, hi = new
    print(f"  answer {lo}")
    print("  count(d) for every d:",
          [pairs_within(a, d) for d in range(a[-1] - a[0] + 1)])


def trace_min(nums):
    lo, hi = 0, len(nums) - 1
    steps = 0
    print(f"nums={nums}")
    while lo < hi:
        mid = (lo + hi) // 2
        steps += 1
        m, h = nums[mid], nums[hi]
        if m > h:
            rule, lo = f"{m} > {h}: lo = mid + 1", mid + 1
        elif m < h:
            rule, hi = f"{m} < {h}: hi = mid", mid
        else:
            rule, hi = f"{m} = {h}: hi -= 1", hi - 1
        print(f"  step {steps}: mid={mid} {rule} -> [{lo}, {hi}]")
    print(f"  answer {nums[lo]} in {steps} steps")
    return steps


def min_steps(nums):
    lo, hi, steps = 0, len(nums) - 1, 0
    while lo < hi:
        mid = (lo + hi) // 2
        steps += 1
        if nums[mid] > nums[hi]:
            lo = mid + 1
        elif nums[mid] < nums[hi]:
            hi = mid
        else:
            hi -= 1
    return steps


def brute_pair(nums, k):
    ds = sorted(abs(x - y) for x, y in itertools.combinations(nums, 2))
    return ds[k - 1]


if __name__ == "__main__":
    print(smallest_distance_pair([1, 3, 1], 1))   # 0
    print(smallest_distance_pair([1, 6, 1], 3))   # 5
    print(find_min([2, 2, 2, 0, 1]))              # 0
    print()
    trace_pair([1, 3, 1], 1)
    trace_pair([1, 6, 1], 3)
    print()
    trace_min([2, 2, 2, 0, 1])
    trace_min([1, 1, 1, 0, 1])
    trace_min([1, 0, 1, 1, 1])
    print()
    n = 1000
    worst = max(min_steps(list(range(r, n)) + list(range(r)))
                for r in range(n))
    print(f"n={n}: all-equal {min_steps([7] * n)} steps, "
          f"distinct {worst} steps (worst rotation)")

    # bisect with key (Python 3.10+) is the same first_true
    a = sorted([1, 6, 1])
    d = bisect.bisect_left(range(a[-1] - a[0] + 1), True,
                           key=lambda d: pairs_within(a, d) >= 3)
    print("bisect_left with key:", d)

    # the classic bug: lo = mid with floor mid never ends at hi = lo + 1
    lo, hi, loops = 0, 1, 0
    while lo < hi and loops < 5:
        mid = (lo + hi) // 2
        lo = mid                  # should be mid + 1
        loops += 1
    print(f"lo = mid bug: still lo={lo} hi={hi} after {loops} loops")

    rng = random.Random(1)
    for _ in range(1000):
        nums = [rng.randint(0, 30) for _ in range(rng.randint(2, 12))]
        k = rng.randint(1, len(nums) * (len(nums) - 1) // 2)
        assert smallest_distance_pair(nums, k) == brute_pair(nums, k)
        s = sorted(rng.randint(0, 5) for _ in range(rng.randint(1, 12)))
        r = rng.randrange(len(s))
        assert find_min(s[r:] + s[:r]) == min(s)
    print("1,000 random cases each match brute force")
