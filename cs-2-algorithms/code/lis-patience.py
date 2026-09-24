"""Longest increasing subsequence by patience sorting (tails array),
and Russian Doll Envelopes reduced to it. Python 3.10."""
import bisect
import itertools
import random


def lis_length(nums):
    tails = []  # tails[k]: smallest tail of any run of length k+1
    for x in nums:
        i = bisect.bisect_left(tails, x)  # first tail >= x
        if i == len(tails):
            tails.append(x)       # x extends the longest run
        else:
            tails[i] = x          # x is a smaller tail for length i+1
    return len(tails)


def max_envelopes(envelopes):
    # width up; height DOWN inside a width, so equal widths can't chain
    order = sorted(envelopes, key=lambda e: (e[0], -e[1]))
    return lis_length([h for _, h in order])


def lis_sequence(nums):
    """One actual LIS: remember each tail's index and its parent."""
    tails, tail_idx, parent = [], [], [-1] * len(nums)
    for j, x in enumerate(nums):
        i = bisect.bisect_left(tails, x)
        parent[j] = tail_idx[i - 1] if i else -1
        if i == len(tails):
            tails.append(x)
            tail_idx.append(j)
        else:
            tails[i], tail_idx[i] = x, j
    out, j = [], tail_idx[-1] if tail_idx else -1
    while j != -1:
        out.append(nums[j])
        j = parent[j]
    return out[::-1]


def lis_dp(nums):
    """O(n^2) reference: best[j] = longest run ending at j."""
    best = []
    for j, x in enumerate(nums):
        best.append(1 + max((best[i] for i in range(j)
                             if nums[i] < x), default=0))
    return max(best, default=0)


def trace(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        what = "append" if i == len(tails) else f"replace {tails[i]}"
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
        print(f"x={x:>3}  i={i}  {what:<12} tails={tails}")


def brute_envelopes(env):
    best = 0
    for r in range(1, len(env) + 1):
        for combo in itertools.permutations(env, r):
            if all(a[0] < b[0] and a[1] < b[1]
                   for a, b in zip(combo, combo[1:])):
                best = max(best, r)
    return best


if __name__ == "__main__":
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    trace(nums)
    print("LIS length:", lis_length(nums))            # 4
    print("one LIS:", lis_sequence(nums))
    print("tails is not an LIS:", [3, 4, 1], "->",
          "tails [1, 4], length", lis_length([3, 4, 1]))

    env = [[5, 4], [6, 4], [6, 7], [2, 3]]
    order = sorted(env, key=lambda e: (e[0], -e[1]))
    print("sorted (w, -h):", order)
    trace([h for _, h in order])
    print("max envelopes:", max_envelopes(env))       # 3

    same_w = [[1, 1], [1, 2], [1, 3]]
    wrong = sorted(same_w)                            # h ascending
    print("h ascending:", lis_length([h for _, h in wrong]),
          "| h descending:", max_envelopes(same_w), "| truth: 1")

    rng = random.Random(1)
    for _ in range(1000):
        a = [rng.randrange(20) for _ in range(rng.randrange(12))]
        assert lis_length(a) == lis_dp(a) == len(lis_sequence(a))
        s = lis_sequence(a)
        assert all(p < q for p, q in zip(s, s[1:]))
    for _ in range(300):
        e = [[rng.randrange(1, 6), rng.randrange(1, 6)]
             for _ in range(rng.randrange(1, 6))]
        assert max_envelopes(e) == brute_envelopes(e)
    print("1,000 random lists and 300 envelope sets"
          " agree with brute force")
