"""Monotonic deque: Sliding Window Maximum.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md,
Queues > Sliding Window Maximum (C++). Rewritten in Python with
collections.deque; same algorithm.
"""
from collections import deque


def max_sliding_window(nums, k):
    dq = deque()                 # indices; their values never rise
    out = []
    for i, x in enumerate(nums):
        if dq and dq[0] <= i - k:        # front fell out of window
            dq.popleft()
        while dq and nums[dq[-1]] < x:   # x beats them: never max
            dq.pop()
        dq.append(i)
        if i >= k - 1:                   # window [i-k+1, i] is full
            out.append(nums[dq[0]])      # front is the max
    return out


def trace(nums, k):
    """Print the deque after every step and what left it."""
    dq, out, pushes, pops = deque(), [], 0, 0
    for i, x in enumerate(nums):
        left = []
        if dq and dq[0] <= i - k:
            left.append(f"expire {nums[dq[0]]}@{dq[0]}")
            dq.popleft()
            pops += 1
        while dq and nums[dq[-1]] < x:
            left.append(f"beat {nums[dq[-1]]}@{dq[-1]}")
            dq.pop()
            pops += 1
        dq.append(i)
        pushes += 1
        shown = " ".join(f"{nums[j]}@{j}" for j in dq)
        ans = ""
        if i >= k - 1:
            out.append(nums[dq[0]])
            ans = f"max={out[-1]}"
        line = (f"i={i} x={x:>2}  {', '.join(left) or '-':<22}"
                f" deque=[{shown}]  {ans}")
        print(line.rstrip())
    print(f"pushes={pushes} pops={pops} left in deque={len(dq)}")
    return out


def brute_force(nums, k):
    return [max(nums[i:i + k]) for i in range(len(nums) - k + 1)]


if __name__ == "__main__":
    import random

    nums, k = [1, 3, -1, -3, 5, 3, 6, 7], 3
    print(max_sliding_window(nums, k))
    trace(nums, k)

    rng = random.Random(0)
    for _ in range(1000):
        n = rng.randint(1, 12)
        a = [rng.randint(-5, 5) for _ in range(n)]
        k = rng.randint(1, n)
        assert max_sliding_window(a, k) == brute_force(a, k)
    print("1000 random cases match brute force")
