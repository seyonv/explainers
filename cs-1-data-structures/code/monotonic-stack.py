"""Monotonic stack: Largest Rectangle in Histogram.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md,
Stacks > Largest Rectangle in Histogram. Same algorithm; this copy
works on `heights + [0]` so the caller's list is not modified.
"""


def largest_rectangle(heights):
    h = heights + [0]            # sentinel 0 empties the stack
    stack = [-1]                 # h[-1] is that 0: never popped
    best = 0
    for i, x in enumerate(h):
        while x < h[stack[-1]]:  # bar i is the right wall for the top
            top = stack.pop()
            width = i - stack[-1] - 1   # new top is the left wall
            best = max(best, h[top] * width)
        stack.append(i)          # heights on the stack never drop
    return best


def trace(heights):
    """Print the stack after every step and each rectangle popped."""
    h = heights + [0]
    stack, best = [-1], 0
    for i, x in enumerate(h):
        pops = []
        while x < h[stack[-1]]:
            top = stack.pop()
            width = i - stack[-1] - 1
            area = h[top] * width
            best = max(best, area)
            pops.append(f"{h[top]}x{width}={area}")
        stack.append(i)
        shown = [h[j] for j in stack[1:]]
        print(f"i={i} h={x}  pop {', '.join(pops) or '-':<20}"
              f" stack={shown}  best={best}")
    return best


def brute_force(heights):
    best = 0
    for i in range(len(heights)):
        low = heights[i]
        for j in range(i, len(heights)):
            low = min(low, heights[j])
            best = max(best, low * (j - i + 1))
    return best


if __name__ == "__main__":
    import random

    heights = [2, 1, 5, 6, 2, 3]
    print(largest_rectangle(heights))
    trace(heights)
    print(heights)               # unchanged: [2, 1, 5, 6, 2, 3]

    rng = random.Random(0)
    for _ in range(1000):
        a = [rng.randint(0, 9) for _ in range(rng.randint(0, 12))]
        assert largest_rectangle(a) == brute_force(a)
    print("1000 random cases match brute force")
