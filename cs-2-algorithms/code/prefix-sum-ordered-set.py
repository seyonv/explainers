"""Prefix sums with an ordered set: Max Sum of Rectangle <= k.

The source guide's C++ uses std::set::lower_bound. Python has no
ordered set in the standard library, so this keeps a sorted list
with bisect: O(log n) to search, O(n) to insert (the list shifts).
Checked against brute force on random matrices.
"""
import bisect
import itertools
import random


def best_at_most(xs, k, trace=False):
    """Largest sum of a contiguous run of xs that is <= k."""
    seen = [0]                  # sorted prefix sums so far
    cur, best = 0, None
    for x in xs:
        cur += x
        i = bisect.bisect_left(seen, cur - k)  # smallest >= cur-k
        if i < len(seen) and (best is None or cur - seen[i] > best):
            best = cur - seen[i]
        if trace:
            got = seen[i] if i < len(seen) else None
            print(f"  x={x:>2} cur={cur:>2} need>={cur - k:>2}"
                  f" set={seen} -> {got} sum="
                  f"{cur - got if got is not None else '-'}"
                  f" best={best}")
        bisect.insort(seen, cur)
    return best


def max_sum_submatrix(matrix, k, trace=False):
    m, n = len(matrix), len(matrix[0])
    best = None
    for j1 in range(n):
        rows = [0] * m          # row sums of columns j1..j2
        for j2 in range(j1, n):
            for i in range(m):
                rows[i] += matrix[i][j2]
            got = best_at_most(rows, k)
            if trace:
                print(f"  cols {j1}..{j2}: rows={rows} -> {got}")
            if got is not None and (best is None or got > best):
                best = got
    return best


def brute(matrix, k):
    m, n = len(matrix), len(matrix[0])
    best = None
    for r1, r2 in itertools.combinations_with_replacement(range(m), 2):
        for c1, c2 in itertools.combinations_with_replacement(
                range(n), 2):
            s = sum(matrix[i][j] for i in range(r1, r2 + 1)
                    for j in range(c1, c2 + 1))
            if s <= k and (best is None or s > best):
                best = s
    return best


if __name__ == "__main__":
    print("1D: [2, 2, -1], k = 3")
    print("answer", best_at_most([2, 2, -1], 3, trace=True))
    print()
    mat = [[1, 0, 1], [0, -2, 3]]
    print("matrix", mat, "k = 2")
    print("answer", max_sum_submatrix(mat, 2, trace=True))
    print()
    rng = random.Random(0)
    for _ in range(1000):
        m, n = rng.randint(1, 5), rng.randint(1, 5)
        a = [[rng.randint(-5, 5) for _ in range(n)] for _ in range(m)]
        k = rng.randint(-5, 10)
        assert max_sum_submatrix(a, k) == brute(a, k), (a, k)
    print("1,000 random matrices match brute force")
