"""Fenwick tree (binary indexed tree), 1-D and 2-D.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md,
Trees > Range Sum Query 2-D - Mutable (a C++ 2-D BIT). Rewritten in
Python; the 1-D class comes first because the 2-D one is the same
loop run once per axis.
"""


class Fenwick:
    """Prefix sums with point updates. Positions are 1-indexed."""

    def __init__(self, n):
        self.t = [0] * (n + 1)       # t[0] unused

    def add(self, i, delta):         # a[i] += delta
        while i < len(self.t):
            self.t[i] += delta
            i += i & -i              # next node that covers i

    def query(self, i):              # a[1] + ... + a[i]
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i              # drop the lowest set bit
        return s

    def range_sum(self, lo, hi):     # a[lo] + ... + a[hi]
        return self.query(hi) - self.query(lo - 1)


class NumMatrix:
    """Range Sum Query 2-D - Mutable. Cells are 0-indexed."""

    def __init__(self, matrix):
        self.m, self.n = len(matrix), len(matrix[0])
        self.a = [[0] * self.n for _ in range(self.m)]
        self.t = [[0] * (self.n + 1) for _ in range(self.m + 1)]
        for r, row in enumerate(matrix):
            for c, v in enumerate(row):
                self.update(r, c, v)

    def update(self, row, col, val):
        delta, self.a[row][col] = val - self.a[row][col], val
        i = row + 1
        while i <= self.m:
            j = col + 1
            while j <= self.n:
                self.t[i][j] += delta
                j += j & -j
            i += i & -i

    def _sum(self, r, c):            # sum of cells above-left of (r, c)
        s, i = 0, r
        while i > 0:
            j = c
            while j > 0:
                s += self.t[i][j]
                j -= j & -j
            i -= i & -i
        return s

    def sumRegion(self, r1, c1, r2, c2):
        return (self._sum(r2 + 1, c2 + 1) - self._sum(r1, c2 + 1)
                - self._sum(r2 + 1, c1) + self._sum(r1, c1))


def query_path(i):
    path = []
    while i > 0:
        path.append(i)
        i -= i & -i
    return path


def update_path(i, n):
    path = []
    while i <= n:
        path.append(i)
        i += i & -i
    return path


if __name__ == "__main__":
    import random

    a = [3, 2, -1, 6, 5, 4, -3, 3]
    f = Fenwick(len(a))
    for i, v in enumerate(a, 1):
        f.add(i, v)
    for i in range(1, len(a) + 1):
        lo = i - (i & -i) + 1
        print(f"t[{i}] = a[{lo}..{i}] = {f.t[i]}")
    print("t =", f.t[1:])            # [3, 5, -1, 10, 5, 9, -3, 19]
    print("query(6) path", query_path(6), "=", f.query(6))  # 19
    print("add(3, +2) path", update_path(3, len(a)))  # [3, 4, 8]
    f.add(3, 2)
    print("t =", f.t[1:])            # [3, 5, 1, 12, 5, 9, -3, 21]
    print("query(6) =", f.query(6))                           # 21
    print("range_sum(3, 6) =", f.range_sum(3, 6))             # 16

    M = [[3, 2, -1, 6],
         [5, 4, -3, 3],
         [1, 0, 2, 7],
         [4, -2, 5, 1]]
    nm = NumMatrix(M)
    print("sumRegion(1,1,2,2) =", nm.sumRegion(1, 1, 2, 2))   # 3
    nm.update(1, 2, 0)
    print("after update(1,2,0):", nm.sumRegion(1, 1, 2, 2))   # 6

    rng = random.Random(0)
    for _ in range(1000):
        n = rng.randint(1, 20)
        b = [rng.randint(-9, 9) for _ in range(n)]
        g = Fenwick(n)
        for i, v in enumerate(b, 1):
            g.add(i, v)
        i, d = rng.randint(1, n), rng.randint(-9, 9)
        b[i - 1] += d
        g.add(i, d)
        lo = rng.randint(1, n)
        hi = rng.randint(lo, n)
        assert g.range_sum(lo, hi) == sum(b[lo - 1:hi])
    print("1000 random cases match brute force")
