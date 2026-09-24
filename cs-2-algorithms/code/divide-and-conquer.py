"""Divide and conquer and the master theorem.

1. Karatsuba multiplication on decimal digits, counting the
   one-digit multiplications it does (3^k for n = 2^k digits).
2. The recursion-tree work per level for four recurrences,
   checked against the master theorem's prediction.
"""
import math
import random

calls = 0                      # one-digit multiplications done


def karatsuba(x, y, n):
    """x * y for non-negative ints with at most n digits.

    n is a power of two. Three half-size products, not four.
    """
    global calls
    if n == 1:
        calls += 1
        return x * y
    m = n // 2
    a, b = divmod(x, 10 ** m)          # x = a·10^m + b
    c, d = divmod(y, 10 ** m)          # y = c·10^m + d
    ac = karatsuba(a, c, m)
    bd = karatsuba(b, d, m)
    # ad + bc = ac + bd + (a - b)(d - c); |a - b| < 10^m fits
    s = (1 if a >= b else -1) * (1 if d >= c else -1)
    mid = ac + bd + s * karatsuba(abs(a - b), abs(d - c), m)
    return ac * 10 ** (2 * m) + mid * 10 ** m + bd


RECURRENCES = {
    # name: (a, b, f), meaning T(n) = a T(n/b) + f(n), T(1) = 1
    "binary search": (1, 2, lambda n: 1),
    "merge sort": (2, 2, lambda n: n),
    "Karatsuba": (3, 2, lambda n: n),
    "quickselect": (1, 2, lambda n: n),
}


def T(a, b, f, n):
    return 1 if n == 1 else a * T(a, b, f, n // b) + f(n)


def level_work(a, b, f, n):
    """Work at each depth of the recursion tree, T(1) = 1."""
    rows, nodes = [], 1
    while n > 1:
        rows.append(nodes * f(n))
        nodes, n = nodes * a, n // b
    rows.append(nodes * 1)             # the leaves
    return rows


if __name__ == "__main__":
    random.seed(0)
    n = 1024
    x = random.randrange(10 ** (n - 1), 10 ** n)
    y = random.randrange(10 ** (n - 1), 10 ** n)
    assert karatsuba(x, y, n) == x * y
    print(f"n = {n} digits")
    print(f"Karatsuba digit multiplications: {calls:,}")
    print(f"schoolbook (n^2 = 4^10):         {n * n:,}")
    print(f"ratio {n * n / calls:.2f}x, "
          f"log2(3) = {math.log2(3):.4f}")

    print("\nwork per level of the recursion tree, n = 1024")
    names = list(RECURRENCES)
    table = {k: level_work(*RECURRENCES[k], n) for k in names}
    print("level " + "".join(f"{k:>15}" for k in names))
    for lvl in range(len(table[names[0]])):
        print(f"{lvl:>5} "
              + "".join(f"{table[k][lvl]:>15,}" for k in names))
    print("total " + "".join(f"{sum(table[k]):>15,}" for k in names))
    for k in names:
        assert sum(table[k]) == T(*RECURRENCES[k], n)

    print("\nclosed forms for n = 2^10:")
    print(f"  log2 n + 1       = {int(math.log2(n)) + 1}")
    print(f"  n log2 n + n     = {n * 10 + n:,}")
    print(f"  3·3^10 − 2·2^10  = {3 * 3**10 - 2 * 2**10:,}")
    print(f"  2n − 1           = {2 * n - 1:,}")
