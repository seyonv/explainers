"""Fibonacci counting: Climbing Stairs and Non-negative Integers
without Consecutive Ones (source guide > Mathematics). Python 3.10."""
import math


def climb_stairs_binet(n):          # the source's closed form
    s5 = math.sqrt(5)
    return round(((1 + s5) / 2) ** (n + 1) / s5)


def climb_stairs(n):                # exact: Python ints never round
    a, b = 1, 1                     # ways(0), ways(1)
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def find_integers(n, trace=None):
    f = [1, 2]                      # f[k]: k-bit strings, no "11"
    while len(f) < n.bit_length():
        f.append(f[-1] + f[-2])
    count, prev = 0, 0
    for k in range(n.bit_length() - 1, -1, -1):
        if n >> k & 1:
            count += f[k]           # put 0 here, k free bits below
            if trace is not None:
                trace.append((k, f[k], count))
            if prev:                # n has "11": nothing more fits
                return count
            prev = 1
        else:
            prev = 0
    return count + 1                # n itself has no "11"


def source_find_integers(n, calls):
    """The source's C++ with its global log_max, which each call
    decrements. calls = how many times it has already run."""
    log_max = 31 - calls
    fib = [1, 2]
    while len(fib) < log_max:
        fib.append(fib[-1] + fib[-2])
    counter, bit = 0, 0
    for i in range(log_max - 1, -1, -1):
        if n & (1 << i):
            counter += fib[i]
            if bit:
                return counter
            bit = 1
        else:
            bit = 0
    return counter + 1


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


if __name__ == "__main__":
    s5 = math.sqrt(5)
    phi = (1 + s5) / 2
    print("climb_stairs_binet(10) =", climb_stairs_binet(10),
          " exact:", climb_stairs(10))
    print("phi**11 / sqrt5 =", phi ** 11 / s5)

    first = next(n for n in range(1, 200)
                 if round(phi ** n / s5) != fib(n))
    print("first n with round(phi**n/sqrt5) != F(n):", first,
          "(climb_stairs n =", first - 1, ")")
    for n in (11, 30, 50, 70, 71, 72, 75):
        v = phi ** n / s5
        print(f"  F({n}) = {fib(n):,}  float {v!r}"
              f"  rounds to {round(v):,}  off by {round(v) - fib(n)}")
    big = next(n for n in range(1, 100)
               if climb_stairs(n) > 2 ** 31 - 1)
    print("C++ int overflows at climbStairs(%d) = %s"
          % (big, f"{climb_stairs(big):,}"))

    for n in (5, 13):
        steps = []
        print(f"find_integers({n}) [{n:b}] =",
              find_integers(n, steps), " steps (k, f[k], count):",
              steps)
    print("find_integers(10**9) =", f"{find_integers(10 ** 9):,}")

    ok = all(find_integers(n) == sum(1 for x in range(n + 1)
                                     if not x & (x >> 1))
             for n in range(2 ** 12))
    print("matches brute force for n < 4096:", ok)

    n = 2 ** 29
    print("source-style, n = 2**29, calls 0/1/2:",
          [source_find_integers(n, c) for c in range(3)],
          " correct:", find_integers(n))
