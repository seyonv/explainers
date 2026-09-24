"""Arithmetic with bits: Divide Two Integers by shift-and-subtract.

Card: cs-3-graphs-dp-math/bit-arithmetic.html
Source: ljeng/cheat-sheet, coding-algorithms/mathematics.md,
Divide Two Integers (Python). Rewritten here with two changes:
the source negates with `quotient *= -1` (a multiplication, which
the problem forbids) and clamps with statistics.median; this
version uses unary minus and min/max. Runs under python3 (3.10).
"""
import ctypes
import random

INT_MIN = -1 << 31              # -2,147,483,648
INT_MAX = -INT_MIN - 1          #  2,147,483,647


def divide(dividend, divisor):
    negative = (dividend < 0) != (divisor < 0)
    a, b = abs(dividend), abs(divisor)
    q = 0
    while a >= b:
        d, m = b, 1             # d = b * m, m a power of two
        while d << 1 <= a:      # double until the next would overshoot
            d <<= 1
            m <<= 1
        a -= d                  # take the biggest chunk out
        q += m                  # ...and record how many b's it held
    q = -q if negative else q
    return max(INT_MIN, min(q, INT_MAX))   # the 32-bit clamp


def divide_trace(dividend, divisor):
    """Same loop, printing each doubling round."""
    a, b, q, rnd = abs(dividend), abs(divisor), 0, 0
    while a >= b:
        rnd += 1
        d, m, tried = b, 1, [b]
        while d << 1 <= a:
            d <<= 1
            m <<= 1
            tried.append(d)
        print(f"  round {rnd}: left {a:>2}, doubles {tried}"
              f" -> take {d} = {b} x {m}, q = {q} + {m} = {q + m}")
        a -= d
        q += m
    print(f"  stop: left {a} < {b}.  quotient {q}, remainder {a}")


def steps(dividend, divisor):
    """Shift-compares done by the source's loop (restarts at b)."""
    a, b, n = abs(dividend), abs(divisor), 0
    while a >= b:
        d = b
        n += 1                  # the failing compare
        while d << 1 <= a:
            d <<= 1
            n += 1
        a -= d
    return n


def bits(x, width=8):
    return format(x & ((1 << width) - 1), f"0{width}b")


def as_int32(x):
    return ctypes.c_int32(x).value   # what a 32-bit register holds


if __name__ == "__main__":
    print("43 / 8:")
    divide_trace(43, 8)
    print(f"  divide(43, 8) = {divide(43, 8)}")
    print(f"  43 = {bits(43)}, 8 = {bits(8)}, 5 = {bits(5)}")
    print()
    for a, b in [(-7, 2), (7, -3), (0, 5), (INT_MIN, -1),
                 (INT_MIN, 1), (INT_MAX, 1)]:
        print(f"divide({a}, {b}) = {divide(a, b)}")
    print(f"true -2^31 / -1 = {INT_MIN // -1}"
          f" > INT_MAX {INT_MAX}: clamped")
    print(f"32-bit wrap of 2^31 = {as_int32(2**31)}")
    print(f"python -7 // 2 = {-7 // 2} (floors, the problem truncates)")
    print(f"int(-7 / 2)    = {int(-7 / 2)}")
    big = 2**53 + 1
    print(f"int(({big}) / 1) = {int(big / 1)}  (float rounds)")
    print()
    print("two's complement, 8 bits, x = 40:")
    x = 40
    print(f"   x      {bits(x)}")
    print(f"  ~x      {bits(~x)}")
    print(f"  -x=~x+1 {bits(-x)}")
    print(f"  x & -x  {bits(x & -x)} = {x & -x}")
    print(f"  x & (x-1) {bits(x & (x - 1))} = {x & (x - 1)}")
    print(f"  -2^31 in 32 bits: {bits(INT_MIN, 32)}")
    print(f"  -(-2^31) wrapped: {bits(as_int32(-INT_MIN), 32)}")
    print()
    print("shift-compares (source loop) vs subtractions (naive):")
    for a, b in [(43, 8), (INT_MAX, 1), (INT_MIN, -1), (10**9, 7)]:
        print(f"  {a} / {b}: {steps(a, b)} vs {abs(a) // abs(b)}")
    print()
    rng = random.Random(0)
    bad = 0
    for _ in range(100_000):
        a = rng.randint(INT_MIN, INT_MAX)
        b = rng.choice([rng.randint(-99, 99),
                        rng.randint(INT_MIN, INT_MAX)]) or 1
        want = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
        want = max(INT_MIN, min(want, INT_MAX))
        bad += divide(a, b) != want
    print(f"100,000 random 32-bit pairs vs truncating //: {bad} wrong")
