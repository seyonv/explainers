"""Newton's method for square roots.

Source: cheat-sheet > Mathematics > sqrt(x).

my_sqrt: integer-only Newton, exact floor(sqrt(x)) for any int.
source_sqrt: the cheat-sheet's C++ version ported line for line;
it is wrong on 3 of the 2**31 inputs (see the demo).
"""
import math
from decimal import Decimal, getcontext


def my_sqrt(x):
    """floor(sqrt(x)) with integers only: Newton on y*y = x."""
    if x < 2:
        return x
    y = 1 << ((x.bit_length() + 1) // 2)  # a power of 2 >= sqrt(x)
    while True:
        z = (y + x // y) // 2          # Newton step, rounded down
        if z >= y:                     # stopped shrinking: done
            return y
        y = z


def source_sqrt(x, y=8000.0, eps=1.0):
    """The cheat-sheet's C++ (double SQRT0 = 8000, EPSILON = 1)."""
    while abs(y * y - x) > eps:
        y = (y + x / y) / 2
    return int(y)                      # truncates, like static_cast


def newton_trace(x):
    """The y values my_sqrt visits, and the final rejected z."""
    y = 1 << ((x.bit_length() + 1) // 2)
    ys = [y]
    while True:
        z = (y + x // y) // 2
        if z >= y:
            return ys, z
        y = z
        ys.append(y)


def digits_of_root2(steps=7):
    """Newton for x = 2 from y = 1; d = floor(-log10(error))."""
    getcontext().prec = 80
    root = Decimal(2).sqrt()
    y = Decimal(1)
    for n in range(steps):
        err = abs(y - root)
        d = int(-err.log10())          # floor(-log10(error))
        print(f"  y{n} = {str(y)[:26]:<26} error {float(err):.2e}"
              f"  d = {d}")
        y = (y + 2 / y) / 2


if __name__ == "__main__":
    print("x = 2, start y0 = 1 (digits double):")
    digits_of_root2()

    x = 2_147_395_599                  # 46340**2 - 1
    ys, z = newton_trace(x)
    print("\nmy_sqrt trace:", " -> ".join(map(str, ys)),
          f"(next {z} >= {ys[-1]}, stop)")
    print("my_sqrt:", my_sqrt(x), " math.isqrt:", math.isqrt(x))
    print("source :", source_sqrt(x))
    print("source with eps = 2:", source_sqrt(x, eps=2.0), "(wrong)")

    print("\nsource_sqrt near 8000**2:")
    for x in (63_984_000, 63_999_999, 64_016_000):
        print(f"  {x:,}: source {source_sqrt(x)}, "
              f"correct {math.isqrt(x)}")

    big = 94_906_266 ** 2 - 1
    print("\nint(math.sqrt(94906266**2 - 1)) =", int(math.sqrt(big)),
          " isqrt =", math.isqrt(big), " my_sqrt =", my_sqrt(big))

    ok = all(my_sqrt(n) == math.isqrt(n) for n in range(200_000))
    ok &= all(my_sqrt(k * k + d) == math.isqrt(k * k + d)
              for k in range(1, 46_341) for d in (-1, 0, 1))
    ok &= my_sqrt(10**100) == 10**50
    print("my_sqrt == math.isqrt on all checks:", ok)
