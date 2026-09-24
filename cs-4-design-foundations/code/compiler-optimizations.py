"""What CPython 3.10 does (and doesn't) optimize, shown with dis."""
import dis
import sys
import time


def folded():
    return 2 * 3            # constant folding: compiled as 6


def square(x):
    return x * x


def use_square():
    return square(5)        # no inlining: the call stays


def times8(x):
    return x * 8            # no strength reduction: still a multiply


def cse(x, y):
    return (x + y) * (x + y)  # no CSE: x + y is computed twice


def sum_to(n, acc=0):       # a tail call: nothing left to do after
    return acc if n == 0 else sum_to(n - 1, acc + n)


def concat(n):              # the source's Java loop, in Python
    s = ""
    for i in range(n):
        s = s + str(i) + " "    # copies all of s every pass
    return s


def concat_inplace(n):      # same result, the shape CPython helps
    s = ""
    for i in range(n):
        s += str(i) + " "       # s may be resized in place
    return s


def joined(n):
    return "".join(str(i) + " " for i in range(n))


def ops(f):
    return [(i.opname + " " + i.argrepr).strip()
            for i in dis.get_instructions(f)]


def chars_copied(n):
    """Lower bound: s + str(i) must copy all of the old s."""
    total = length = 0
    for i in range(n):
        total += length
        length += len(str(i)) + 1
    return total, length


def seconds(f, n):
    t = time.perf_counter()
    f(n)
    return time.perf_counter() - t


if __name__ == "__main__":
    for f in (folded, use_square, times8, cse):
        print(f.__name__ + ":", " · ".join(ops(f)))
    print("recursion limit:", sys.getrecursionlimit())
    print("sum_to(900) =", sum_to(900))
    try:
        sum_to(10_000)
    except RecursionError as e:
        print("sum_to(10_000): RecursionError:", e)
    n = 10**5
    assert concat(n) == concat_inplace(n) == joined(n)
    for f in (concat, concat_inplace, joined):
        print(f"{f.__name__:15} {seconds(f, n):.4f} s  (one run)")
    total, length = chars_copied(n)
    print(f"final length {length:,} chars; "
          f"s + str(i) copies at least {total:,}")
