"""Decode Ways II: count decodings of digits and '*', mod 1e9+7.

From the cheat-sheet's Recursion page (numpy and C++ versions),
rewritten in plain Python. Three running counts replace the table:
  e   ways to decode the prefix so far (a letter just ended)
  one ways whose last char is a '1' still waiting for a partner
  two ways whose last char is a '2' still waiting for a partner
Runs under python3 (3.10).
"""
import itertools
import random

MOD = 10**9 + 7


def weights(c):
    """(alone, after a 1, after a 2) for the char c."""
    if c == "*":
        return 9, 9, 6          # 1-9 / 11-19 / 21-26
    return c != "0", 1, c <= "6"   # 1-9 / 10-19 / 20-26


def num_decodings(s):
    e, one, two = 1, 0, 0       # empty prefix: one way
    for c in s:
        a, b, d = weights(c)
        e, one, two = (
            (e * a + one * b + two * d) % MOD,
            e if c in "1*" else 0,   # c opens "1?"
            e if c in "2*" else 0,   # c opens "2?"
        )
    return e


def brute(s):
    """Expand every '*', then try every split into 1-26."""
    total = 0
    stars = s.count("*")
    for fill in itertools.product("123456789", repeat=stars):
        it = iter(fill)
        t = "".join(next(it) if c == "*" else c for c in s)
        total += splits(t)
    return total % MOD


def splits(t):
    if not t:
        return 1
    n = 0
    if t[0] != "0":
        n += splits(t[1:])
        if len(t) > 1 and 10 <= int(t[:2]) <= 26:
            n += splits(t[2:])
    return n


def source_numpy(s):
    """The source's numpy version, verbatim apart from names."""
    import numpy as np
    ways = np.array([1, 0, 0])
    for x in s:
        if x.isdigit():
            y = [[x > "0", 1, x <= "6"], [x == "1", x == "2"]]
        else:
            y = [[9, 9, 6], [1, 1]]
        ways = np.array([1, ways[0], ways[0]]) * (
            [sum(ways * y[0]) % MOD] + y[1])
    return int(ways[0])


def trace(s):
    e, one, two = 1, 0, 0
    print(f"{s!r}: char  alone,after1,after2   e  one  two")
    for c in s:
        a, b, d = weights(c)
        ne = e * a + one * b + two * d
        print(f"  {c}  ({int(a)},{b},{int(d)})  "
              f"{e}*{int(a)}+{one}*{b}+{two}*{int(d)} = {ne:<4}"
              f" one={e if c in '1*' else 0}"
              f" two={e if c in '2*' else 0}")
        e, one, two = (ne % MOD, e if c in "1*" else 0,
                       e if c in "2*" else 0)
    print("  answer", e)
    return e


def raw(s):
    """Same DP with no modulo: the true count."""
    e, one, two = 1, 0, 0
    for c in s:
        a, b, d = weights(c)
        e, one, two = (e * a + one * b + two * d,
                       e if c in "1*" else 0, e if c in "2*" else 0)
    return e


if __name__ == "__main__":
    for s in ["*", "1*", "2*", "**", "11106"]:
        print(f"{s!r:8} -> {num_decodings(s)}")
    trace("1*")
    trace("11106")
    trace("*1*0")
    for n in [5, 9, 10, 20, 100]:
        r = raw("*" * n)
        print(f"'*'x{n}: true {r} ({len(str(r))} digits),"
              f" mod {r % MOD}, fits int64: {r < 2**63}")
    big = "*" * 100_000
    print("'*'x100000 ->", num_decodings(big))

    random.seed(0)
    for _ in range(2000):
        n = random.randint(0, 7)
        s = "".join(random.choice("0123456789**") for _ in range(n))
        want = brute(s)
        assert num_decodings(s) == want, s
        assert source_numpy(s) == want, s
    print("2000 random strings match brute force (source numpy too)")
