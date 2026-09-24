"""DP on two strings: regex matching and distinct subsequences.

Both problems from the cheat-sheet's Recursion page (C++ there),
rewritten in Python. dp[i][j] answers the question for the first
i characters of s and the first j characters of the pattern / t.
Runs under python3 (3.10).
"""
import itertools
import random
import re


def is_match_table(s, p):
    """Full (len(s)+1) x (len(p)+1) table of regex matches."""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True                      # empty matches empty
    for j in range(2, n + 1):            # "x*" can match nothing
        dp[0][j] = p[j - 1] == "*" and dp[0][j - 2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = p[j - 1]
            if c == "*":
                x = p[j - 2]
                zero = dp[i][j - 2]      # drop "x*" entirely
                more = x in (s[i - 1], ".") and dp[i - 1][j]
                dp[i][j] = zero or more  # or eat one s char
            else:
                dp[i][j] = c in (s[i - 1], ".") and dp[i - 1][j - 1]
    return dp


def is_match(s, p):              # regex with . and *, two rows
    n = len(p)
    prev = [True] + [False] * n     # row i-1; starts as row 0
    for j in range(2, n + 1):        # "x*" can match nothing
        prev[j] = p[j - 1] == "*" and prev[j - 2]
    for ch in s:
        cur = [False] * (n + 1)     # cur[0]: s nonempty, p empty
        for j in range(1, n + 1):
            c = p[j - 1]
            if c == "*":
                cur[j] = cur[j - 2] or (   # zero times, or
                    p[j - 2] in (ch, ".") and prev[j])  # one more
            else:
                cur[j] = c in (ch, ".") and prev[j - 1]
        prev = cur
    return prev[n]


def is_match_source(s, p):
    """Line-by-line port of the source's C++, to test it."""
    m, n = len(s), len(p)
    rows = [[False] * (n + 1) for _ in range(2)]
    rows[0][0] = True
    for j in range(1, n):
        if p[j] == "*":
            rows[0][j + 1] = rows[0][j - 1]
    for i in range(m):
        for j in range(n):
            if p[j] == s[i] or p[j] == ".":
                rows[1][j + 1] = rows[0][j]
            elif p[j] == "*" and j:
                rows[1][j + 1] = rows[1][j - 1] or (
                    (p[j - 1] == s[i] or p[j - 1] == ".")
                    and (rows[0][j + 1] or rows[1][j]
                         or rows[1][j + 1]))
            elif p[j] != "*":
                rows[1][j + 1] = False
        rows[0] = rows[1][:]
    return rows[0][n]


def num_distinct_table(s, t):
    """dp[i][j] = ways t[:j] appears as a subsequence of s[:i]."""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1                     # empty t: one way
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]      # skip s[i-1]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]  # or use it
    return dp


def num_distinct(s, t):          # one row, j backwards
    dp = [1] + [0] * len(t)
    for ch in s:
        for j in range(len(t), 0, -1):
            if ch == t[j - 1]:
                dp[j] += dp[j - 1]  # dp[j-1] is still last row's
    return dp[-1]


def num_distinct_trace(s, t):
    """Same as num_distinct, printing the row after each char."""
    n = len(t)
    dp = [1] + [0] * n
    for ch in s:
        for j in range(n, 0, -1):
            if ch == t[j - 1]:
                dp[j] += dp[j - 1]
        print(f"  after {ch}: {dp}")
    return dp[n]


def num_distinct_forward(s, t):
    """Bug demo: j forwards reads a value already updated."""
    n = len(t)
    dp = [1] + [0] * n
    for ch in s:
        for j in range(1, n + 1):
            if ch == t[j - 1]:
                dp[j] += dp[j - 1]
    return dp[n]


def brute_distinct(s, t):
    return sum(1 for idx in itertools.combinations(range(len(s)),
                                                   len(t))
               if "".join(s[i] for i in idx) == t)


def show_bool_table(s, p, dp):
    print("      " + "  ".join(f"{c:>2}" for c in "ε" + p))
    for i, row in enumerate(dp):
        label = "ε" if i == 0 else s[i - 1]
        cells = "  ".join(" T" if v else " ." for v in row)
        print(f"  {label:>2}  {cells}")


if __name__ == "__main__":
    print(is_match("aab", "c*a*b"), num_distinct("rabbbit", "rabbit"))
    s, p = "aab", "c*a*b"
    print(f"\nregex: s={s!r} p={p!r}")
    table = is_match_table(s, p)
    show_bool_table(s, p, table)
    print("  full table:", table[-1][-1],
          " two rows:", is_match(s, p),
          " source port:", is_match_source(s, p))

    print("\ndistinct subsequences: s='rabbbit' t='rabbit'")
    print("  answer:", num_distinct_trace("rabbbit", "rabbit"))
    print("  j forwards (wrong):",
          num_distinct_forward("rabbbit", "rabbit"))
    print("  full-table answer:",
          num_distinct_table("rabbbit", "rabbit")[-1][-1])

    rng = random.Random(0)
    bad = 0
    for _ in range(20000):
        s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 6)))
        p = ""
        for _ in range(rng.randint(0, 4)):
            p += rng.choice("ab.")
            if rng.random() < 0.5:
                p += "*"
        want = re.fullmatch(p, s) is not None
        got = (is_match(s, p), is_match_table(s, p)[-1][-1],
               is_match_source(s, p))
        bad += any(g != want for g in got)
    print("\nregex vs re.fullmatch, 20,000 random cases, mismatches:",
          bad)
    bad = 0
    for _ in range(2000):
        s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 9)))
        t = "".join(rng.choice("ab") for _ in range(rng.randint(0, 4)))
        bad += num_distinct(s, t) != brute_distinct(s, t)
    print("distinct vs brute force, 2,000 random cases, mismatches:",
          bad)
