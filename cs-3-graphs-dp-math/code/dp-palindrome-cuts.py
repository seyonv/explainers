"""Palindrome Partitioning II: minimum cuts, expand around centres.

Card: cs-3-graphs-dp-math/dp-palindrome-cuts.html
Source: ljeng/cheat-sheet, coding-algorithms/recursion.md,
Palindrome Partitioning (C++). Rewritten in Python.
The source updates cut[] inside the for-loop's increment:
    cut[i + j] = min(cut[i - j + k] + 1, cut[i + ++j])
which reads j and does ++j in one expression (clang warns
-Wunsequenced). C++ doesn't fix the order in which min's arguments
are evaluated, so it is correct only when they run left to right
(clang 17 does). Here the update and j += 1 are separate lines.
Runs under python3 (3.10).
"""
import itertools
import random


def min_cut(s):
    n = len(s)
    cut = list(range(-1, n))       # cut[e]: min cuts for s[:e]
    for c in range(n):             # centre c
        for k in (0, 1):           # k=0 odd length, k=1 even
            lo, hi = c, c + k      # s[lo..hi] is the window
            while lo >= 0 and hi < n and s[lo] == s[hi]:
                # s[lo:hi+1] is a palindrome: cut before it
                cut[hi + 1] = min(cut[hi + 1], cut[lo] + 1)
                lo, hi = lo - 1, hi + 1
    return cut[n]


def min_cut_trace(s):
    """Same algorithm, printing each centre's palindromes."""
    n = len(s)
    cut = list(range(-1, n))
    print("start      cut =", cut)
    for c in range(n):
        for k in (0, 1):
            lo, hi = c, c + k
            found = []
            while lo >= 0 and hi < n and s[lo] == s[hi]:
                old = cut[hi + 1]
                cut[hi + 1] = min(old, cut[lo] + 1)
                found.append(f"{s[lo:hi + 1]}: cut[{hi + 1}]="
                             f"min({old},cut[{lo}]+1={cut[lo] + 1})"
                             f"={cut[hi + 1]}")
                lo, hi = lo - 1, hi + 1
            if found:
                tag = "odd " if k == 0 else "even"
                print(f"c={c} {tag}", "; ".join(found))
                print("           cut =", cut)
    return cut[n]


def min_cut_source(s):
    """The source's C++ loop, ported with the same indices."""
    n = len(s)
    cut = list(range(-1, n))
    for i in range(n):
        for k in range(2):
            j = k
            while (j < min(i + k + 1, n - i)
                   and s[i - j + k] == s[i + j]):
                cut[i + j + 1] = min(cut[i - j + k] + 1,
                                     cut[i + j + 1])
                j += 1
    return cut[n]


def brute(s):
    """Try all 2^(n-1) ways to cut; keep those of palindromes."""
    n, best = len(s), len(s) - 1
    for mask in range(1 << max(n - 1, 0)):
        pieces, start = [], 0
        for b in range(n - 1):
            if mask >> b & 1:
                pieces.append(s[start:b + 1])
                start = b + 1
        pieces.append(s[start:])
        if all(p == p[::-1] for p in pieces):
            best = min(best, len(pieces) - 1)
    return best


def greedy(s):
    """Take the longest palindromic prefix each time (wrong)."""
    cuts, i = -1, 0
    while i < len(s):
        e = max(e for e in range(i + 1, len(s) + 1)
                if s[i:e] == s[i:e][::-1])
        cuts, i = cuts + 1, e
    return cuts


def shortest_greedy_counterexample():
    for n in range(1, 12):
        for t in itertools.product("ab", repeat=n):
            s = "".join(t)
            if greedy(s) != min_cut(s):
                return s, greedy(s), min_cut(s)


if __name__ == "__main__":
    print(min_cut("aab"))                       # 1: aa|b
    print(min_cut_trace("abccbc"))              # 2: a|bccb|c

    rng = random.Random(0)
    bad = 0
    for _ in range(3000):
        s = "".join(rng.choice("abc")
                    for _ in range(rng.randint(1, 12)))
        want = brute(s)
        bad += (min_cut(s), min_cut_source(s)) != (want, want)
    print("3,000 random strings (both ports) vs brute force,",
          "mismatches:", bad)

    s, g, m = shortest_greedy_counterexample()
    print(f"greedy longest prefix fails first on {s!r}:",
          f"{g} cuts vs {m}")
    for n in (6, 30):
        print(f"n={n}: ways to cut, 2^(n-1) = {2 ** (n - 1):,}")
    print(f"n=2000: n*n palindrome table = {2000 * 2000:,}")
