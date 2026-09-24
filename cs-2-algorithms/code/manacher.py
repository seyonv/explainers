"""Manacher's algorithm: every palindrome radius in O(n).

Card: cs-2-algorithms/manacher.html
Source: ljeng/cheat-sheet, coding-algorithms/algorithms.md
(Longest Palindromic Substring) and code/manacher.cpp.
"""


def manacher(s):
    t = "#" + "#".join(s) + "#"      # 'babad' -> '#b#a#b#a#d#'
    n = len(t)
    p = [0] * n        # p[i] = radius in t = palindrome length in s
    c = r = 0          # centre and right edge of rightmost palindrome
    for i in range(n):
        if i < r:                    # inside it: copy the mirror
            p[i] = min(r - i, p[2 * c - i])
        while (i - p[i] > 0 and i + p[i] + 1 < n
               and t[i - p[i] - 1] == t[i + p[i] + 1]):
            p[i] += 1                # expand past what is known
        if i + p[i] > r:
            c, r = i, i + p[i]       # new rightmost palindrome
    return p


def longest_palindrome(s):
    p = manacher(s)
    i = max(range(len(p)), key=p.__getitem__)   # first maximum
    start = (i - p[i]) // 2          # t index -> s index
    return s[start:start + p[i]]


def radii(s, mirror=True):
    """Radii plus the number of character comparisons made.

    mirror=False is plain expand-around-centre on the same '#' string.
    """
    t = "#" + "#".join(s) + "#"
    n = len(t)
    p, c, r, cmps, rows = [0] * n, 0, 0, 0, []
    for i in range(n):
        start, mir = 0, None
        if mirror and i < r:
            mir = 2 * c - i
            p[i] = start = min(r - i, p[mir])
        k = 0
        while i - p[i] > 0 and i + p[i] + 1 < n:
            k += 1
            if t[i - p[i] - 1] != t[i + p[i] + 1]:
                break
            p[i] += 1
        cmps += k
        if i + p[i] > r:
            c, r = i, i + p[i]
        rows.append((i, t[i], mir, start, k, p[i], c, r))
    return p, cmps, rows


def brute(s):
    best = ""
    for i in range(len(s)):
        for j in range(i, len(s)):
            w = s[i:j + 1]
            if w == w[::-1] and len(w) > len(best):
                best = w
    return best


if __name__ == "__main__":
    import random

    s = "babad"
    print("#" + "#".join(s) + "#")
    print(manacher(s))
    print(longest_palindrome(s))
    print()
    print(" i t[i] mirror start cmps p[i] (c, r) after")
    for i, ch, mir, st, k, pi, c, r in radii(s)[2]:
        m = "-" if mir is None else mir
        row = f"{i:2} {ch:>4} {m:>6} {st:>5} {k:>4} {pi:>4}"
        print(row, f"({c}, {r})")
    print()
    for w in ["babad", "a" * 10, "a" * 1000]:
        _, m, _ = radii(w)
        _, e, _ = radii(w, mirror=False)
        name = w if len(w) < 12 else f"'a' * {len(w)}"
        print(f"{name}: manacher {m} cmps, expand {e} cmps")

    random.seed(1)
    for _ in range(2000):
        w = "".join(random.choice("ab") for _ in range(
            random.randint(1, 12)))
        got = longest_palindrome(w)
        assert got == got[::-1] and got in w
        assert len(got) == len(brute(w))
    print("2,000 random strings match brute force")
