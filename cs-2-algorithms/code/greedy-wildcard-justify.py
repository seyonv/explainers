"""Greedy I: wildcard matching and text justification.

Python rewrite of the source guide's Wildcard Matching (C++) and its
Text Justification (Python, reshaped here). Run: python3 this file.
"""
import random


def is_match(s, p, trace=False):
    """'?' matches one char, '*' any run of chars (even empty)."""
    i = j = 0
    star, k = -1, 0          # last '*' in p; s index after its run
    step = 0
    while i < len(s):
        if j < len(p) and p[j] in (s[i], "?"):
            i, j, act = i + 1, j + 1, "match"
        elif j < len(p) and p[j] == "*":
            star, k, j, act = j, i, j + 1, "star"   # '*' eats nothing
        elif star != -1:
            k += 1                # the last '*' eats one more char
            i, j, act = k, star + 1, "back"
        else:
            return False
        step += 1
        if trace:
            print(f"{step}  {act:5}  i={i} j={j} star={star} k={k}")
    while j < len(p) and p[j] == "*":
        j += 1                    # trailing stars match the empty run
    return j == len(p)


def full_justify(words, width):
    lines, line, letters = [], [], 0
    for w in words:
        if letters + len(line) + len(w) > width:   # len(line) = gaps
            lines.append(spread(line, letters, width))
            line, letters = [], 0
        line.append(w)
        letters += len(w)
    return lines + [" ".join(line).ljust(width)]   # last line: left


def spread(line, letters, width):
    if len(line) == 1:
        return line[0].ljust(width)
    q, r = divmod(width - letters, len(line) - 1)
    gaps = [q + 1] * r + [q] * (len(line) - 1 - r) + [0]
    return "".join(w + " " * g for w, g in zip(line, gaps))


def is_match_dp(s, p):
    """O(m*n) table, used only to check the greedy version."""
    m, n = len(s), len(p)
    d = [[False] * (n + 1) for _ in range(m + 1)]
    d[0][0] = True
    for b in range(1, n + 1):
        d[0][b] = d[0][b - 1] and p[b - 1] == "*"
    for a in range(1, m + 1):
        for b in range(1, n + 1):
            if p[b - 1] == "*":
                d[a][b] = d[a - 1][b] or d[a][b - 1]
            else:
                d[a][b] = (d[a - 1][b - 1]
                           and p[b - 1] in (s[a - 1], "?"))
    return d[m][n]


def justify_ref(words, width):
    """Straightforward reference: pack, then spread gaps one by one."""
    lines, cur = [], []
    for w in words:
        if cur and len(" ".join(cur + [w])) > width:
            lines.append(cur)
            cur = []
        cur.append(w)
    out = []
    for li, ws in enumerate(lines):
        slack = width - sum(map(len, ws))
        row = ws[0]
        for g, w in enumerate(ws[1:]):
            row += " " * (slack // (len(ws) - 1)
                          + (g < slack % (len(ws) - 1))) + w
        out.append(row.ljust(width))
    return out + [" ".join(cur).ljust(width)]


if __name__ == "__main__":
    print("is_match('adceb', '*a*b'):")
    print("->", is_match("adceb", "*a*b", trace=True))
    print("is_match('acdcb', 'a*c?b'):",
          is_match("acdcb", "a*c?b"))

    words = ["This", "is", "an", "example", "of", "text",
             "justification."]
    for row in full_justify(words, 16):
        print(repr(row))

    rng = random.Random(0)
    for _ in range(5000):
        s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 8)))
        p = "".join(rng.choice("ab?*")
                    for _ in range(rng.randint(0, 6)))
        assert is_match(s, p) == is_match_dp(s, p), (s, p)
        w = rng.randint(1, 20)
        ws = ["x" * rng.randint(1, w) for _ in range(rng.randint(1, 8))]
        assert full_justify(ws, w) == justify_ref(ws, w), (ws, w)
    print("5,000 random cases each agree with DP / reference")
