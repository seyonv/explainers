"""Knuth-Morris-Pratt: the lps (prefix-function) table, search,
and Shortest Palindrome. Python 3.10, standard library only."""
import random


def build_lps(p):
    """lps[j] = length of the longest proper prefix of p[:j+1]
    that is also a suffix of it (its longest border)."""
    lps = [0] * len(p)
    t = 0                        # length of the border so far
    for j in range(1, len(p)):
        while t and p[j] != p[t]:
            t = lps[t - 1]       # fall back to the next border
        if p[j] == p[t]:
            t += 1               # extend the border by one char
        lps[j] = t
    return lps


def kmp_search(p, s):
    """Start index of every match of p in s. Never moves back in s."""
    lps, j, out = build_lps(p), 0, []
    for k, c in enumerate(s):
        while j and c != p[j]:
            j = lps[j - 1]       # keep the matched border, retry
        if c == p[j]:
            j += 1
        if j == len(p):
            out.append(k - j + 1)
            j = lps[j - 1]       # allow overlapping matches
    return out


def shortest_palindrome(s):
    """Add the fewest characters in front of s to make a palindrome."""
    r = s[::-1]
    k = build_lps(s + "#" + r)[-1]   # longest palindromic prefix
    return r[:len(s) - k] + s


# ---- helpers for the demo: traces, comparison counts, checks ----

def lps_trace(p):
    lps, t = [0] * len(p), 0
    for j in range(1, len(p)):
        falls = []
        while t and p[j] != p[t]:
            falls.append(f"{t}->{lps[t - 1]}")
            t = lps[t - 1]
        hit = p[j] == p[t]
        cmp = f"p[{j}]={p[j]} vs p[{t}]={p[t]}"
        t += hit
        lps[j] = t
        print(f"  j={j} {cmp}  fall {falls or '-'}  "
              f"{'match' if hit else 'miss '}  lps={t}")
    return lps


def kmp_count(p, s):
    lps, j, cmp, out = build_lps(p), 0, 0, []
    for k, c in enumerate(s):
        while j and c != p[j]:
            cmp += 1
            j = lps[j - 1]
        cmp += 1
        if c == p[j]:
            j += 1
        if j == len(p):
            out.append(k - j + 1)
            j = lps[j - 1]
    return out, cmp


def naive_count(p, s):
    cmp, out = 0, []
    for i in range(len(s) - len(p) + 1):
        for j in range(len(p)):
            cmp += 1
            if s[i + j] != p[j]:
                break
        else:
            out.append(i)
    return out, cmp


def brute_palindrome(s):
    for k in range(len(s), -1, -1):
        if s[:k] == s[:k][::-1]:
            return s[k:][::-1] + s


if __name__ == "__main__":
    p = "ABABCABAB"
    print("build lps for", p)
    print("lps =", lps_trace(p))

    s = "ABABDABACDABABCABAB"
    print("\nsearch", p, "in", s)
    print("KMP  :", kmp_count(p, s), " (matches, char comparisons)")
    print("naive:", naive_count(p, s))

    P, S = "a" * 99 + "b", "a" * 10_000
    print("\nworst case for naive: m = 100, n = 10,000")
    print("KMP  comparisons:", kmp_count(P, S)[1])
    print("naive comparisons:", naive_count(P, S)[1])

    x = "aacecaaa"
    sup = x + "#" + x[::-1]
    print("\nshortest palindrome of", x)
    print("  lps of", sup, "=", build_lps(sup))
    print("  ->", shortest_palindrome(x))
    print("  'aaa' without '#': border of 'aaaaaa' =",
          build_lps("aaaaaa")[-1], "> len 3; with '#':",
          build_lps("aaa#aaa")[-1])

    rng = random.Random(0)
    for _ in range(2000):
        a = "".join(rng.choices("ab", k=rng.randint(1, 5)))
        b = "".join(rng.choices("ab", k=rng.randint(0, 30)))
        assert kmp_search(a, b) == naive_count(a, b)[0]
        assert shortest_palindrome(b) == brute_palindrome(b)
    print("\n2,000 random checks vs brute force: ok")
