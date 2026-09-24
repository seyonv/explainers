"""Rolling hash (Rabin-Karp) and Longest Duplicate Substring.

Run: python3 rolling-hash.py   (Python 3.10, standard library only)
"""
import random

B, P = 256, 101                   # the card's tiny example: base, prime


def rabin_karp(text, pat, base=B, mod=P):
    m = len(pat)
    if m == 0 or m > len(text):
        return []
    top = pow(base, m - 1, mod)   # weight of the char that leaves
    hp = ht = 0
    for i in range(m):            # hash the pattern and first window
        hp = (hp * base + ord(pat[i])) % mod
        ht = (ht * base + ord(text[i])) % mod
    hits = []
    for i in range(len(text) - m + 1):
        if ht == hp and text[i:i + m] == pat:   # verify every hit
            hits.append(i)
        if i + m < len(text):     # slide: drop text[i], add text[i+m]
            ht = ((ht - ord(text[i]) * top) * base
                  + ord(text[i + m])) % mod
    return hits


M1, M2 = (1 << 61) - 1, 10**9 + 7          # two moduli: double hash
BASE = random.Random(0).randrange(256, M2)  # fixed seed: same output


def window_keys(s, L):
    """Yield (start, (h1, h2)) for every length-L window of s."""
    h1 = h2 = 0
    t1, t2 = pow(BASE, L, M1), pow(BASE, L, M2)
    for i, c in enumerate(s):
        h1 = (h1 * BASE + ord(c)) % M1
        h2 = (h2 * BASE + ord(c)) % M2
        if i >= L:                # remove s[i-L], now worth BASE**L
            h1 = (h1 - ord(s[i - L]) * t1) % M1
            h2 = (h2 - ord(s[i - L]) * t2) % M2
        if i >= L - 1:
            yield i - L + 1, (h1, h2)


def dup_of_length(s, L):
    seen = {}
    for start, key in window_keys(s, L):
        j = seen.setdefault(key, start)
        if j != start and s[j:j + L] == s[start:start + L]:
            return start
    return -1


def longest_dup_substring(s, trace=False):
    lo, hi, best = 1, len(s) - 1, ""
    while lo <= hi:               # "a length-L duplicate exists" is
        L = (lo + hi) // 2        # monotone: true for L, true for L-1
        i = dup_of_length(s, L)
        if trace:
            print(f"  lo={lo} hi={hi} L={L} ->",
                  repr(s[i:i + L]) if i >= 0 else "none")
        if i >= 0:
            best, lo = s[i:i + L], L + 1
        else:
            hi = L - 1
    return best


def h(s, base=B, mod=P):
    v = 0
    for c in s:
        v = (v * base + ord(c)) % mod
    return v


if __name__ == "__main__":
    text = "abracadabra"
    print("windows of", text, "(base 256, mod 101):")
    for i in range(len(text) - 2):
        w = text[i:i + 3]
        hit = "hit" if h(w) == h("bra") else ""
        print(f"  {i:2} {w} {h(w):3}", hit)
    top = pow(B, 2, P)
    x = h("abr")
    print("256^2 mod 101 =", top)
    print("slide abr->bra:",
          f"(({x} - 97*{top}) * 256 + 97) % 101 =",
          ((x - 97 * top) * 256 + 97) % P)
    print("rabin_karp:", rabin_karp(text, "bra"))
    print("h('ahw') =", h("ahw"), "(same as h('bra'): a false hit)")
    print("rabin_karp('xahwx', 'bra'):", rabin_karp("xahwx", "bra"))
    print("longest duplicate in 'banana':")
    print(" ", repr(longest_dup_substring("banana", trace=True)))

    rng = random.Random(1)        # check against brute force
    for _ in range(300):
        s = "".join(rng.choice("ab") for _ in range(rng.randint(1, 12)))
        want = max((s[i:j] for i in range(len(s))
                    for j in range(i + 1, len(s) + 1)
                    if s.find(s[i:j], i + 1) != -1),
                   key=len, default="")
        assert len(longest_dup_substring(s)) == len(want), s
        p = s[:rng.randint(1, 3)]
        assert rabin_karp(s, p) == [i for i in range(len(s))
                                    if s.startswith(p, i)]
    print("300 random strings agree with brute force")
