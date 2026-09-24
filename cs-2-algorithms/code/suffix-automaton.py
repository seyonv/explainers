"""Suffix automaton and Longest Duplicate Substring.

Run: python3 suffix-automaton.py   (Python 3.10, standard library only)
"""
import random


def build(s, trace=False):
    """Suffix automaton of s as parallel lists: len, link, next."""
    length, link, nxt = [0], [-1], [{}]
    last = 0
    best, end = 0, 0              # longest repeat: its length, end
    for i, c in enumerate(s):
        cur = len(length)         # new state for the prefix s[:i+1]
        length.append(length[last] + 1)
        link.append(0)
        nxt.append({})
        p, walked = last, []
        while p != -1 and c not in nxt[p]:
            nxt[p][c] = cur       # every suffix without c now gets c
            walked.append(p)
            p = link[p]
        clone = None
        if p != -1:
            q = nxt[p][c]
            if length[q] == length[p] + 1:
                link[cur] = q     # q is exactly "suffix + c": reuse
            else:                 # q is too long: split it with a clone
                clone = len(length)
                length.append(length[p] + 1)
                link.append(link[q])
                nxt.append(dict(nxt[q]))
                while p != -1 and nxt[p].get(c) == q:
                    nxt[p][c] = clone
                    p = link[p]
                link[q] = link[cur] = clone
        last = cur
        rep = length[link[cur]]   # longest suffix seen before
        if rep > best:
            best, end = rep, i + 1
        if trace:
            cl = f"  clone {clone}" if clone is not None else ""
            print(f"  {i + 1} {c!r}: new {cur} (len {length[cur]})"
                  f"  walked {walked}{cl}  link -> {link[cur]}"
                  f"  repeat {s[i + 1 - rep:i + 1]!r}")
    return length, link, nxt, s[end - best:end]


# The card's version: same algorithm as build(), no tracing.
def longest_dup_substring(s):
    length, link, nxt = [0], [-1], [{}]   # state 0 = ""
    last, best, end = 0, 0, 0
    for i, c in enumerate(s):
        cur = len(length)                 # state for s[:i+1]
        length.append(length[last] + 1)
        link.append(0)
        nxt.append({})
        p = last
        while p != -1 and c not in nxt[p]:
            nxt[p][c] = cur               # suffixes lacking c get it
            p = link[p]
        if p != -1:
            q = nxt[p][c]
            if length[q] == length[p] + 1:
                link[cur] = q             # q fits exactly: reuse
            else:                         # q too long: clone it
                clone = len(length)
                length.append(length[p] + 1)
                link.append(link[q])
                nxt.append(dict(nxt[q]))
                while p != -1 and nxt[p].get(c) == q:
                    nxt[p][c] = clone
                    p = link[p]
                link[q] = link[cur] = clone
        last = cur
        rep = length[link[cur]]           # longest suffix seen before
        if rep > best:
            best, end = rep, i + 1
    return s[end - best:end]


class State:                      # the source's version, verbatim
    def __init__(self, link=-1):
        self.link = link
        self.word = ''
        self.next = dict()


def longestDupSubstring(s):
    automaton = [State()]
    last = 0
    lds = ''
    for x in s:
        last, p = len(automaton), last
        automaton.append(State())
        automaton[last].word = automaton[p].word + x
        while p >= 0 and x not in automaton[p].next:
            automaton[p].next[x] = last
            p = automaton[p].link
        if p >= 0:
            q = automaton[p].next[x]
            if len(automaton[q].word) == len(automaton[p].word) + 1:
                automaton[last].link = q
                lds = max([lds, automaton[q].word], key=len)
            else:
                automaton.append(State(automaton[q].link))
                last += 1
                automaton[last].word = automaton[p].word + x
                automaton[last].next = automaton[q].next.copy()
                lds = max([lds, automaton[last].word], key=len)
                while p >= 0 and automaton[p].next.get(x, None) == q:
                    automaton[p].next[x] = last
                    p = automaton[p].link
                last -= 1
                automaton[q].link = automaton[last].link = last + 1
        else: automaton[last].link = 0
    return lds


def brute(s):
    n = len(s)
    for L in range(n - 1, 0, -1):
        seen = set()
        for i in range(n - L + 1):
            t = s[i:i + L]
            if t in seen:
                return t
            seen.add(t)
    return ""


def strings_of(s, length, link, nxt):
    """All substrings grouped by state (only for tiny s)."""
    by = {}
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            v = 0
            for c in s[i:j]:
                v = nxt[v][c]
            by.setdefault(v, set()).add(s[i:j])
    return by


def endpos(s, t):
    return [i + len(t) for i in range(len(s) - len(t) + 1)
            if s.startswith(t, i)]


if __name__ == "__main__":
    s = "banana"
    print(f"build({s!r}):")
    length, link, nxt, lds = build(s, trace=True)
    print("longest duplicate substring:", repr(lds))
    by = strings_of(s, length, link, nxt)
    print("\nstate len link  strings (endpos)  next")
    for v in range(len(length)):
        ws = sorted(by.get(v, {""}), key=len, reverse=True)
        ep = endpos(s, ws[0]) if ws[0] else "all"
        print(f"  {v}   {length[v]}   {link[v]:>2}   {ws} {ep}"
              f"  {nxt[v]}")
    n, T = len(s), sum(len(d) for d in nxt)
    d = sum(length[v] - length[link[v]] for v in range(1, len(length)))
    subs = {s[i:j] for i in range(n) for j in range(i + 1, n + 1)}
    assert d == len(subs)
    print(f"distinct substrings = sum(len - len(link)) = {d}")
    print(f"states {len(length)} <= 2n-1 = {2 * n - 1};"
          f" transitions {T} <= 3n-4 = {3 * n - 4}")

    words = ["banana", "mississippi", "abcd"]
    print([longest_dup_substring(w) for w in words])

    r = random.Random(0)
    for _ in range(3000):
        t = "".join(r.choice("abc"[:r.randint(1, 3)])
                    for _ in range(r.randint(0, 14)))
        d = longest_dup_substring(t)
        assert len(d) == len(brute(t)) == len(build(t)[3]), t
        assert not d or len(endpos(t, d)) >= 2, t
        assert len(longestDupSubstring(t)) == len(d), t
    print("3,000 random strings: card, traced and source"
          " versions all match brute force")

    for name, t in [("'a' * 30000", "a" * 30000),
                    ("random a-z, n=30000", "".join(
                        r.choice("abcdefghijklmnopqrstuvwxyz")
                        for _ in range(30000)))]:
        length, link, nxt, lds = build(t)
        print(f"{name}: states {len(length)}, transitions "
              f"{sum(len(d) for d in nxt)}, repeat len {len(lds)}")
    n = 30000
    print(f"source stores whole strings: 'a'*{n} keeps"
          f" 1+2+...+{n} = {n * (n + 1) // 2:,} chars")
