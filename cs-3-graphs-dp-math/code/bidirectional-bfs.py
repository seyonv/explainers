"""Bidirectional BFS on Word Ladder (cheat-sheet > Graphs > BFS).

Grow one BFS layer at a time from both ends, always from the smaller
frontier, and stop the first time a new word lies on the other
frontier.  Python 3.10, standard library only.
"""
import os
from collections import deque
from string import ascii_lowercase


def neighbours(w, words):
    for i in range(len(w)):
        for c in ascii_lowercase:
            v = w[:i] + c + w[i + 1:]
            if v != w and v in words:
                yield v


def ladder_length(begin, end, word_list, log=None):
    words = set(word_list)             # never mutate the caller's list
    if end not in words:
        return 0
    front, back = {begin}, {end}
    seen = {begin, end}
    length = 1                         # words on the ladder so far
    while front and back:
        if len(front) > len(back):     # always grow the smaller side
            front, back = back, front
        length += 1
        nxt = set()
        for w in sorted(front):        # sorted: same trace every run
            for v in neighbours(w, words):
                if v in back:          # the two searches touch
                    if log is not None:
                        log.append((sorted(front), v, sorted(back)))
                    return length
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        if log is not None:
            log.append((sorted(front), sorted(nxt), sorted(back)))
        front = nxt
    return 0


def one_way(begin, end, word_list):
    """Plain BFS from begin only; returns (length, words expanded)."""
    words = set(word_list)
    q, dist, expanded = deque([begin]), {begin: 1}, 0
    while q:
        w = q.popleft()
        if w == end:
            return dist[w], expanded
        expanded += 1
        for v in neighbours(w, words):
            if v not in dist:
                dist[v] = dist[w] + 1
                q.append(v)
    return 0, expanded


def two_way_count(begin, end, word_list):
    """Same as ladder_length but counts words expanded."""
    words = set(word_list)
    front, back, seen = {begin}, {end}, {begin, end}
    length, expanded = 1, 0
    while front and back:
        if len(front) > len(back):
            front, back = back, front
        length += 1
        nxt = set()
        for w in sorted(front):
            expanded += 1
            for v in neighbours(w, words):
                if v in back:
                    return length, expanded
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        front = nxt
    return 0, expanded


if __name__ == "__main__":
    wl = ["hot", "dot", "dog", "lot", "log", "cog"]
    log = []
    print(ladder_length("hit", "cog", wl, log))       # 5
    for step, (f, new, b) in enumerate(log, 1):
        print(f"  step {step}: expand {f} -> {new}   other {b}")
    print(ladder_length("hit", "cog", wl[:-1]))       # 0, no cog
    print("one-way (length, expanded):", one_way("hit", "cog", wl))
    print("two-way (length, expanded):",
          two_way_count("hit", "cog", wl))

    b, d = 10, 6
    print(f"b^d = {b**d:,}   2*b^(d/2) = {2 * b**(d // 2):,}"
          f"   ratio = {b**d // (2 * b**(d // 2)):,}x")

    path = "/usr/share/dict/words"                    # macOS web2
    if os.path.exists(path):
        with open(path) as f:
            dic = {w.strip() for w in f}
        four = [w for w in dic if len(w) == 4 and w.isalpha()
                and w.islower()]
        for s, t in [("cold", "warm"), ("head", "tail")]:
            print(f"{s}->{t} ({len(four):,} words):",
                  "one-way", one_way(s, t, four),
                  "two-way", two_way_count(s, t, four))
