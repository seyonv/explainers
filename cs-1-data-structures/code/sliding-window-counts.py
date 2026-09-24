"""Sliding window with counts.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md,
Hash > Maps > Minimum Window Substring and Substring with
Concatenation of All Words (both C++ there). Rewritten in Python
with the same ideas: a need-counter plus a `missing` total, and
k separate word-aligned windows for the concatenation problem.
"""
from collections import Counter


def min_window(s, t):
    need = Counter(t)          # chars still owed; < 0 means spare
    missing = len(t)           # total chars still owed
    left, best = 0, None
    for right, ch in enumerate(s, 1):    # window is s[left:right]
        if need[ch] > 0:
            missing -= 1       # ch paid off a debt
        need[ch] -= 1
        while missing == 0:    # valid: shrink from the left
            if best is None or right - left < best[1] - best[0]:
                best = (left, right)
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1   # dropped a char we needed
            left += 1
    return s[best[0]:best[1]] if best else ""


def find_substring(s, words):
    k, n = len(words[0]), len(words)
    want = Counter(words)
    out = []
    for off in range(k):       # k word-aligned scans
        left, have, count = off, Counter(), 0
        for right in range(off, len(s) - k + 1, k):
            w = s[right:right + k]
            if w not in want:  # a stranger: restart after it
                left, have, count = right + k, Counter(), 0
                continue
            have[w] += 1
            count += 1
            while have[w] > want[w]:     # too many w: shrink
                have[s[left:left + k]] -= 1
                left += k
                count -= 1
            if count == n:
                out.append(left)
    return sorted(out)


def trace(s, t):
    """Print one row per right step: char, missing, windows."""
    need, missing, left, best = Counter(t), len(t), 0, None
    for right, ch in enumerate(s, 1):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        m_after_grow = missing
        seen = []
        while missing == 0:
            seen.append(f"[{left},{right}) {s[left:right]}")
            if best is None or right - left < best[1] - best[0]:
                best = (left, right)
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1
        print(f"r={right:2} {ch} missing={m_after_grow} "
              f"valid={'; '.join(seen) or '-':<34} left={left}"
              f" best={s[best[0]:best[1]] if best else '-'}")


def brute_min_window(s, t):
    want = Counter(t)
    best = ""
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            if not want - Counter(s[i:j]):
                if not best or j - i < len(best):
                    best = s[i:j]
                break
    return best


def brute_find_substring(s, words):
    k, n, want = len(words[0]), len(words), Counter(words)
    return [i for i in range(len(s) - n * k + 1)
            if Counter(s[i + j * k:i + j * k + k]
                       for j in range(n)) == want]


if __name__ == "__main__":
    import random

    print(min_window("ADOBECODEBANC", "ABC"))
    trace("ADOBECODEBANC", "ABC")
    print(find_substring("barfoothefoobarman", ["foo", "bar"]))
    print(find_substring("barfoofoobarthefoobarman",
                         ["bar", "foo", "the"]))

    rng = random.Random(0)
    for _ in range(1000):
        s = "".join(rng.choice("ABC")
                    for _ in range(rng.randint(0, 12)))
        t = "".join(rng.choice("ABC") for _ in range(rng.randint(1, 3)))
        assert len(min_window(s, t)) == len(brute_min_window(s, t))
        s2 = "".join(rng.choice("ab")
                     for _ in range(rng.randint(0, 14)))
        ws = ["".join(rng.choice("ab") for _ in range(2))
              for _ in range(rng.randint(1, 3))]
        assert find_substring(s2, ws) == brute_find_substring(s2, ws)
    print("1000 random cases match brute force")
