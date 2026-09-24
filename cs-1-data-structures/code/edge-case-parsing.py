"""Corner cases: String to Integer (atoi), Valid Number, Strong
Password Checker.

Source: ljeng/cheat-sheet, coding-algorithms/coding.md, Corner Cases
and Edge Cases. atoi and Strong Password are C++ in the source;
rewritten in Python here. Two source problems fixed:
- atoi declares a global `int div`, which clashes with the C library's
  div() (a compile error with Apple clang 17 / libc++).
- Strong Password never initialises mod[3] and counter[3], so its
  answer is whatever was on the stack.
Valid Number is Python in the source; it used str.isdigit(), which is
also True for '²' and '٣', so digits are tested as '0' <= c <= '9'.
"""

INT_MAX, INT_MIN = 2**31 - 1, -(2**31)


def my_atoi(s):
    i, n = 0, len(s)
    while i < n and s[i] == " ":        # 1. skip spaces
        i += 1
    sign = 1
    if i < n and s[i] in "+-":          # 2. at most one sign
        sign = -1 if s[i] == "-" else 1
        i += 1
    x = 0
    while i < n and "0" <= s[i] <= "9":  # 3. digits
        d = ord(s[i]) - ord("0")
        # 4. would 10*x + d pass 2147483647? check BEFORE multiplying
        if x > INT_MAX // 10 or (x == INT_MAX // 10 and d > 7):
            return INT_MAX if sign == 1 else INT_MIN
        x = 10 * x + d
        i += 1
    return sign * x


def atoi_trace(s):
    """Print x before each digit and the overflow check."""
    i = 0
    while i < len(s) and s[i] == " ":
        i += 1
    if i < len(s) and s[i] in "+-":
        i += 1
    x = 0
    while i < len(s) and "0" <= s[i] <= "9":
        d = int(s[i])
        over = x > INT_MAX // 10 or (x == INT_MAX // 10 and d > 7)
        print(f"  d={d}  x={x:<10} {'CLAMP' if over else 'ok'}")
        if over:
            return
        x = 10 * x + d
        i += 1


def is_number(s):
    digits = dot = e = False
    exp_ok = True                       # digits seen after the e
    for i, c in enumerate(s):
        if "0" <= c <= "9":
            digits = exp_ok = True
        elif c == ".":
            if dot or e: return False      # one dot, not in exponent
            dot = True
        elif c in "eE":
            if not digits or e: return False
            e, exp_ok = True, False
        elif c not in "+-" or (i and s[i - 1] not in "eE"):
            return False                # sign only at 0 or after e
    return digits and exp_ok


def runs_of_3(p):
    """Lengths of maximal runs of one character, length >= 3."""
    out, i = [], 0
    while i < len(p):
        j = i
        while j < len(p) and p[j] == p[i]:
            j += 1
        if j - i >= 3:
            out.append(j - i)
        i = j
    return out


def strong_password(p):
    n = len(p)
    missing = 3 - (any(c.islower() for c in p)
                   + any(c.isupper() for c in p)
                   + any(c.isdigit() for c in p))
    runs = runs_of_3(p)                 # lengths of runs >= 3
    if n < 6:
        return max(missing, 6 - n)
    replace = sum(L // 3 for L in runs)
    if n <= 20:
        return max(missing, replace)
    delete = left = n - 20
    one = min(left, sum(L % 3 == 0 for L in runs))
    replace, left = replace - one, left - one      # 1 del saves 1
    two = min(left, 2 * sum(L % 3 == 1 for L in runs)) // 2
    replace, left = replace - two, left - 2 * two  # 2 del save 1
    replace -= left // 3                           # 3 del save 1
    return delete + max(missing, replace)


def strong_brute(max_len=6, alphabet="abA1"):
    """Exact answers by BFS from every strong string (edits are
    symmetric), over short strings from a 4-letter alphabet."""
    from collections import deque
    from itertools import product

    def strong(p):
        return (6 <= len(p) <= 20 and not runs_of_3(p)
                and any(c.islower() for c in p)
                and any(c.isupper() for c in p)
                and any(c.isdigit() for c in p))

    cap = max_len + 2
    dist, q = {}, deque()
    for L in range(cap + 1):
        for t in product(alphabet, repeat=L):
            p = "".join(t)
            if strong(p):
                dist[p] = 0
                q.append(p)
    while q:
        p = q.popleft()
        nxt = set()
        for k in range(len(p) + 1):
            if len(p) < cap:
                nxt.update(p[:k] + c + p[k:] for c in alphabet)
            if k < len(p):
                nxt.add(p[:k] + p[k + 1:])
                nxt.update(p[:k] + c + p[k + 1:] for c in alphabet)
        for t in nxt:
            if t not in dist:
                dist[t] = dist[p] + 1
                q.append(t)
    return {p: d for p, d in dist.items() if len(p) <= max_len}


if __name__ == "__main__":
    import re

    print("atoi:")
    for s in ["42", "   -42", "4193 with words", "words and 987",
              "+-12", "", "  +0 123", "2147483647", "2147483648",
              "-2147483648", "-2147483649", "-91283472332"]:
        print(f"  {s!r:<18} -> {my_atoi(s)}")
    print("trace '-91283472332':")
    atoi_trace("-91283472332")

    ok = ["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3",
          "3e7", "-6e-1", "53.5e93", "-123.456e789"]
    bad = ["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3",
           "95a54e53"]
    assert all(map(is_number, ok)) and not any(map(is_number, bad))
    print("valid number: all 20 source examples agree")
    spec = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?",
                      re.ASCII)
    for s in [".", "+.8", "4.e5", ".e1", "1e+", " 1", "1_0", "inf",
              "nan", "²", "٣"]:
        try:
            f = repr(float(s))
        except ValueError:
            f = "ValueError"
        print(f"  {s!r:<7} is_number={is_number(s)!s:<5}"
              f" regex={bool(spec.fullmatch(s))!s:<5} float={f}")
    print("  '²'.isdigit() =", "²".isdigit(),
          " '٣'.isdigit() =", "٣".isdigit())

    from itertools import product
    fuzz = ["".join(t) for L in range(6)
            for t in product("1.e+-", repeat=L)]
    assert all(is_number(s) == bool(spec.fullmatch(s)) for s in fuzz)
    print(f"is_number == regex on all {len(fuzz)} strings over"
          " '1.e+-' of length <= 5")

    print("strong password:")
    for p in ["a", "aA1", "1337C0d3", "aaa111", "aaaaaaaaaaaaaaaaaaaaa",
              "aaaabbbbbbccccc12345678A", "aaabbbcccdddeeefffggg1A"]:
        print(f"  {p!r:<27} n={len(p):<2} runs={runs_of_3(p)}"
              f" -> {strong_password(p)}")
    exact = strong_brute()
    assert all(strong_password(p) == d for p, d in exact.items())
    print(f"matches BFS on all {len(exact)} strings over 'abA1'"
          " of length <= 6")
