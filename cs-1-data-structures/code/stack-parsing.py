"""Stacks for parsing: Basic Calculator, Longest Valid Parentheses,
Tag Validator.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md,
Stacks. Basic Calculator is Java in the source; rewritten in Python
here with the same idea (on '(' push the running result and the sign
in front of the bracket; on ')' pop them and fold the inner value in).
"""


def calculate(s):
    result, sign, x = 0, 1, 0
    stack = []
    for c in s:
        if c.isdigit():
            x = 10 * x + int(c)          # build multi-digit numbers
        elif c in "+-":
            result += sign * x           # flush the number just read
            sign, x = (1 if c == "+" else -1), 0
        elif c == "(":
            stack.append((result, sign))  # save the outer state
            result, sign = 0, 1          # start a fresh inner sum
        elif c == ")":
            inner = result + sign * x
            outer, before = stack.pop()
            result, x = outer + before * inner, 0
    return result + sign * x


def longest_valid(s):
    stack = [-1]                 # index just before the current run
    best = 0
    for i, c in enumerate(s):
        if c == "(":
            stack.append(i)      # an open bracket waiting for a match
        else:
            stack.pop()          # match it (or pop the old base)
            if stack:
                best = max(best, i - stack[-1])
            else:
                stack.append(i)  # unmatched ')': new base
    return best


def is_valid_tag(code):
    """Tag Validator: a stack of open TAG_NAMEs (source version)."""
    def parse_tag(i, k):
        j = i + k
        i = code.find(">", j)
        if i in (-1, j) or i - j > 9:
            return None, -1
        name = code[j:i]
        if not all(ch.isupper() for ch in name):
            return None, -1
        return name, i

    i, names = 0, []
    while i < len(code):
        if i and not names:
            return False         # text outside the outer tag
        if code.startswith("<![CDATA[", i):
            i = code.find("]]>", i + 9)
            if i == -1:
                return False
            i += 2
        elif code.startswith("</", i):
            name, i = parse_tag(i, 2)
            if not names or names.pop() != name:
                return False
        elif code.startswith("<", i):
            name, i = parse_tag(i, 1)
            if not name:
                return False
            names.append(name)
        i += 1
    return not names


def trace_calculate(s):
    """One line per non-digit character, plus the final flush."""
    result, sign, x = 0, 1, 0
    stack = []
    for c in s + "$":
        if c.isdigit():
            x = 10 * x + int(c)
            continue
        read = x
        if c in "+-":
            result += sign * x
            sign, x = (1 if c == "+" else -1), 0
            what = f"result += {read}"
        elif c == "(":
            stack.append((result, sign))
            what = f"push ({result}, {sign:+d})"
            result, sign = 0, 1
        elif c == ")":
            inner = result + sign * x
            outer, before = stack.pop()
            what = f"pop: {outer} {before:+d}*{inner}"
            result, x = outer + before * inner, 0
        else:
            result, what = result + sign * x, "end"
        print(f"{c}  x={read:<2} {what:<20} result={result:<3}"
              f" sign={sign:+d} stack={stack}")
    return result


def trace_longest(s):
    stack, best = [-1], 0
    for i, c in enumerate(s):
        if c == "(":
            stack.append(i)
            note = "push"
        else:
            stack.pop()
            if stack:
                best = max(best, i - stack[-1])
                note = f"len {i} - {stack[-1]} = {i - stack[-1]}"
            else:
                stack.append(i)
                note = "empty: new base"
        print(f"i={i} {c}  {note:<16} stack={stack} best={best}")
    return best


def brute_longest(s):
    def ok(t):
        depth = 0
        for c in t:
            depth += 1 if c == "(" else -1
            if depth < 0:
                return False
        return depth == 0
    n = len(s)
    return max([j - i for i in range(n) for j in range(i, n + 1)
                if ok(s[i:j])] or [0])


if __name__ == "__main__":
    import random

    expr = "(1+(4+5+2)-3)+(6+8)"
    print(calculate(expr))                      # 23
    trace_calculate(expr)
    print(calculate("- (3 + (4 - 5))"), calculate("2-(5-6)"))

    print(longest_valid(")()())"))             # 4
    trace_longest(")()())")

    print(is_valid_tag("<DIV>This is <![CDATA[<a>]]></DIV>"),
          is_valid_tag("<A><B></A></B>"))      # True False

    rng = random.Random(0)
    for _ in range(1000):
        s = "".join(rng.choice("()") for _ in range(rng.randint(0, 12)))
        assert longest_valid(s) == brute_longest(s)
    for _ in range(1000):
        e = str(rng.randint(0, 99))
        for _ in range(rng.randint(0, 6)):
            op = rng.choice("+-")
            a, b = rng.randint(0, 99), rng.randint(0, 9)
            e = (f"({e}){op}{a}" if rng.random() < .3
                 else f"{e}{op}({a}-{b})")
        assert calculate(e) == eval(e), e      # eval only as a checker
    print("1000 + 1000 random cases match brute force / eval")
