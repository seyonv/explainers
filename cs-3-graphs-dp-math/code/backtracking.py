"""Backtracking: Expression Add Operators (source guide, Recursion).

Insert +, - or * between the digits of num so the expression equals
target. Iterative DFS over partial expressions; each state carries
(value so far, last operand) so * can undo the last operand in O(1).
"""
import itertools
import random


def add_operators(num, target):
    n = len(num)
    stack = []                      # (i, expr, value, last)
    for j in range(1, n + 1):       # first operand is num[:j]
        if j > 1 and num[0] == "0":
            break                   # no leading zeros: "05"
        d = int(num[:j])
        stack.append((j, num[:j], d, d))
    found = []
    while stack:
        i, expr, value, last = stack.pop()
        if i == n:                  # leaf: every digit used
            if value == target:
                found.append(expr)
            continue
        for j in range(i + 1, n + 1):
            if j > i + 1 and num[i] == "0":
                break               # prune "05", "007", ...
            s = num[i:j]
            d = int(s)
            stack.append((j, expr + "+" + s, value + d, d))
            stack.append((j, expr + "-" + s, value - d, -d))
            stack.append((j, expr + "*" + s,  # undo last, redo x d
                          value - last + last * d, last * d))
    return found


def add_operators_rec(num, target):
    """Same search, recursive, with an explicit choose / un-choose."""
    n, found, path = len(num), [], []

    def go(i, value, last):
        if i == n:
            if value == target:
                found.append("".join(path))
            return
        for j in range(i + 1, n + 1):
            if j > i + 1 and num[i] == "0":
                break
            s = num[i:j]
            d = int(s)
            if i == 0:
                path.append(s)                      # choose
                go(j, d, d)
                path.pop()                          # un-choose
                continue
            for op, v, l in (("+", value + d, d),
                             ("-", value - d, -d),
                             ("*", value - last + last * d, last * d)):
                path.append(op + s)                 # choose
                go(j, v, l)
                path.pop()                          # un-choose

    go(0, 0, 0)
    return found


def leaves(num):
    """Every complete expression the DFS reaches, with (value, last)."""
    n, out = len(num), []
    stack = [(len(num[:j]), num[:j], int(num[:j]), int(num[:j]))
             for j in range(1, n + 1)
             if not (j > 1 and num[0] == "0")]
    nodes = 0
    while stack:
        i, expr, value, last = stack.pop()
        nodes += 1
        if i == n:
            out.append((expr, value))
            continue
        for j in range(i + 1, n + 1):
            if j > i + 1 and num[i] == "0":
                break
            s = num[i:j]
            d = int(s)
            stack += [(j, expr + "+" + s, value + d, d),
                      (j, expr + "-" + s, value - d, -d),
                      (j, expr + "*" + s,
                       value - last + last * d, last * d)]
    return out, nodes


def brute(num, target):
    """4^(n-1) strings, eval() each, skip operands with leading 0s."""
    out = []
    for ops in itertools.product(["", "+", "-", "*"],
                                 repeat=len(num) - 1):
        expr = num[0] + "".join(o + c for o, c in zip(ops, num[1:]))
        parts = expr.replace("+", " ").replace("-", " ")
        parts = parts.replace("*", " ").split()
        if any(len(p) > 1 and p[0] == "0" for p in parts):
            continue
        if eval(expr) == target:
            out.append(expr)
    return out


if __name__ == "__main__":
    print(add_operators("123", 6))          # ['1*2*3', '1+2+3']
    print(add_operators("105", 5))          # ['1*0+5', '10-5']
    print(add_operators("00", 0))
    print(add_operators("3456237490", 9191))

    out, nodes = leaves("123")
    print("\n'123': %d leaves, %d states popped" % (len(out), nodes))
    for expr, v in sorted(out):
        hit = "  <- hit" if v == 6 else ""
        print("  %-7s = %4d%s" % (expr, v, hit))

    out, nodes = leaves("105")
    print("\n'105': %d leaves (4^2 = 16 without pruning), %d states"
          % (len(out), nodes))

    out, nodes = leaves("3456237490")
    print("'3456237490': %d leaves (4^9 = %d), %d states popped"
          % (len(out), 4 ** 9, nodes))

    z = add_operators("0" * 10, 0)
    print("'0000000000', 0: %d answers (3^9 = %d)" % (len(z), 3 ** 9))

    try:
        eval("1*05")
    except SyntaxError as e:
        print("\neval('1*05') ->", type(e).__name__)

    random.seed(0)
    ok = 0
    for _ in range(500):
        num = "".join(random.choice("0123456789")
                      for _ in range(random.randint(1, 6)))
        t = random.randint(-50, 50)
        a = sorted(add_operators(num, t))
        assert a == sorted(add_operators_rec(num, t))
        assert a == sorted(brute(num, t)), (num, t)
        ok += 1
    print("iterative == recursive == brute force, %d cases" % ok)
