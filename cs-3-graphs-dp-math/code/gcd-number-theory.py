"""GCD and number theory: Euclid, Reaching Points, Max Points on a
Line, Nth Magical Number. Run with python3 (3.10+)."""
import math
import random
from collections import Counter
from functools import lru_cache

MOD = 10**9 + 7


def gcd(a, b):
    while b:                     # gcd(a, b) = gcd(b, a mod b)
        a, b = b, a % b
    return a                     # gcd(a, 0) = a


def euclid_trace(a, b):
    while b:
        q, r = divmod(a, b)
        print(f"  {a} = {q} * {b} + {r}")
        a, b = b, r
    return a


def reaching_points(sx, sy, tx, ty):
    # walk backwards: the larger coordinate must have been the sum,
    # so subtract the smaller one as many times as possible at once
    while sx < tx and sy < ty:
        if tx > ty:
            tx %= ty
        else:
            ty %= tx
    if sx == tx:                 # only y can still shrink by sx steps
        return sy <= ty and (ty - sy) % sx == 0
    if sy == ty:
        return sx <= tx and (tx - sx) % sy == 0
    return False


def direction(dx, dy):
    """Reduce (dx, dy) to one exact key per line direction."""
    g = math.gcd(dx, dy)         # always >= 0 in Python
    dx, dy = dx // g, dy // g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy        # (-1, 1) and (1, -1): same line
    return dx, dy


def max_points(points):
    best = 0
    for i, (x0, y0) in enumerate(points):
        same, dirs = 1, Counter()
        for x, y in points[i + 1:]:
            if (x, y) == (x0, y0):
                same += 1        # duplicates lie on every line
            else:
                dirs[direction(x - x0, y - y0)] += 1
        best = max(best, same + max(dirs.values(), default=0))
    return best


def nth_magical(n, a, b):
    lcm = a * b // math.gcd(a, b)

    def count(x):                # multiples of a or b in [1, x]
        return x // a + x // b - x // lcm

    lo, hi = 1, n * min(a, b)    # the answer is at most n * min(a, b)
    while lo < hi:               # smallest x with count(x) >= n
        mid = (lo + hi) // 2
        if count(mid) >= n:
            hi = mid
        else:
            lo = mid + 1
    return lo % MOD


# --- the source's versions, kept for the cross-checks below ---------

def src_reaching_points(sx, sy, tx, ty):
    while sx < tx and sy < ty:
        tx, ty = tx % ty, ty % tx
    return all([sx == tx, sy <= ty, not (ty - sy) % sx]) or all(
        [sy == ty, sx <= tx, not (tx - sx) % sy])


def src_nth_magical(n, a, b):
    lcm = math.lcm(a, b)
    div, m = divmod(n, lcm // a + lcm // b - 1)
    magical_number = m / (1 / a + 1 / b)
    return (div * lcm + min(math.ceil(magical_number / a) * a,
                            math.ceil(magical_number / b) * b)) % MOD


def brute_reach(sx, sy, tx, ty):
    @lru_cache(None)
    def go(x, y):
        if x > tx or y > ty:
            return False
        return (x, y) == (tx, ty) or go(x, x + y) or go(x + y, y)
    return go(sx, sy)


def brute_max_points(pts):
    n, best = len(pts), min(len(pts), 1)
    for i in range(n):
        for j in range(n):
            if pts[i] == pts[j]:
                best = max(best, pts.count(pts[i]))
                continue
            (x1, y1), (x2, y2) = pts[i], pts[j]
            best = max(best, sum(
                (x2 - x1) * (y - y1) == (y2 - y1) * (x - x1)
                for x, y in pts))
    return best


if __name__ == "__main__":
    print("gcd(1071, 462):")
    g = euclid_trace(1071, 462)
    print("  gcd =", g, " lcm =", 1071 * 462 // g)

    print("reaching (1,1) -> (3,5):", reaching_points(1, 1, 3, 5))
    print("reaching (1,1) -> (2,2):", reaching_points(1, 1, 2, 2))
    print("reaching (1,1) -> (10**9, 3):",
          reaching_points(1, 1, 10**9, 3))

    pts = [(1, 1), (3, 2), (5, 3), (4, 1), (2, 3), (1, 4)]
    x0, y0 = pts[1]
    for x, y in pts[2:]:
        print(f"  from (3,2) to ({x},{y}):",
              (x - x0, y - y0), "->", direction(x - x0, y - y0))
    print("max points:", max_points(pts))
    p, q = (94911151, 94911150), (94911152, 94911151)
    print("float slopes equal:", p[1] / p[0] == q[1] / q[0],
          "| exact keys:", direction(*p), direction(*q))

    print("count(x), a=2 b=3:",
          [x // 2 + x // 3 - x // 6 for x in range(1, 8)])
    print("nth_magical(4, 2, 3):", nth_magical(4, 2, 3))
    print("nth_magical(10**9, 40000, 39999):",
          nth_magical(10**9, 40000, 39999))

    rng = random.Random(0)
    for a in range(1, 7):
        for b in range(1, 7):
            for tx in range(1, 30):
                for ty in range(1, 30):
                    want = brute_reach(a, b, tx, ty)
                    assert reaching_points(a, b, tx, ty) == want
                    assert src_reaching_points(a, b, tx, ty) == want
    for _ in range(3000):
        pts = [(rng.randint(-3, 3), rng.randint(-3, 3))
               for _ in range(rng.randint(1, 8))]
        assert max_points(pts) == brute_max_points(pts)
    for _ in range(100_000):
        n = rng.randint(1, 10**9)
        a, b = rng.randint(2, 40000), rng.randint(2, 40000)
        assert nth_magical(n, a, b) == src_nth_magical(n, a, b)
    print("checks: reaching points (brute force), max points "
          "(3,000 random sets), nth magical (100,000 vs source): ok")
