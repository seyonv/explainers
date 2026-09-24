"""Permutations and combinations: the k-th permutation via the
factorial number system, and n-choose-k without huge factorials.

Run: python3 factorial-number-system.py   (Python 3.10, stdlib only)
"""
import itertools
import math


def kth_permutation(n, k, trace=False):
    """k-th (1-based) permutation of 1..n in lexicographic order."""
    k -= 1                          # 0-based rank
    pool = list(range(1, n + 1))    # digits not used yet, sorted
    out = []
    for m in range(n - 1, -1, -1):  # block sizes (n-1)!, ..., 0!
        d, k = divmod(k, math.factorial(m))
        if trace:
            print(f"  {m}! = {math.factorial(m)}  digit {d}  "
                  f"pool {pool}  pick {pool[d]}  rest {k}")
        out.append(str(pool.pop(d)))  # d-th smallest unused
    return "".join(out)


def permutation_rank(perm):
    """Inverse: 1-based rank of a permutation string (Lehmer code)."""
    pool = sorted(perm)
    rank = 0
    for i, ch in enumerate(perm):
        d = pool.index(ch)
        rank += d * math.factorial(len(perm) - 1 - i)
        pool.pop(d)
    return rank + 1


def comb(n, k, trace=False):
    """n choose k by the multiplicative formula."""
    if not 0 <= k <= n:
        return 0
    k = min(k, n - k)               # C(n, k) = C(n, n - k)
    r = 1
    for i in range(1, k + 1):
        r = r * (n - k + i) // i    # r is C(n-k+i, i): exact
        if trace:
            print(f"  i={i}: r = C({n - k + i},{i}) = {r:,}")
    return r


def pascal(rows):
    """Pascal's triangle: C(n,k) = C(n-1,k-1) + C(n-1,k)."""
    tri = [[1]]
    for _ in range(rows - 1):
        prev = tri[-1]
        tri.append([1] + [a + b for a, b in zip(prev, prev[1:])] + [1])
    return tri


if __name__ == "__main__":
    print("kth_permutation(4, 9):")
    print(" ", kth_permutation(4, 9, trace=True))
    print("permutation_rank('2314') =", permutation_rank("2314"))

    # check against brute force for every n <= 6 and every k
    for n in range(1, 7):
        perms = ["".join(p) for p in
                 itertools.permutations("123456"[:n])]
        for k, p in enumerate(perms, 1):
            assert kth_permutation(n, k) == p
            assert permutation_rank(p) == k
    print("all 873 permutations for n <= 6 match itertools")

    print("comb(52, 5):")
    print(" ", comb(52, 5, trace=True))
    print("math.comb(52, 5) =", f"{math.comb(52, 5):,}")
    for n in range(80):
        for k in range(n + 1):
            assert comb(n, k) == math.comb(n, k)
    print("comb matches math.comb for all n < 80")

    for row in pascal(7):
        print(" ", row)

    big = 2**63 - 1
    print("first n with n! > 2^63-1:",
          next(n for n in range(99) if math.factorial(n) > big))
    print("52! has", len(str(math.factorial(52))), "digits")

    def peak(n, k):                 # largest r * (n-k+i) in comb()
        r, top = 1, 0
        for i in range(1, k + 1):
            top = max(top, r * (n - k + i))
            r = r * (n - k + i) // i
        return top

    print("largest intermediate in comb(52, 5):", f"{peak(52, 5):,}")
    print("C(n, n//2) in int64: intermediate overflows at n =",
          next(n for n in range(99) if peak(n, n // 2) > big),
          "| result at n =",
          next(n for n in range(99) if math.comb(n, n // 2) > big))
