"""LSD radix sort, base 10, with negatives. Python 3.10."""
import math
import random


def radix_sort(a, w=10, trace=False):
    a = list(a)
    if not a:
        return a
    m = max(abs(x) for x in a)
    d = 1
    while w ** d <= m:          # passes = digits of the largest |x|
        d += 1
    for i in range(d):          # least significant digit first
        buckets = [[] for _ in range(w)]
        for x in a:             # -90 // 10 % 10 == 1: Python floors,
            buckets[x // w**i % w].append(x)  # so -90 reads as 910
        a = [x for b in buckets for x in b]   # keep bucket order
        if trace:
            print(f"pass {i} (digit {w**i}):", a)
    neg = [x for x in a if x < 0]   # already in order among themselves
    return neg + [x for x in a if x >= 0]


def comparisons_vs_passes(n, bits, w):
    d = math.ceil(bits / math.log2(w))
    return n * math.log2(n), d, d * (n + w)


if __name__ == "__main__":
    a = [170, 45, 75, -90, 802, 24, 2, 66]
    print(radix_sort(a, trace=True))
    for t in ([0, -5], [-5, 0, 5], [0, 0], []):
        print(t, "->", radix_sort(t))
    random.seed(0)
    for _ in range(2000):
        t = [random.randint(-10**4, 10**4)
             for _ in range(random.randint(0, 30))]
        assert radix_sort(t) == sorted(t)
        assert radix_sort(t, w=256) == sorted(t)
    print("2,000 random lists match sorted() (w = 10 and 256)")
    nlogn, d, work = comparisons_vs_passes(10**6, 32, 256)
    print(f"n=10^6, 32-bit keys: n log2 n = {nlogn:,.0f};"
          f" w=256: {d} passes, d(n+w) = {work:,}")
