"""An algorithm beats a faster computer (CLRS 1.2, via the source guide).

Computer A: 10**10 instructions/s running insertion sort, 2 n**2 steps.
Computer B: 10**7 instructions/s running merge sort, 50 n lg n steps.
1. Time both for n = 10**7 (the source's example) and other sizes.
2. Find the crossover n where B starts winning.
3. Proebsting's law: years of hardware vs compiler gains for 1000x.
4. Count real comparisons: insertion sort vs merge sort.
Runs under python3 (3.10). Standard library only; no timing.
"""
import math
import random

A_SPEED, B_SPEED = 10**10, 10**7   # instructions per second

def time_a(n):                    # insertion sort on A
    return 2 * n * n / A_SPEED

def time_b(n):                    # merge sort on B
    return 50 * n * math.log2(n) / B_SPEED

def crossover():
    """Smallest n >= 2 where slow B with merge sort beats fast A."""
    lo, hi = 2, 10**7              # A wins at 2, B wins at 1e7
    while lo < hi:
        mid = (lo + hi) // 2
        if time_b(mid) < time_a(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def years_for(factor, growth):
    """Years of compound growth at `growth` per year to reach factor."""
    return math.log(factor) / math.log(1 + growth)


def insertion_comparisons(xs):
    a, count = list(xs), 0
    for i in range(1, len(a)):
        x, j = a[i], i - 1
        while j >= 0:
            count += 1
            if a[j] <= x:
                break
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x
    return count


def merge_comparisons(xs):
    if len(xs) <= 1:
        return list(xs), 0
    mid = len(xs) // 2
    left, cl = merge_comparisons(xs[:mid])
    right, cr = merge_comparisons(xs[mid:])
    out, i, j, count = [], 0, 0, cl + cr
    while i < len(left) and j < len(right):
        count += 1
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    return out + left[i:] + right[j:], count


if __name__ == "__main__":
    print(time_a(10**7), round(time_b(10**7), 1), crossover())
    n = 10**7
    ta, tb = time_a(n), time_b(n)
    print(f"n lg n       = {n * math.log2(n):.4e}")
    print(f"A: 2n^2/1e10 = {ta:,.0f} s = {ta / 3600:.2f} h")
    print(f"B: 50nlgn/1e7 = {tb:,.1f} s = {tb / 60:.1f} min")
    print(f"A is {A_SPEED // B_SPEED}x faster; B finishes"
          f" {ta / tb:.1f}x sooner")
    print()
    for k in (3, 4, 5, 6, 7, 8, 9):
        m = 10**k
        a, b = time_a(m), time_b(m)
        print(f"n=1e{k}: A {a:>13,.2f} s  B {b:>9,.2f} s"
              f"  A/B {a / b:>8,.3f}")
    c = crossover()
    print(f"\ncrossover: B wins from n = {c:,}"
          f" (both about {time_a(c):.1f} s)")
    print()
    for name, g in (("hardware 60%/yr", 0.60),
                    ("compiler 4%/yr", 0.04)):
        print(f"{name}: 2x in {years_for(2, g):.1f} yr,"
              f" 1000x in {years_for(1000, g):.1f} yr")
    print()
    rng = random.Random(1)
    for m in (100, 1000, 4000):
        xs = [rng.random() for _ in range(m)]
        ins = insertion_comparisons(xs)
        srt, mer = merge_comparisons(xs)
        assert srt == sorted(xs)
        print(f"n={m:>4}: insertion {ins:>9,}"
              f" (n^2/4 {m * m // 4:>9,})  merge {mer:>6,}"
              f" (n lg n {m * math.log2(m):>6,.0f})")
