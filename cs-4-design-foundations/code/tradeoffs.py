"""Everything is a trade-off: put numbers on both sides.

1. Space vs time: storing a set of IPv4 addresses three ways.
2. DP spends memory to save time: rod cutting (CLRS 15.1 prices).
3. Interviewing less vs hiring better: the secretary 37% rule.
4. The source's sorting footnote, corrected with comparison counts.
Runs under python3 (3.10). Standard library only.
"""
import array
import math
import random
import sys


# 1. Space vs time --------------------------------------------------
class Bitmap:
    """One bit per possible address: 2**32 bits = 512 MiB."""

    def __init__(self, nbits):
        self.bits = bytearray((nbits + 7) // 8)

    def add(self, x):
        self.bits[x >> 3] |= 1 << (x & 7)

    def __contains__(self, x):
        return self.bits[x >> 3] >> (x & 7) & 1 == 1


def bytes_per_address(n, seed=0):
    """Python set vs sorted array('I'), measured with getsizeof."""
    rng = random.Random(seed)
    xs = rng.sample(range(2**32), n)
    s = set(xs)
    in_set = sys.getsizeof(s) + sum(sys.getsizeof(x) for x in xs)
    arr = array.array("I", sorted(xs))
    return in_set / n, sys.getsizeof(arr) / n


# 2. Rod cutting: memory for time -----------------------------------
PRICES = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30]  # length 1..10


def rod_cut(p, n):
    r = [0] * (n + 1)            # r[j] = best revenue for length j
    first = [0] * (n + 1)        # first piece in that best plan
    for j in range(1, n + 1):
        for i in range(1, j + 1):          # first piece has length i
            if p[i - 1] + r[j - i] > r[j]:
                r[j], first[j] = p[i - 1] + r[j - i], i
    return r, first


def pieces(first, n):
    out = []
    while n:
        out.append(first[n])
        n -= first[n]
    return out


def naive_calls(n):
    """Calls made by plain recursion: 1 + sum of calls for 0..n-1."""
    return 1 + sum(naive_calls(j) for j in range(n))


# 3. Secretary problem ----------------------------------------------
def p_best(n, k):
    """P(hire the very best) if you skip k, then take the first
    candidate better than all of them."""
    if k == 0:
        return 1 / n
    return k / n * sum(1 / (i - 1) for i in range(k + 1, n + 1))


def best_skip(n):
    return max(range(n), key=lambda k: p_best(n, k))


def simulate(n, k, trials, seed=0):
    rng = random.Random(seed)
    wins = 0
    for _ in range(trials):
        ranks = list(range(n))           # n-1 is the best
        rng.shuffle(ranks)
        bar = max(ranks[:k], default=-1)
        pick = next((x for x in ranks[k:] if x > bar), ranks[-1])
        wins += pick == n - 1
    return wins / trials


# 4. The source says quicksort is best on nearly sorted data ------
def insertion_compares(a):
    a, c = list(a), 0
    for j in range(1, len(a)):
        x, i = a[j], j - 1
        while i >= 0:
            c += 1
            if a[i] <= x:
                break
            a[i + 1] = a[i]
            i -= 1
        a[i + 1] = x
    return c


def quicksort_compares(a):
    """Lomuto partition, last element as pivot, iterative."""
    a, c, todo = list(a), 0, [(0, len(a) - 1)]
    while todo:
        lo, hi = todo.pop()
        if lo >= hi:
            continue
        pivot, k = a[hi], lo
        for j in range(lo, hi):
            c += 1
            if a[j] < pivot:
                a[k], a[j] = a[j], a[k]
                k += 1
        a[k], a[hi] = a[hi], a[k]
        todo += [(lo, k - 1), (k + 1, hi)]
    return c


def nearly_sorted(n, swaps, seed=0):
    rng = random.Random(seed)
    a = list(range(n))
    for _ in range(swaps):
        i = rng.randrange(n - 1)
        a[i], a[i + 1] = a[i + 1], a[i]
    return a


if __name__ == "__main__":
    print("1. IPv4 membership: memory per representation")
    full = 2**32 // 8
    print(f"   bitmap: 2**32 bits = {full:,} B"
          f" = {full / 2**20:.0f} MiB")
    per_set, per_arr = bytes_per_address(10**6)
    print(f"   set, n=10**6: {per_set:.1f} B/address")
    print(f"   sorted array('I'): {per_arr:.2f} B/address")
    print(f"   break-even vs bitmap: set {full / per_set:,.0f},"
          f" array {full // 4:,} addresses")
    bm = Bitmap(2**32)                 # really allocates 512 MiB
    ip = 192 << 24 | 168 << 16 | 1 << 8 | 1      # 192.168.1.1
    bm.add(ip)
    print(f"   len(bm.bits) = {len(bm.bits):,}")
    print("   192.168.1.1 in bm:", ip in bm, "| 7 in bm:", 7 in bm)

    print("2. Rod cutting")
    r, first = rod_cut(PRICES, 10)
    print("   r =", r)
    print("   first =", first)
    print("   n=7:", pieces(first, 7), "->", r[7])
    print("   naive calls n=10:", naive_calls(10), "| DP steps:",
          10 * 11 // 2)

    print("3. Secretary")
    for n in (5, 10, 100, 1000):
        k = best_skip(n)
        print(f"   n={n:<5} skip {k:<4} P(best) = {p_best(n, k):.4f}")
    print(f"   1/e = {1 / math.e:.4f}")
    sim = simulate(100, 37, 10**5)
    print(f"   simulated n=100, skip 37, 10**5 runs: {sim:.4f}")

    print("4. Sorting 2,000 nearly sorted items (10 adjacent swaps)")
    a = nearly_sorted(2000, 10)
    print(f"   insertion sort compares: {insertion_compares(a):,}")
    print(f"   quicksort (last pivot) compares: "
          f"{quicksort_compares(a):,}")
