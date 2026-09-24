"""Where in-memory speed comes from: count cache lines per lookup.

A cache miss costs a whole 64-byte line from DRAM (~100 ns), so a
lookup's cost in memory is roughly the number of distinct lines it
touches. This file counts them, with a cold cache, for four ways of
finding one of 10**7 8-byte keys:

  1. binary search over a sorted array
  2. a disk-style B+tree: 4 KB nodes of 16-byte entries (fan-out 256)
  3. a cache-conscious tree: one 64-byte line per node, 8 keys
  4. a linear-probing hash table at load 0.5

It also redoes the Shore arithmetic from the cheat-sheet.
"""
import math
import random

LINE, KEY, N = 64, 8, 10**7


def probes(n, p):
    """Indices bisect_right visits to find slot p in n keys."""
    lo, hi, seen = 0, n, []
    while lo < hi:
        mid = (lo + hi) // 2
        seen.append(mid)
        if p < mid: hi = mid          # key p sits at index p
        else: lo = mid + 1
    return seen


def lines(idxs, width):            # distinct 64-byte lines
    return len({i * width // LINE for i in idxs})


def sorted_array(r):
    return lines(probes(N, r), KEY)


def page_btree(r, fan=256, entry=16):
    total, n = 0, N
    sizes = []                     # entries per node, leaf level up
    while n > 1:
        sizes.append(n)
        n = math.ceil(n / fan)
    for lvl, count in enumerate(sizes):
        pos = r // fan**lvl        # our entry's index on this level
        node, p = divmod(pos, fan)
        size = min(fan, count - node * fan)
        total += lines(probes(size, p), entry)
    return total


def line_tree_levels(keys=LINE // KEY):
    n, levels = -(-N // keys), 1       # leaves: 8 keys per line
    while n > 1:                      # inner nodes: 9 children
        n, levels = -(-n // (keys + 1)), levels + 1
    return levels                      # one line per level


def hash_table(slots=2**20, load=0.5, seed=0):
    rnd, table, runs = random.Random(seed), [None] * slots, []
    for k in range(int(slots * load)):
        i, run = rnd.randrange(slots), []
        while table[i] is not None:
            run.append(i)
            i = (i + 1) % slots
        table[i] = k
        runs.append(run + [i])     # a later lookup walks this run
    return sum(lines(r, KEY) for r in runs) / len(runs)


def shore(start=100.0):
    for cut in (0.80, 0.88):
        left = start * (1 - cut)
        final = left * (1 - 0.75)
        print(f"  remove {cut:.0%}: {start:.0f} -> {left:.0f}"
              f" -> x 1/4 = {final:.0f}   ({start / final:.0f}x)")


if __name__ == "__main__":
    rnd = random.Random(0)
    targets = [rnd.randrange(N) for _ in range(10_000)]
    avg = lambda f: sum(map(f, targets)) / len(targets)
    rows = [("sorted array + bisect", avg(sorted_array)),
            ("B+tree, 4 KB nodes", avg(page_btree)),
            ("tree of 64-byte nodes", line_tree_levels()),
            ("hash table, load 0.5", hash_table())]
    print(f"cache lines per lookup, N = {N:,}, cold cache")
    for name, n in rows:
        print(f"  {name:22} {n:5.2f} lines  ~{n * 100:5.0f} ns")
    k = avg(lambda r: len(probes(N, r)))
    print(f"binary search: {k:.2f} probes"
          f" (log2 N = {math.log2(N):.2f})")
    print("Shore B-tree time, 100 units:")
    shore()
