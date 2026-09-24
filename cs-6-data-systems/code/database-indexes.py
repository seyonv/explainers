"""B-trees and database indexes: bulk load, point lookup, range scan,
and a bitmap AND. Runs under python3 (3.10), standard library only.

Running example (illustrative sizing): orders, 10 million rows,
4 KB pages, a secondary index on created_at with 16-byte entries
(8-byte key + 8-byte row pointer), so 4096 / 16 = 256 per page.
"""
import bisect
import math

PAGE, ENTRY, N = 4096, 16, 10**7


def bulk_load(keys, fanout):
    """Build a B-tree bottom-up from sorted keys (no inserts)."""
    nodes = [keys[i:i + fanout] for i in range(0, len(keys), fanout)]
    levels = [nodes]                       # levels[0] = leaves
    while len(nodes) > 1:
        firsts = [n[0] for n in nodes]     # separator = child's 1st key
        nodes = [firsts[i:i + fanout]
                 for i in range(0, len(firsts), fanout)]
        levels.append(nodes)
    return levels[::-1]                    # root level first


def descend(tree, key, fanout):
    """Walk root -> leaf. Return (leaf number, pages read)."""
    i = 0
    for level in tree[:-1]:                # branch levels
        j = bisect.bisect_right(level[i], key) - 1
        i = i * fanout + max(j, 0)         # full nodes: child index
    return i, len(tree)


def search(tree, key, fanout):
    i = 0
    for level in tree[:-1]:                # branch levels: cached
        j = bisect.bisect_right(level[i], key) - 1
        i = i * fanout + max(j, 0)         # full nodes: child index
    leaf = tree[-1][i]                     # the one disk read
    j = bisect.bisect_left(leaf, key)
    return j < len(leaf) and leaf[j] == key, len(tree)


def range_scan(tree, lo, hi, fanout):
    """Keys in [lo, hi]: descend once, then walk the leaves."""
    leaf, reads = descend(tree, lo, fanout)
    found, leaves = 0, tree[-1]
    while leaf < len(leaves) and leaves[leaf][0] <= hi:
        node = leaves[leaf]
        found += (bisect.bisect_right(node, hi)
                  - bisect.bisect_left(node, lo))
        leaf += 1
        reads += 1
    return found, reads - 1                # first leaf counted twice


def bitmaps(column):
    """One int per distinct value; bit r set if row r has it."""
    maps = {}
    for r, v in enumerate(column):
        maps[v] = maps.get(v, 0) | (1 << r)
    return maps


def bits(x, n=16):
    return format(x, f"0{n}b")[::-1]       # row 0 on the left


if __name__ == "__main__":
    fanout = PAGE // ENTRY
    print(f"fan-out = {PAGE} // {ENTRY} = {fanout}")
    keys = range(N)            # created_at: one order per second
    tree = bulk_load(keys, fanout)
    print("pages per level, root first:",
          [len(level) for level in tree])
    print(f"log_{fanout}({N:,}) = {math.log(N, fanout):.3f}")
    print("search 7,654,321:", search(tree, 7_654_321, fanout))
    print("search 10**7 (absent):", search(tree, N, fanout))
    hour = range_scan(tree, 7_200_000, 7_203_599, fanout)
    print("one hour, 3,600 keys: (found, pages read) =", hour)
    half = bulk_load(keys, fanout // 2)
    print("half-full pages, fan-out 128:",
          [len(level) for level in half])
    print("full scan of the table:", N // (PAGE // 100), "pages")

    status = ("paid shipped shipped paid refund shipped paid "
              "shipped shipped paid shipped refund shipped paid "
              "shipped paid").split()
    region = ("EU US EU EU US EU AS EU US EU EU US AS EU EU "
              "US").split()
    s, r = bitmaps(status), bitmaps(region)
    hit = s["shipped"] & r["EU"]
    print("shipped =", bits(s["shipped"]))
    print("EU      =", bits(r["EU"]))
    print("AND     =", bits(hit),
          "rows", [i for i in range(16) if hit >> i & 1])
