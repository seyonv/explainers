"""Consistent hashing: hash mod N vs a hash ring with virtual nodes.

Adds a 5th node to 4 and counts how many of 100,000 keys change
owner, then measures how evenly the ring spreads load with 1 vs
100 virtual nodes per node. Deterministic: md5 is fixed, no seed.
"""
import bisect
import collections
import hashlib
import statistics

RING = 2 ** 32                   # positions 0 .. 2^32 - 1


def h(s):
    """32-bit hash of a string: first 4 bytes of md5."""
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "big")


def mod_owner(key, nodes):
    return nodes[h(key) % len(nodes)]


class Ring:
    def __init__(self, nodes, vnodes=1):
        pts = sorted((h(f"{n}#{i}"), n)
                     for n in nodes for i in range(vnodes))
        self.pos = [p for p, _ in pts]      # sorted ring positions
        self.own = [n for _, n in pts]      # node at each position

    def owner(self, key):
        # first position clockwise from the key; wrap past the top
        i = bisect.bisect(self.pos, h(key)) % len(self.pos)
        return self.own[i]


def moved(before, after, keys):
    return sum(before(k) != after(k) for k in keys) / len(keys)


def spread(nodes, vnodes, keys):
    r = Ring(nodes, vnodes)
    load = {n: 0 for n in nodes}
    for k in keys:
        load[r.owner(k)] += 1
    share = [load[n] / len(keys) for n in nodes]
    return share, statistics.pstdev(share)


def deg(x):
    return x * 360 / RING


if __name__ == "__main__":
    old, new = list("ABCD"), list("ABCDE")

    print("hand trace (ring: 1 vnode per node)")
    for n in new:
        x = h(n + "#0")
        print(f"  node {n}#0  h={x:>10}  {deg(x):5.1f} deg")
    r4, r5 = Ring(old), Ring(new)
    for k in [f"img-{i}" for i in range(1, 7)]:
        print(f"  {k}  h={h(k):>10}  {deg(h(k)):5.1f} deg"
              f"  mod {mod_owner(k, old)}->{mod_owner(k, new)}"
              f"  ring {r4.owner(k)}->{r5.owner(k)}")

    keys = [f"key-{i}" for i in range(100_000)]
    print("\nadd a 5th node to 4, share of 100,000 keys moved")
    m = moved(lambda k: mod_owner(k, old),
              lambda k: mod_owner(k, new), keys)
    print(f"  hash mod N          {m:.2%}   (ideal 4/5 = 80%)")
    for v in (1, 100):
        a, b = Ring(old, v), Ring(new, v)
        m = moved(a.owner, b.owner, keys)
        movers = [k for k in keys if a.owner(k) != b.owner(k)]
        to_e = all(b.owner(k) == "E" for k in movers)
        donors = collections.Counter(a.owner(k) for k in movers)
        print(f"  ring, {v:>3} vnodes    {m:.2%}   (ideal 1/5 = 20%)")
        print(f"      all went to E: {to_e}; taken from"
              f" {dict(sorted(donors.items()))}")

    print("\nload per node on the 5-node ring (ideal 20% each)")
    for v in (1, 100):
        share, sd = spread(new, v, keys)
        cells = " ".join(f"{s:6.2%}" for s in share)
        print(f"  {v:>3} vnodes  {cells}  sd {sd:.2%}"
              f"  max/mean {max(share) * 5:.2f}")
