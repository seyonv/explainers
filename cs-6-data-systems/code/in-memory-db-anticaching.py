"""Anti-caching (H-Store): evict cold tuples in blocks, and when a
transaction touches one, pre-pass, abort, fetch, merge, restart.
Runs under python3 (3.10), standard library only.

Toy sizes (illustrative): room for 4 tuples in memory, 2-tuple
eviction blocks, one table of orders keyed 1..7.
Paper: DeBrabant et al., "Anti-Caching", VLDB 2013.
"""
from collections import OrderedDict


class AntiCache:
    def __init__(self, cap, block):
        self.mem = OrderedDict()   # LRU chain: head = coldest
        self.evicted = {}          # Evicted Table: key -> block id
        self.disk = []             # blocks, each written once
        self.cap, self.block = cap, block
        self.log = []

    def insert(self, k, v):
        self.mem[k] = v            # new tuples join the hot tail
        self.shrink()

    def shrink(self):              # the eviction "transaction"
        while len(self.mem) > self.cap:
            blk = {}
            while len(blk) < self.block:
                k, v = self.mem.popitem(last=False)  # pop head
                blk[k] = v
                self.evicted[k] = len(self.disk)
            self.disk.append(blk)  # one sequential write
            self.log.append(f"evict {list(blk)} -> block "
                            f"{len(self.disk) - 1}")

    def run(self, name, keys):
        # pre-pass: run the reads, record evicted keys, change nothing
        miss = [k for k in keys if k in self.evicted]
        for k in keys:
            if k in self.mem:
                self.mem.move_to_end(k)   # O(1): doubly linked
        if miss:
            blocks = sorted({self.evicted[k] for k in miss})
            self.log.append(f"{name} pre-pass: {miss} evicted, "
                            f"abort, fetch blocks {blocks}")
            for k in miss:                # tuple-merge: only these
                b = self.evicted.pop(k)
                self.mem[k] = self.disk[b][k]
            self.log.append(f"{name} restart")
        out = [self.mem[k] for k in keys]  # all in memory now
        self.log.append(f"{name} commit {out}")
        self.shrink()
        return out


def chain(db):
    return list(db.mem), dict(db.evicted)


if __name__ == "__main__":
    db = AntiCache(cap=4, block=2)
    for k in range(1, 7):
        db.insert(k, 10 * k)          # price = 10 x id (illustrative)
    print("after inserts 1..6:", *chain(db))
    db.run("T1", [3])
    print("after T1 reads 3:  ", *chain(db))
    db.insert(7, 70)
    print("after insert 7:    ", *chain(db))
    db.run("T2", [3, 1, 4])
    print("after T2:          ", *chain(db))
    print("disk blocks:", db.disk)
    print()
    print(*db.log, sep="\n")

    # Why a doubly-linked chain: unlinking a tuple needs its
    # predecessor. With only next pointers you walk from the cold
    # head; hot tuples live near the tail, so the walk is ~n hops.
    n = 100_000
    pos = n - 10                  # a hot tuple, 10 from the tail
    print()
    print(f"singly linked: {pos:,} hops to find the predecessor; "
          f"doubly linked: 1")

    # Block size: bytes fetched to bring back one 1,000-byte YCSB
    # tuple (10 columns x 100 B) with the paper's block sizes.
    for kb in (256, 1024):
        per = kb * 1024 // 1000
        print(f"{kb:>5} KB block: fetch {kb * 1024:,} B for "
              f"1,000 B, block holds {per:,} tuples")
    print("eviction writes: 5 x 1 MB =", 5 * 1024, "KB;",
          "20 x 256 KB =", 20 * 256, "KB")
