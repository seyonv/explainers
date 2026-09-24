"""A toy LSM tree: memtable -> flush to sorted runs -> compaction.

Keys are order ids, values are prices. A tombstone (None) marks a
delete. Each run carries a small Bloom filter so a read can skip it.
Run with python3 (3.10+).
"""
import hashlib
import math

TOMB = None          # tombstone: "this key was deleted"
MEM_LIMIT = 3        # flush after 3 keys (illustrative)
L0_LIMIT = 3         # compact when L0 holds 3 runs (illustrative)


class Bloom:
    def __init__(self, keys, bits_per_key=10):
        self.m = max(8, bits_per_key * len(keys))
        self.k = max(1, round(bits_per_key * math.log(2)))  # 7
        self.bits = 0
        for key in keys:
            for p in self._probes(key):
                self.bits |= 1 << p

    def _probes(self, key):
        for i in range(self.k):
            h = hashlib.sha256(f"{i}:{key}".encode()).digest()
            yield int.from_bytes(h[:8], "big") % self.m

    def might_have(self, key):
        return all(self.bits >> p & 1 for p in self._probes(key))


class Run:                            # an immutable sorted file
    def __init__(self, items):
        self.items = sorted(items)    # [(key, value)], one per key
        self.index = dict(self.items)
        self.bloom = Bloom([k for k, _ in self.items])


class LSM:
    def __init__(self):
        self.mem, self.l0, self.l1 = {}, [], None
        self.user_writes = self.disk_writes = 0

    def put(self, key, value):
        self.user_writes += 1
        self.mem[key] = value         # newest value wins in memory
        if len(self.mem) >= MEM_LIMIT:
            self.flush()

    def delete(self, key):
        self.put(key, TOMB)           # a delete is just a write

    def flush(self):                  # one sequential write
        self.l0.insert(0, Run(self.mem.items()))   # newest first
        self.disk_writes += len(self.mem)
        self.mem = {}
        if len(self.l0) >= L0_LIMIT:
            self.compact()

    def compact(self):                # merge L0 + L1 into new L1
        merged = {}
        older = [self.l1] if self.l1 else []
        for run in older + self.l0[::-1]:   # oldest to newest
            merged.update(run.items)        # newer overwrites
        live = [(k, v) for k, v in merged.items() if v is not TOMB]
        self.l1, self.l0 = Run(live), []    # L1 is the last level,
        self.disk_writes += len(live)       # so tombstones drop

    def get(self, key):
        if key in self.mem:
            return self.mem[key], ["memtable"]
        path = ["memtable"]
        runs = [(f"L0[{i}]", r) for i, r in enumerate(self.l0)]
        if self.l1:
            runs.append(("L1", self.l1))
        for name, run in runs:            # newest to oldest
            if not run.bloom.might_have(key):
                path.append(f"{name} skip")
                continue
            path.append(f"{name} read")
            if key in run.index:
                return run.index[key], path
        return TOMB, path


def show(db):
    fmt = lambda items: " ".join(
        f"{k}:{'†' if v is TOMB else v}" for k, v in items)
    print("   mem:", fmt(sorted(db.mem.items())) or "-")
    for i, r in enumerate(db.l0):
        print(f"   L0[{i}]:", fmt(r.items))
    print("   L1:", fmt(db.l1.items) if db.l1 else "-")


def numbers():
    b = 10                                   # Bloom bits per key
    k = round(b * math.log(2))
    fpr = (1 - math.exp(-k / b)) ** k
    print(f"Bloom: k = {k}, FPR = (1-e^(-{k}/{b}))^{k}"
          f" = {fpr:.5f} = {fpr:.2%}")
    runs = 4 + 2                             # 4 L0 files + L1, L2
    print(f"missing key, {runs} runs: {runs} reads without filters,"
          f" {runs * fpr:.3f} expected with")
    print(f"filter memory, 10^7 keys: {b * 10**7 / 8 / 1e6:.1f} MB")
    T, mem = 10, 64                          # fan-out, memtable MB
    for data_mb in (1024, 1024**2):
        L = math.ceil(math.log(data_mb / mem, T))
        print(f"data {data_mb:>7} MB: levels = ceil(log10"
              f"({data_mb}/{mem})) = {L}, write amp <= {T}*{L}"
              f" = {T * L}")
    print(f"B-tree, 100 B row, 4 KB page: {4096 / 100:.0f}x worst")


if __name__ == "__main__":
    db = LSM()
    ops = [("put", 7, 30), ("put", 3, 12), ("put", 9, 55),
           ("put", 7, 35), ("del", 3), ("put", 5, 20)]
    for op in ops:
        if op[0] == "put":
            db.put(op[1], op[2])
        else:
            db.delete(op[1])
        print(op)
        show(db)
    for key in (7, 3):
        print(f"get({key}) ->", *db.get(key))
    for key, price in ((4, 8), (1, 15), (2, 40)):
        db.put(key, price)
        print(("put", key, price))
        show(db)
    for key in (7, 3, 6):
        print(f"get({key}) ->", *db.get(key))
    print(f"user writes {db.user_writes}, entries written to disk"
          f" {db.disk_writes}, write amp"
          f" {db.disk_writes / db.user_writes:.2f}")
    numbers()
