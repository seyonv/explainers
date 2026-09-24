"""Sharding and hot keys: hash sharding, Zipf skew, key salting.

100,000 users (the image-resize service's user count) send requests
with Zipf(s = 1.1) popularity (illustrative). Keys are hashed to 16
shards. How much load lands on the hottest shard, and what does
salting the hottest key into 8 sub-keys buy and cost?

Run: python3 sharding-hot-keys.py   (Python 3.10, stdlib only)
"""
import hashlib
import itertools
import random
from collections import Counter

N_KEYS, S, SHARDS, SALTS = 100_000, 1.1, 16, 8
RPS = 10_000  # image-resize service, per region


def shard_of(key, n=16):
    # md5, not hash(): str hashes change every run
    d = hashlib.md5(key.encode()).digest()
    return int.from_bytes(d[:8], "big") % n


def zipf(n=100_000, s=1.1):
    w = [k ** -s for k in range(1, n + 1)]
    total = sum(w)
    return [x / total for x in w]    # p[k-1] = share of user k


def shard_loads(p, salted=(), n=16, salts=8):
    load = Counter()
    for k, pk in enumerate(p, start=1):
        if k in salted:              # writes pick 1 of 8 sub-keys
            for i in range(salts):
                load[shard_of(f"user{k}#{i}", n)] += pk / salts
        else:
            load[shard_of(f"user{k}", n)] += pk
    return load


def salted(k, hot, i):
    return f"user{k}#{i}" if k in hot else f"user{k}"


def sample_hottest(p, n_req=1_000_000, seed=0):
    rng = random.Random(seed)
    cum = list(itertools.accumulate(p))
    keys = rng.choices(range(1, len(p) + 1), cum_weights=cum, k=n_req)
    load = Counter(shard_of(f"user{k}") for k in keys)
    shard, hits = load.most_common(1)[0]
    return shard, hits, keys.count(1)


class SaltedCounter:
    """A per-user usage counter; hot users are split into 8 sub-keys."""

    def __init__(self, hot, seed=0):
        self.hot, self.store = set(hot), Counter()
        self.rng = random.Random(seed)
        self.reads = 0  # sub-key reads issued

    def add(self, k, amount=1):  # write: one sub-key, one shard
        i = self.rng.randrange(SALTS)
        self.store[salted(k, self.hot, i)] += amount

    def get(self, k):  # read: fan out to every sub-key, then sum
        subs = ([salted(k, self.hot, i) for i in range(SALTS)]
                if k in self.hot else [f"user{k}"])
        self.reads += len(subs)
        return sum(self.store[s] for s in subs)


if __name__ == "__main__":
    p = zipf()
    fair = 1 / SHARDS
    print(f"user 1 share {p[0]:.2%}, user 2 {p[1]:.2%}, "
          f"top 10 {sum(p[:10]):.1%}, fair shard share {fair:.2%}")

    load = shard_loads(p)
    hot = max(load, key=load.get)
    print(f"\nhash, 16 shards, exact: hottest = shard {hot}"
          f" at {load[hot]:.2%} ({load[hot] / fair:.2f}x fair,"
          f" {load[hot] * RPS:,.0f} of {RPS:,} req/s)")
    print(f"  = user 1 {p[0]:.2%} + rest of shard "
          f"{load[hot] - p[0]:.2%}; coldest {min(load.values()):.2%}")
    row = " ".join(f"{load[s]:.1%}" for s in range(SHARDS))
    print("  all 16:", row)

    shard, hits, u1 = sample_hottest(p)
    print(f"sampled 1,000,000 requests (seed 0): shard {shard} got "
          f"{hits:,} = {hits / 1e6:.2%}; user 1 sent {u1:,}")

    print("\nsalting the top keys into 8 sub-keys each:")
    for top in (0, 1, 2, 10):
        ld = shard_loads(p, set(range(1, top + 1)))
        s = max(ld, key=ld.get)
        print(f"  salt top {top:>2}: hottest shard {s:>2} at "
              f"{ld[s]:.2%} = {ld[s] * RPS:,.0f} req/s")
    salt_shards = [shard_of(f"user1#{i}") for i in range(SALTS)]
    exp = SHARDS * (1 - (1 - 1 / SHARDS) ** SALTS)
    print(f"user 1's 8 salts land on shards {salt_shards}: "
          f"{len(set(salt_shards))} distinct (expected {exp:.2f})")

    print("\nmore shards, no salting (one key can't be split):")
    for n in (16, 32, 64, 256):
        ld = shard_loads(p, n=n)
        m = max(ld.values())
        print(f"  {n:>3} shards: hottest {m:.2%} = {m * n:.1f}x fair")

    c = SaltedCounter(hot=[1])
    for _ in range(1_000):
        c.add(1)
        c.add(2)
    print(f"\nSaltedCounter: user1 = {c.get(1)}, user2 = {c.get(2)}, "
          f"sub-key reads = {c.reads} (8 for user 1 + 1 for user 2)")

    # range vs hash: 1,000 consecutive ids, 6,250 ids per range shard
    ids = range(1_000, 2_000)
    by_range = {(k - 1) // (N_KEYS // SHARDS) for k in ids}
    by_hash = {shard_of(f"user{k}") for k in ids}
    print(f"scan users 1000-1999: range sharding touches "
          f"{len(by_range)} shard, hash sharding {len(by_hash)}")
