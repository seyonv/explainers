"""Caching patterns: cache-aside reads, LRU vs LFU hit ratios on a
Zipf(1.0) workload, and request coalescing for a thundering herd.

Run with python3 (3.10+). Standard library only; fixed seed.
"""
import random
import threading
import time
from collections import OrderedDict, defaultdict
from itertools import accumulate

N_KEYS, N_REQ, SEED = 10_000, 200_000, 42


def zipf_stream(n_keys=N_KEYS, n_req=N_REQ, s=1.0, seed=SEED):
    """Key k (1-based) is requested with probability ~ 1 / k**s."""
    cum = list(accumulate(1 / k**s for k in range(1, n_keys + 1)))
    rng = random.Random(seed)
    return rng.choices(range(1, n_keys + 1), cum_weights=cum, k=n_req)


class LRU:
    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def get(self, k):
        if k not in self.d:
            return None
        self.d.move_to_end(k)            # now the most recent
        return self.d[k]

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)   # drop least recent

    def delete(self, k):
        self.d.pop(k, None)


class LFU:
    """O(1) LFU: buckets of keys by hit count, oldest first."""

    def __init__(self, cap):
        self.cap, self.val, self.cnt = cap, {}, {}
        self.bucket = defaultdict(OrderedDict)
        self.low = 0

    def _touch(self, k):
        c = self.cnt[k]
        del self.bucket[c][k]
        if not self.bucket[c] and self.low == c:
            self.low = c + 1
        self.cnt[k] = c + 1
        self.bucket[c + 1][k] = None

    def get(self, k):
        if k not in self.val:
            return None
        self._touch(k)
        return self.val[k]

    def put(self, k, v):
        if k in self.val:
            self.val[k] = v
            self._touch(k)
            return
        if len(self.val) >= self.cap:
            old, _ = self.bucket[self.low].popitem(last=False)
            del self.val[old], self.cnt[old]
        self.val[k], self.cnt[k], self.low = v, 1, 1
        self.bucket[1][k] = None


def cache_aside_get(cache, db, k):
    v = cache.get(k)                     # 1. try the cache
    if v is None:                        # 2. miss: read the db
        v = db[k]
        cache.put(k, v)                  # 3. fill the cache
    return v


def cache_aside_write(cache, db, k, v):
    db[k] = v                            # 1. the db is the truth
    cache.delete(k)                      # 2. invalidate, don't set


def hit_ratio(cache, stream):
    db = defaultdict(lambda: "row")
    hits = 0
    for k in stream:
        hits += cache.get(k) is not None
        cache_aside_get(cache, db, k)
    return hits / len(stream)


def top_k_share(k, n_keys=N_KEYS, s=1.0):
    """Best possible steady-state ratio: keep the k hottest keys."""
    w = [1 / i**s for i in range(1, n_keys + 1)]
    return sum(w[:k]) / sum(w)


class SingleFlight:
    """Coalesce concurrent misses on one key into one db call."""

    def __init__(self):
        self.lock, self.calls = threading.Lock(), {}

    def do(self, key, fn):
        with self.lock:
            ev = self.calls.get(key)
            leader = ev is None
            if leader:
                ev = self.calls[key] = [threading.Event(), None]
        if leader:
            ev[1] = fn()
            ev[0].set()
            with self.lock:
                del self.calls[key]
        else:
            ev[0].wait()
        return ev[1]


def herd(n_clients=100, coalesce=False):
    """n clients miss the same hot key at once; count db queries."""
    db_calls = [0]
    count_lock = threading.Lock()

    def slow_db():
        with count_lock:
            db_calls[0] += 1
        time.sleep(0.05)                 # a 50 ms query
        return "row"

    sf = SingleFlight()
    go = threading.Barrier(n_clients)

    def client():
        go.wait()                        # everyone misses together
        sf.do("hot", slow_db) if coalesce else slow_db()

    ts = [threading.Thread(target=client) for _ in range(n_clients)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return db_calls[0]


if __name__ == "__main__":
    stream = zipf_stream()
    print(f"Zipf(1.0), {N_KEYS:,} keys, {N_REQ:,} requests,"
          f" seed {SEED}")
    for cap in (100, 1000):
        lru = hit_ratio(LRU(cap), stream)
        lfu = hit_ratio(LFU(cap), stream)
        best = top_k_share(cap)
        print(f"cap {cap:>5}: LRU {lru:.3f}  LFU {lfu:.3f}"
              f"  top-{cap} share {best:.3f}")
    cache, db = LRU(3), {"A": 1, "B": 2}
    print("read A:", cache_aside_get(cache, db, "A"), "(miss, filled)")
    cache_aside_write(cache, db, "A", 10)
    print("after write, cached:", cache.get("A"),
          "| next read:", cache_aside_get(cache, db, "A"))
    trace = []
    lru = LRU(3)
    for k in "ABCADBAE":
        hit = lru.get(k) is not None
        if not hit:
            lru.put(k, 1)
        trace.append(f"{k}:{'hit' if hit else 'miss'} {''.join(lru.d)}")
    print("LRU(3) trace:", " | ".join(trace))
    print("herd, 100 clients: db queries without coalescing",
          herd(coalesce=False), "with", herd(coalesce=True))
