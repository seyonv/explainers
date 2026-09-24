"""Image-resize service: the numbers behind its per-subsystem choices.

Source: ljeng/cheat-sheet, distributed-systems.md, The Reality
(Figure 1 and the five decisions). Everything marked ILLUSTRATIVE
is an assumption of this card, not a number from the source.
Runs under Python 3.10, standard library only.
"""
import random
from collections import defaultdict
from math import comb

MONTH_MIN = 30 * 24 * 60           # a 30-day month, in minutes
YEAR_H = 365 * 24                  # a 365-day year, in hours


def downtime(avail):
    """Allowed downtime: (minutes per 30-day month, hours per year)."""
    return (1 - avail) * MONTH_MIN, (1 - avail) * YEAR_H


def k_of_n(a, k, n):
    """P(at least k of n servers up), each up with prob a,
    ASSUMING independent failures (see correlated-failure)."""
    return sum(comb(n, i) * a**i * (1 - a)**(n - i)
               for i in range(k, n + 1))


def servers_needed(rps, per_server, spare=1):
    need = -(-rps // per_server)   # ceiling division
    return need, need + spare


# --- Decision 5: usage tracking with local aggregation --------------
def local_aggregate(requests):
    """requests: [(customer, cpu_ms, nbytes)] seen by ONE API server.
    Returns one row per customer: [count, cpu_ms, bytes]."""
    agg = defaultdict(lambda: [0, 0.0, 0])
    for cust, cpu_ms, nbytes in requests:
        row = agg[cust]
        row[0] += 1
        row[1] += cpu_ms
        row[2] += nbytes
    return agg                     # shipped as ONE batch write


def simulate_usage(rps=10_000, window_s=10, servers=3,
                   users=100_000, seed=0):
    """ILLUSTRATIVE traffic: uniform customers, image size uniform on
    0..512 KB (mean 256 KB), CPU time proportional to size so the
    mean is 100 ms (the source's 10 conversions/s per core)."""
    rng = random.Random(seed)
    per_server = [[] for _ in range(servers)]
    for i in range(rps * window_s):
        size = rng.randint(0, 512 * 1024)
        cpu_ms = 100 * size / (256 * 1024)
        cust = rng.randrange(users)
        per_server[i % servers].append((cust, cpu_ms, size))
    batches = [local_aggregate(reqs) for reqs in per_server]
    rows = sum(len(b) for b in batches)
    cpu_s = sum(r[1] for b in batches for r in b.values()) / 1000
    return rps * window_s, len(batches), rows, cpu_s / window_s


# --- Decision 4: an edge cache that refreshes every TTL seconds ------
def edge_copy_time(t, ttl, down):
    """When the edge last copied identity data from the data plane.
    It tries every `ttl` s; copies fail while down[0] <= s < down[1]."""
    return max(s for s in range(0, t + 1, ttl)
               if not down[0] <= s < down[1])


def edge_auth(created, t, ttl=5, down=(10, 30)):
    """HTTP status the edge gives at time t to a user created at
    `created`: it answers from its last copy, never blocks."""
    return 200 if edge_copy_time(t, ttl, down) >= created else 401


if __name__ == "__main__":
    for a in (0.999, 0.9999):
        m, y = downtime(a)
        print(f"{a:.2%} up: {m:5.2f} min/month, {y:5.2f} h/year")
    need, total = servers_needed(10_000, 5_000)
    print(f"API servers: need {need}, run {total} (N+1)")
    for a in (0.99, 0.995):
        p = k_of_n(a, need, total)
        print(f"  each {a:.1%} up -> 2 of 3 up {p:.5%} (indep.)")
    budget, _ = downtime(0.999)
    for name, mins in (("manual", 30), ("automated", 10 / 60)):
        print(f"{name} failover {mins:6.2f} min = "
              f"{mins / budget:6.1%} of the month's budget")
    n, writes, rows, cores = simulate_usage()
    print(f"usage, 10 s: {n:,} requests -> {writes} batch writes")
    print(f"  {rows:,} rows; CPU summed = {cores:,.0f} cores busy")
    print("edge, TTL 5 s, data plane down 10-30 s (ILLUSTRATIVE)")
    for t in (3, 6, 12, 20, 31):
        copy = edge_copy_time(t, 5, (10, 30))
        print(f"  t={t:2}  copy of t={copy:2}  u1 (made t=2): "
              f"{edge_auth(2, t)}  u2 (made t=12): {edge_auth(12, t)}")
