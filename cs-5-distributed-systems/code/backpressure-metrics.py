"""Backpressure, percentiles, yield and harvest.

The image-resize service from the cheat-sheet: one core converts 10
images/s, the SLO is p99 < 500 ms and 99.9% availability. Arrival
rates, the latency distribution and the traffic curve are
illustrative; every number printed is computed here.
"""
import math
import random


def queue(arrive, serve, seconds, cap=None):
    """Once-a-second tick model of one server's queue.

    Each second `arrive` requests show up and `serve` finish. With a
    cap, at most `cap` may be left waiting, so anything beyond
    serve + cap - backlog is rejected at once. Returns
    (second, backlog, wait_s, rejected_so_far) for every second.
    """
    backlog, rejected, rows = 0, 0, []
    for t in range(1, seconds + 1):
        room = arrive if cap is None else serve + cap - backlog
        taken = min(arrive, room)
        rejected += arrive - taken     # fast "503, back off"
        backlog = max(0, backlog + taken - serve)
        rows.append((t, backlog, backlog / serve, rejected))
    return rows


def pct(xs, p):
    """Nearest-rank percentile of an already sorted list."""
    return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]


def latency_sample(n=10_000, median_ms=50, sigma=1.0, seed=7):
    rng = random.Random(seed)
    mu = math.log(median_ms)
    return sorted(rng.lognormvariate(mu, sigma) for _ in range(n))


def p_slow(fanout, p=0.01):
    """P(at least one of `fanout` parallel calls is slow)."""
    return 1 - (1 - p) ** fanout


def fanout_sim(xs, fanout=100, trials=2_000, slo=500, seed=8):
    rng = random.Random(seed)
    slow = sum(max(rng.choices(xs, k=fanout)) > slo
               for _ in range(trials))
    return slow / trials


def yield_vs_uptime(outage_s, rps_during, rps_avg, day=86_400):
    uptime = 1 - outage_s / day
    yld = 1 - outage_s * rps_during / (rps_avg * day)
    return uptime, yld


def harvest(shards_up, shards_total):
    return shards_up / shards_total


if __name__ == "__main__":
    print("1. Backpressure: 12 req/s offered, 10 req/s served")
    free = queue(12, 10, 60)
    capped = queue(12, 10, 60, cap=4)
    for t in (1, 2, 3, 5, 10, 60):
        _, b, w, _ = free[t - 1]
        _, cb, cw, rej = capped[t - 1]
        print(f"  t={t:>2}s  unbounded backlog {b:>3} wait {w:>4.1f}s"
              f"  | cap 4: backlog {cb} wait {cw:.1f}s"
              f" rejected {rej}")
    print(f"  cap 4 steady state: serves 10/s, rejects 2/s ="
          f" {2 / 12:.1%}, yield {10 / 12:.1%}")

    print("2. Percentiles: lognormal, median 50 ms, sigma 1.0")
    xs = latency_sample()
    mean = sum(xs) / len(xs)
    print(f"  mean {mean:.0f} ms")
    for p in (50, 90, 99, 99.9):
        print(f"  p{p:<4} {pct(xs, p):6.0f} ms")
    print(f"  max    {xs[-1]:6.0f} ms")
    over = sum(x > 500 for x in xs)
    print(f"  over 500 ms: {over} of {len(xs)} = {over / len(xs):.2%}")
    print(f"  p99 / p50 = {pct(xs, 99) / pct(xs, 50):.1f}x,"
          f" p99 / mean = {pct(xs, 99) / mean:.1f}x")

    print("3. Fan-out: each server slow 1% of the time")
    for n in (1, 10, 100):
        print(f"  fan-out {n:>3}: P(request slow) = {p_slow(n):.1%}")
    frac = sum(x > 500 for x in xs) / len(xs)
    print(f"  with the sample above (P(one > 500 ms) = {frac:.2%}):"
          f" 1-(1-{frac:.4f})^100 = {p_slow(100, frac):.1%},"
          f" simulated {fanout_sim(xs):.1%}")
    q = 0.5 ** (1 / 100)
    print(f"  median of max of 100 = the p{q * 100:.2f} of one server"
          f" = {pct(xs, q * 100):.0f} ms")

    print("4. Uptime vs yield: 60 s outage, average 5,500 req/s")
    for label, rps in (("peak 10,000/s", 10_000),
                       ("trough 1,000/s", 1_000)):
        up, yl = yield_vs_uptime(60, rps, 5_500)
        print(f"  {label:<15} uptime {up:.3%}  yield {yl:.3%}"
              f"  ({'meets' if yl >= 0.999 else 'misses'} 99.9%)")

    print("5. Harvest: shards A, B, C; B is down")
    print(f"  answer from A and C: harvest = 2/3 = {harvest(2, 3):.1%}")
