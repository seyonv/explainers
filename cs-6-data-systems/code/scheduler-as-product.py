"""The scheduler as a product: explain the wait, measure the waste.

Numbers come from the source guide's big-product-design-picture page:
the 64-pods-with-46-free case and the GPU utilization PDF.
"""
from collections import deque

# Source's PDF: share of allocated GPU time (%) per utilization bin
BINS = [(0, 0), (0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
        (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
SHARE = [28, 10, 9, 8, 7, 7, 6, 6, 6, 6, 7]


def why_pending(need, free):
    """Gang job: all GPUs at once or none. Say why, with numbers."""
    if free >= need:
        return "RUNNING"
    return (f"PENDING (gang): needs {need} GPUs, {free} free, "
            f"waiting for {need - free} more")


def util_stats(bins, share):
    """Mean utilization, taking each bin at its midpoint."""
    mean = sum((lo + hi) / 2 * s for (lo, hi), s in zip(bins, share))
    return mean / sum(share), share[0]


def start_of(jobs, order):
    """1 GPU, unit-length jobs: when does each job start?"""
    return {job: t for t, job in enumerate(order(jobs))}


def fcfs(jobs):
    return jobs


def fair(jobs):
    """Round-robin across users (fair queueing, one job each)."""
    per = {}
    for j in jobs:
        per.setdefault(j[0], deque()).append(j)
    out, qs = [], deque(per.values())
    while qs:
        q = qs.popleft()
        out.append(q.popleft())
        if q:
            qs.append(q)
    return out


if __name__ == "__main__":
    print(why_pending(64, 46))
    print(f"available share of request: {46 / 64:.1%}")
    mean, idle = util_stats(BINS, SHARE)
    print(f"bins sum to {sum(SHARE)}%")
    print(f"time at exactly 0%: {idle}%  under 10%: "
          f"{SHARE[0] + SHARE[1]}%  90-100%: {SHARE[-1]}%")
    print(f"mean utilization (bin midpoints): {mean:.1f}%")
    print(f"95% allocated x (1 - 0.28) busy = {0.95 * 0.72:.1%}")
    jobs = [f"A{i}" for i in range(1, 11)] + ["B1"]
    for name, order in (("FCFS", fcfs), ("fair", fair)):
        t = start_of(jobs, order)
        print(f"{name:4}: B1 waits {t['B1']}, A10 waits {t['A10']}")
