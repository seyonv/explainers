"""Fair-share GPU scheduling: a hierarchical split and DRF.

1. The source guide's split: users share the cluster equally, and each
   user's share is divided among its jobs in proportion to priority.
2. Dominant resource fairness (DRF) with two resources, GPUs and CPU
   cores: always give the next task to the user whose dominant share
   is smallest (progressive filling, Ghodsi et al., NSDI 2011).
Cluster sizes and task shapes are illustrative. Python 3.10.
"""
from fractions import Fraction as F


def split(users, total=F(1)):
    """users: {user: {job: priority}} -> {job: fraction of cluster}."""
    out = {}
    for jobs in users.values():
        share = total / len(users)          # equal fair share per user
        weight = sum(jobs.values())
        for job, prio in jobs.items():
            out[job] = share * prio / weight
    return out


def dominant(tasks, need, cap):
    """Largest share of any one resource held by this user."""
    return max(F(tasks * n, c) for n, c in zip(need, cap))


def drf(cap, need, key=dominant, log=None):
    """Progressive filling. need: {user: per-task demand tuple}."""
    tasks = {u: 0 for u in need}
    used = [0] * len(cap)
    active = set(need)
    while active:
        u = min(sorted(active),
                key=lambda u: key(tasks[u], need[u], cap))
        after = [x + n for x, n in zip(used, need[u])]
        if any(a > c for a, c in zip(after, cap)):
            active.discard(u)               # next task no longer fits
            continue
        tasks[u] += 1
        used = after
        if log is not None:
            log.append((u, dict(tasks), tuple(used)))
    return tasks, used


def gpus_only(tasks, need, cap):
    """The same loop, but fairness counts GPUs and ignores CPUs."""
    return F(tasks * need[0], cap[0])


def pct(x):
    return f"{float(x) * 100:.1f}%"


if __name__ == "__main__":
    users = {"A": {"A1": 3, "A2": 1},
             "B": {"B1": 2, "B2": 1, "B3": 1}}
    shares = split(users)
    print(*(f"{j} {float(s):.1%}" for j, s in shares.items()))
    print(drf((16, 128), {"A": (2, 8), "B": (1, 16)}))
    print("\nHierarchical split (16 GPUs):")
    for job, s in split(users).items():
        print(f"  {job}  {pct(s):>6}  {s * 16} GPUs")

    cap = (16, 128)                         # GPUs, CPU cores
    need = {"A": (2, 8), "B": (1, 16)}      # per task
    log = []
    tasks, used = drf(cap, need, log=log)
    print("\nDRF trace (user, tasks, GPUs+CPUs used):")
    for u, t, us in log:
        ds = {v: str(dominant(t[v], need[v], cap)) for v in t}
        print(f"  +{u}  A={t['A']} B={t['B']}  used={us}  dom={ds}")
    print("DRF:", tasks, "used", used)

    t2, u2 = drf(cap, need, key=gpus_only)
    doms = {v: str(dominant(t2[v], need[v], cap)) for v in t2}
    print("GPUs only:", t2, "used", u2, "dominant", doms)

    half = {v: min(c // 2 // n for c, n in zip(cap, need[v]))
            for v in need}
    print("Static half split:", half)
