"""Deadlock: make one, detect it with a timeout, then prevent it.

Two threads take locks A and B in opposite orders. A barrier makes
sure each holds its first lock before asking for the second, so the
deadlock happens every run instead of only sometimes. The second
acquire uses timeout=1, so the program reports the deadlock instead
of hanging forever. Taking locks in one global order fixes it.
"""
import threading
import time

A, B = threading.Lock(), threading.Lock()


def worker(name, first, second, gate, log):
    with first:
        if gate:
            gate.wait()                  # everyone holds one lock
        if second.acquire(timeout=1.0):  # False after 1 s
            second.release()
            log.append(f"{name}: got both")
        else:
            log.append(f"{name}: timed out")


def run(orders, force=True):
    gate = threading.Barrier(len(orders)) if force else None
    log = []
    ts = [threading.Thread(target=worker, args=(n, f, s, gate, log))
          for n, (f, s) in orders.items()]
    t0 = time.perf_counter()
    for t in ts: t.start()
    for t in ts: t.join()
    return sorted(log), time.perf_counter() - t0


def ordered(*locks):
    return sorted(locks, key=id)         # one global order for all


def find_cycle(waits_for):
    """waits_for: {thread: thread it waits on}. Return a cycle."""
    for start in waits_for:
        path, t = [], start
        while t in waits_for and t not in path:
            path.append(t)
            t = waits_for[t]
        if t in path:
            return path[path.index(t):] + [t]
    return None


def philosophers(n, fix):
    forks = [threading.Lock() for _ in range(n)]
    orders = {}
    for i in range(n):
        lo, hi = sorted((i, (i + 1) % n))
        first, second = (lo, hi) if fix else (i, (i + 1) % n)
        orders[f"P{i}"] = (forks[first], forks[second])
    # naive: force everyone to hold a fork first (the bad timing);
    # fixed: no barrier, since P0 and P4 now both start at fork 0
    log, dt = run(orders, force=not fix)
    return sum("got both" in s for s in log), dt


if __name__ == "__main__":
    log, dt = run({"T1": (A, B), "T2": (B, A)})
    print(f"opposite order: {log}  {dt:.2f} s")
    log, dt = run({"T1": ordered(A, B), "T2": ordered(B, A)},
                  force=False)
    print(f"global order:   {log}  {dt:.4f} s")
    print("wait-for cycle:", find_cycle({"T1": "T2", "T2": "T1"}))
    ring = {f"P{i}": f"P{(i + 1) % 5}" for i in range(5)}
    print("philosopher cycle:", " -> ".join(find_cycle(ring)))
    for fix in (False, True):
        ate, dt = philosophers(5, fix)
        label = "lowest fork first" if fix else "left fork first"
        print(f"{label}: {ate} of 5 ate  {dt:.4f} s")
