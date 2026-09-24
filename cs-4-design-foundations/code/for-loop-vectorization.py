"""Why Python loops are slow, and what vectorization buys.

Source: ljeng/cheat-sheet, large-scale-design/system-design/
for-loop-problems.md. Its example adds 17 to every item of a
list (a CPython loop) and of a NumPy array (one vectorized op).
This file shows the per-item bytecode, the memory per item, the
temporary array hidden in np.abs(arr).mean() and a chunked fix.

Timings vary with machine and load; the card shows the series'
best-of-3 measurements (10**7 items: 0.582 s vs 0.00286 s).
"""
import dis
import sys
import time
import tracemalloc

import numpy as np


def add_loop(l):
    for i in range(len(l)):
        l[i] += 17              # 11 bytecodes per item


def add_numpy(a):
    a += 17                     # one C loop over raw int64s


def loop_bytecodes():
    """Bytecodes executed per iteration of add_loop."""
    ops = [x for x in dis.get_instructions(add_loop)]
    start = next(k for k, x in enumerate(ops)
                 if x.opname == "FOR_ITER")
    body = []
    for x in ops[start:]:
        body.append(x.opname)
        if x.opname == "JUMP_ABSOLUTE":
            return body


def mean_abs(a):
    return np.abs(a).mean()     # np.abs makes a full temp array


def mean_abs_chunked(a, chunk=100_000):
    total = 0.0
    for s in range(0, len(a), chunk):
        total += np.abs(a[s:s + chunk]).sum()
    return total / len(a)


def peak_bytes(fn, a):
    tracemalloc.start()
    fn(a)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def best_of(fn, make, runs=3):
    best = float("inf")
    for _ in range(runs):
        x = make()
        t = time.perf_counter()
        fn(x)
        best = min(best, time.perf_counter() - t)
    return best


if __name__ == "__main__":
    body = loop_bytecodes()
    print(len(body), "bytecodes per item:", " ".join(body))

    n = 10**7
    per_item = 8 + sys.getsizeof(10**6)   # pointer + int object
    print(f"list: {per_item} B/item -> {n * per_item / 1e6:.0f} MB;"
          f" int64 array: 8 B/item -> {n * 8 / 1e6:.0f} MB")

    a = np.random.default_rng(0).standard_normal(n)
    print(f"mean_abs         peak {peak_bytes(mean_abs, a):>11,} B")
    print(f"mean_abs_chunked peak "
          f"{peak_bytes(mean_abs_chunked, a):>11,} B")
    print("same answer:", np.isclose(mean_abs(a), mean_abs_chunked(a)))

    m = 10**6                              # small: under a second
    t_loop = best_of(add_loop, lambda: list(range(m)))
    t_np = best_of(add_numpy, lambda: np.arange(m))
    print(f"10**6 items: loop {t_loop:.4f} s, numpy {t_np:.6f} s,"
          f" {t_loop / t_np:.0f}x (varies with load)")
