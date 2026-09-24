"""The hardware underneath a context switch, seen from Python.

Sums a fresh 64 MB numpy array twice. The first pass takes a page
fault (a hardware exception into the kernel) for every 16 KB page it
touches; the second finds the pages mapped and the TLB warm.
Also times a trivial system call (a trap into the kernel) against
a plain Python call. Timings are one noisy run: keep them light.
"""
import os
import resource
import time

import numpy as np

MB = 1 << 20
PAGE = resource.getpagesize()           # 16,384 on Apple silicon


def faults():
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt


def timed_sum(a):
    f0, t0 = faults(), time.perf_counter()
    s = a.sum()
    return time.perf_counter() - t0, faults() - f0, s


def cold_vs_warm(size=64 * MB):
    a = np.zeros(size // 8)             # mapped lazily, not touched
    cold = timed_sum(a)                 # every page faults in
    warm = timed_sum(a)                 # pages mapped, TLB warm
    return cold, warm


def trap_vs_call(n=100_000):
    def f():
        return 1
    t0 = time.perf_counter()
    for _ in range(n):
        f()                             # stays in user mode (EL0)
    t1 = time.perf_counter()
    for _ in range(n):
        os.getppid()                    # svc: trap to kernel (EL1)
    t2 = time.perf_counter()
    return (t1 - t0) / n, (t2 - t1) / n


if __name__ == "__main__":
    size = 64 * MB
    print(f"page size {PAGE:,} B -> {size // PAGE:,} pages in 64 MB")
    (tc, fc, _), (tw, fw, _) = cold_vs_warm(size)
    print(f"cold sum: {tc * 1e3:6.1f} ms  page faults {fc:,}")
    print(f"warm sum: {tw * 1e3:6.1f} ms  page faults {fw:,}")
    print(f"cold / warm = {tc / tw:.1f}x")
    call, trap = trap_vs_call()
    print(f"python call  {call * 1e9:5.0f} ns")
    print(f"os.getppid() {trap * 1e9:5.0f} ns  (syscall round trip)")
