"""Modern concurrency constructs: async tasks, futures, and CAS.

1. 10,000 waits of 0.1 s on one thread with asyncio.
2. The same idea with a thread pool and futures (16 jobs, 8 workers).
3. A lock-free style counter: read, compute, compare-and-swap, retry.
   Python has no user-level CAS, so cas() emulates the one atomic
   CPU instruction with a tiny lock. The retry loop is the real part.
4. How big a thread's stack is on this machine (macOS only).
"""
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor


# 1. asyncio: one thread, 10,000 coroutines, each parked on a timer
async def many_waits(n):            # one thread, n coroutines
    await asyncio.gather(*(asyncio.sleep(0.1) for _ in range(n)))


# 2. futures: submit work, get a handle, collect results later
def fetch(i):
    time.sleep(0.1)                 # stands in for a network call
    return i * i


def pool_demo():                    # a future = a handle to a result
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(fetch, i) for i in range(16)]
        return sum(f.result() for f in futs)   # blocks until done


# 3. compare-and-swap counter
_hw = threading.Lock()              # stands in for the CPU's atomicity


def cas(cell, expected, new):
    with _hw:                       # one indivisible step in hardware
        if cell[0] != expected:
            return False
        cell[0] = new
        return True


def add_one(cell):
    retries = 0
    while True:
        old = cell[0]               # read
        time.sleep(0)               # let another thread get in
        if cas(cell, old, old + 1):  # only if unchanged
            return retries
        retries += 1                # someone beat us: try again


def cas_demo(threads=4, per=10_000):
    cell, retries = [0], []

    def work():
        retries.append(sum(add_one(cell) for _ in range(per)))
    ts = [threading.Thread(target=work) for _ in range(threads)]
    for t in ts: t.start()
    for t in ts: t.join()
    return cell[0], sum(retries)


# 4. stack size of a new Python thread vs the C default (macOS)
def stack_sizes():
    import ctypes
    libc = ctypes.CDLL(None)
    libc.pthread_self.restype = ctypes.c_void_p
    libc.pthread_get_stacksize_np.restype = ctypes.c_size_t
    libc.pthread_get_stacksize_np.argtypes = [ctypes.c_void_p]
    attr, size = ctypes.create_string_buffer(128), ctypes.c_size_t()
    libc.pthread_attr_init(attr)
    libc.pthread_attr_getstacksize(attr, ctypes.byref(size))
    got = {}

    def probe():
        got["py"] = libc.pthread_get_stacksize_np(libc.pthread_self())
    t = threading.Thread(target=probe)
    t.start(); t.join()
    return size.value, got["py"]


if __name__ == "__main__":
    t0 = time.perf_counter()
    asyncio.run(many_waits(10_000))
    dt = time.perf_counter() - t0
    print(f"10,000 asyncio.sleep(0.1): {dt:.2f} s")

    t0 = time.perf_counter()
    total = pool_demo()
    print(f"16 x fetch on 8 threads: {time.perf_counter() - t0:.2f} s,"
          f" sum = {total}")

    count, retries = cas_demo()
    print(f"CAS counter: {count:,} of 40,000, {retries:,} retries")

    try:
        c_default, py_thread = stack_sizes()
        print(f"pthread default stack: {c_default:,} B;"
              f" Python thread stack: {py_thread:,} B")
    except (OSError, AttributeError):
        print("stack probe needs macOS")
