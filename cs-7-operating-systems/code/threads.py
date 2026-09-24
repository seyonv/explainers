"""Threads and the GIL: when do threads make Python faster?

Four CPU-bound fib(27) calls, run serially, on 4 threads and on 4
processes; then 8 x sleep(0.1), serially and on 8 threads. Threads
share one heap (the `seen` list) but each has its own OS thread id.
"""
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor


def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


def clock(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def on_threads(fn, args):
    ts = [threading.Thread(target=fn, args=(a,)) for a in args]
    for t in ts: t.start()
    for t in ts: t.join()


def on_processes(fn, args):
    with ProcessPoolExecutor(len(args)) as ex:
        list(ex.map(fn, args))


seen = []                   # one list on the shared heap


def who(_):
    seen.append(threading.get_native_id())   # own kernel thread


if __name__ == "__main__":
    print("python", sys.version.split()[0],
          "switch interval", sys.getswitchinterval(), "s")
    cpu = [27] * 4
    serial = clock(lambda: [fib(n) for n in cpu])
    print(f"4 x fib(27) serial    {serial:.3f} s")
    print(f"4 x fib(27) threads   "
          f"{clock(lambda: on_threads(fib, cpu)):.3f} s")
    print(f"4 x fib(27) processes "
          f"{clock(lambda: on_processes(fib, cpu)):.3f} s"
          "  (includes process start)")
    io = [0.1] * 8
    ser = clock(lambda: [time.sleep(s) for s in io])
    print(f"8 x sleep(0.1) serial  {ser:.3f} s")
    print(f"8 x sleep(0.1) threads "
          f"{clock(lambda: on_threads(time.sleep, io)):.3f} s")
    on_threads(who, range(4))
    print("4 threads appended to one list:", len(seen), "items,",
          len(set(seen)), "distinct native ids")
