"""Semaphores: cap concurrency, then build a bounded buffer.

Part 1: 10 fake downloads of 0.1 s each behind Semaphore(3). At
most 3 run at once, so they finish in 4 waves (3 + 3 + 3 + 1),
about 0.4 s instead of 1.0 s serially or 0.1 s with no cap.

Part 2: a bounded buffer of capacity 2 built from two counting
semaphores (free slots, filled slots) plus a mutex.
"""
import threading
import time
from collections import deque

# ---- Part 1: limit how many downloads run at once ----------------
N, LIMIT, SECS = 10, 3, 0.1
gate = threading.Semaphore(LIMIT)     # counter starts at 3
count_lock = threading.Lock()
in_flight = max_seen = 0
starts = {}


def download(i, t0):
    global in_flight, max_seen
    with gate:                        # P: wait if counter is 0
        with count_lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            starts[i] = time.perf_counter() - t0
        time.sleep(SECS)              # the "download"
        with count_lock:
            in_flight -= 1
    # leaving the with-block is V: counter + 1, wakes one waiter


def run_downloads():
    t0 = time.perf_counter()
    ts = [threading.Thread(target=download, args=(i, t0))
          for i in range(N)]
    for t in ts: t.start()
    for t in ts: t.join()
    return time.perf_counter() - t0


# ---- Part 2: bounded buffer with two semaphores -------------------
CAP = 2
buf = deque()
free = threading.Semaphore(CAP)       # empty slots, starts at 2
full = threading.Semaphore(0)         # filled slots, starts at 0
mutex = threading.Lock()              # guards the deque itself
log = []
max_buf = 0


def producer(items):
    global max_buf
    for x in items:
        free.acquire()                # P(free): wait for a slot
        with mutex:
            buf.append(x)
            max_buf = max(max_buf, len(buf))
            log.append(f"put {x}  buffer={list(buf)}")
        full.release()                # V(full): one more item


def consumer(n):
    for _ in range(n):
        full.acquire()                # P(full): wait for an item
        with mutex:
            x = buf.popleft()
            log.append(f"get {x}  buffer={list(buf)}")
        free.release()                # V(free): one more slot
        time.sleep(0.01)              # slow consumer


if __name__ == "__main__":
    total = run_downloads()
    waves = sorted(round(s, 1) for s in starts.values())
    print(f"10 downloads, Semaphore({LIMIT}): {total:.3f} s")
    print(f"max in flight: {max_seen}")
    print(f"starts (s): {waves}")
    print()
    items = list("abcde")
    p = threading.Thread(target=producer, args=(items,))
    c = threading.Thread(target=consumer, args=(len(items),))
    p.start(); c.start(); p.join(); c.join()
    for line in log:
        print(line)
    print(f"max buffer size seen: {max_buf} (capacity {CAP})")
    b = threading.BoundedSemaphore(1)
    b.acquire(); b.release()
    try:
        b.release()                   # one V too many
    except ValueError as e:
        print(f"extra release -> ValueError: {e}")
