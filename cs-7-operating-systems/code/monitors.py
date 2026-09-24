"""Monitors and condition variables: a bounded buffer of capacity 2.

A monitor is one lock plus condition variables that threads wait on
while holding it. The producer waits on not_full, the consumer waits
on not_empty, and both re-check their condition in a while loop.
"""
import collections
import queue
import threading
import time

CAP = 2
lock = threading.Lock()
not_full = threading.Condition(lock)    # two conditions,
not_empty = threading.Condition(lock)   # one shared lock
buf = collections.deque()
log = []


def put(x):
    with lock:
        while len(buf) == CAP:          # while, not if (Mesa)
            log.append(f"P  waits, full    {list(buf)}")
            not_full.wait()             # releases lock, sleeps
        buf.append(x)
        log.append(f"P  put {x}          {list(buf)}")
        not_empty.notify()              # a hint, not a handoff


def get():
    with lock:
        while not buf:
            log.append(f"C  waits, empty  {list(buf)}")
            not_empty.wait()
        x = buf.popleft()
        log.append(f"C  got {x}          {list(buf)}")
        not_full.notify()
        return x


def producer(n):
    for i in range(1, n + 1):
        put(i)


def consumer(n):
    for _ in range(n):
        time.sleep(0.05)                # a slow consumer
        get()


def stolen_wakeup(recheck):
    """C1 waits on an empty buffer. Main puts 1 and notifies, but
    before C1 gets the lock back, a second consumer takes the item.
    With `if`, C1 pops an empty deque; with `while`, it re-checks."""
    cv = threading.Condition()          # RLock inside
    b, out = collections.deque(), []

    def c1():
        with cv:
            if recheck:
                woke = False
                while not b:
                    if woke:            # woke up to an empty buffer
                        out.append("C1 woke, empty: waits again")
                    cv.wait()
                    woke = True
            elif not b:
                cv.wait()
            try:
                out.append(f"C1 got {b.popleft()}")
            except IndexError:
                out.append("C1 IndexError: buffer empty")

    t = threading.Thread(target=c1)
    t.start()
    while not getattr(cv, "_waiters", None):
        time.sleep(0.001)               # until C1 is waiting
    with cv:
        b.append(1)
        cv.notify()                     # C1 is now runnable...
        out.append(f"C2 got {b.popleft()}")  # ...but C2 got in first
    time.sleep(0.05)
    if recheck:
        with cv:
            b.append(2)
            cv.notify()
    t.join()
    return out


if __name__ == "__main__":
    n = 5
    p = threading.Thread(target=producer, args=(n,))
    c = threading.Thread(target=consumer, args=(n,))
    p.start(); c.start(); p.join(); c.join()
    print("\n".join(log))
    print()
    print("if:   ", stolen_wakeup(recheck=False))
    print("while:", stolen_wakeup(recheck=True))
    print()
    q = queue.Queue(maxsize=CAP)        # the same monitor, ready-made
    print(q.not_full._lock is q.mutex is q.not_empty._lock)
