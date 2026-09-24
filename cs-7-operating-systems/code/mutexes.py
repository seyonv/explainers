"""Mutexes: ownership, re-entry, and what an uncontended lock costs.

1. A plain Lock taken twice by the same thread never succeeds: the
   thread would wait on itself forever. acquire(timeout=0.1) shows it.
2. An RLock remembers its owner and a count, so re-entry works.
3. Ownership: Python's Lock may be released by any thread (like a
   binary semaphore); RLock refuses unless the owner releases it.
4. Uncontended acquire+release cost, measured lightly with timeit.
"""
import threading
import timeit


def reenter(lock):
    with lock:                          # outer: e.g. transfer()
        got = lock.acquire(timeout=0.1) # inner: e.g. withdraw()
        if got:
            lock.release()
        return got


def release_from_other_thread(lock):
    lock.acquire()                      # main thread owns it
    result = []

    def other():
        try:
            lock.release()              # not the owner
            result.append("released")
        except RuntimeError as e:
            result.append(f"RuntimeError: {e}")

    t = threading.Thread(target=other)
    t.start()
    t.join()
    if result[0] != "released":
        lock.release()                  # owner cleans up
    return result[0]


def cost_ns(make, n=1_000_000, repeat=3):
    lk = make()
    a, r = lk.acquire, lk.release

    def loop():
        for _ in range(n):
            a(); r()
    best = min(timeit.repeat(loop, number=1, repeat=repeat))
    return best / n * 1e9


if __name__ == "__main__":
    print("re-enter Lock :", reenter(threading.Lock()))
    print("re-enter RLock:", reenter(threading.RLock()))
    print("other thread releases Lock :",
          release_from_other_thread(threading.Lock()))
    print("other thread releases RLock:",
          release_from_other_thread(threading.RLock()))
    for name, make in [("Lock", threading.Lock),
                       ("RLock", threading.RLock)]:
        print(f"{name:5} acquire+release: {cost_ns(make):.0f} ns")
