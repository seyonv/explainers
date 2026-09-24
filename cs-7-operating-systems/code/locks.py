"""Locks and race conditions: lose updates, then stop losing them.

Four threads each add 1 to a shared counter 100,000 times. The
unsafe version reads, yields, then writes back, so another thread
can slip in between and its update gets overwritten.
"""
import dis
import threading
import time

N, T = 100_000, 4
lock = threading.Lock()


def unsafe(c):
    for _ in range(N):
        v = c["n"]          # read
        time.sleep(0)       # yield: another thread may run here
        c["n"] = v + 1      # write back, maybe over a newer value


def safe(c):
    for _ in range(N):
        with lock:          # one thread at a time in this block
            v = c["n"]
            time.sleep(0)
            c["n"] = v + 1


def run(body):
    c = {"n": 0}
    ts = [threading.Thread(target=body, args=(c,)) for _ in range(T)]
    for t in ts: t.start()
    for t in ts: t.join()
    return c["n"]


def bump(c):
    c["n"] += 1             # looks atomic; it is several bytecodes


if __name__ == "__main__":
    want = N * T
    for body in (unsafe, safe):
        got = run(body)
        lost = want - got
        print(f"{body.__name__:6} {got:>7,} of {want:,}"
              f"  lost {lost:,} ({lost / want:.0%})")
    print()
    for ins in dis.get_instructions(bump):
        print(f"{ins.opname:14} {ins.argrepr}".rstrip())
