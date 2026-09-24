"""Processes: fork, copy-on-write, exec and wait, in Python.

POSIX only (macOS, Linux). The parent forks a child; the child
changes its own copy of x, then either exits with a code or
replaces itself with another program via exec. The parent waits.
"""
import multiprocessing as mp
import os
import sys


def fork_and_wait():
    x = 1
    sys.stdout.flush()      # else the child inherits unprinted text
    pid = os.fork()         # one call, two returns
    if pid == 0:            # child: a copy of the parent
        x += 100            # page copied on this write (COW)
        print(f"child  pid={os.getpid()} parent={os.getppid()}"
              f" x={x}", flush=True)
        os._exit(7)         # exits now: no cleanup, no flush
    _, status = os.waitpid(pid, 0)      # block until it exits
    code = os.waitstatus_to_exitcode(status)
    print(f"parent pid={os.getpid()} child={pid} x={x}"
          f" exit={code}")


def fork_exec_wait():
    sys.stdout.flush()
    pid = os.fork()
    if pid == 0:            # same pid, brand-new program image
        os.execvp("echo", ["echo", "child is now echo"])
    _, status = os.waitpid(pid, 0)
    print(f"parent reaped {pid},"
          f" exit={os.waitstatus_to_exitcode(status)}")


def square(n):
    return n * n


if __name__ == "__main__":
    fork_and_wait()
    fork_exec_wait()
    print("default start method:", mp.get_start_method())
    with mp.Pool(4) as pool:     # 4 worker processes, reused
        print("pool:", pool.map(square, range(8)))
