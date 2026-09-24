"""What triggers a context switch: count them with getrusage.

The kernel keeps two counters per process:
  ru_nvcsw  voluntary switches: the thread gave up the CPU itself
            (sleep, blocking read, waiting for a lock, yield)
  ru_nivcsw involuntary switches: the kernel took the CPU away
            (time slice used up, or a more urgent thread woke up)
Each demo below runs one kind of work and prints both deltas.
"""
import resource
import subprocess
import sys
import time


def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


def switches(work):
    """Run work(); return (voluntary, involuntary, seconds)."""
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    work()
    t1 = time.perf_counter()
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    return (r1.ru_nvcsw - r0.ru_nvcsw,
            r1.ru_nivcsw - r0.ru_nivcsw, t1 - t0)


def sleeps():                    # 200 naps of 1 ms: block on a timer
    for _ in range(200):
        time.sleep(0.001)


def pipe_reads():                # 50 reads that must wait for data
    child = ("import sys, time\n"
             "for _ in range(50):\n"
             "    time.sleep(0.002)\n"
             "    sys.stdout.buffer.write(b'x')\n"
             "    sys.stdout.flush()\n")
    p = subprocess.Popen([sys.executable, "-c", child],
                         stdout=subprocess.PIPE)
    for _ in range(50):
        p.stdout.read(1)         # blocks until the byte arrives
    p.wait()


def cpu_bound():                 # never blocks: only preemption
    fib(30)


if __name__ == "__main__":
    print(f"{'work':<22}{'voluntary':>10}{'involuntary':>13}"
          f"{'seconds':>9}")
    for name, work in [("200 x sleep(1 ms)", sleeps),
                       ("50 blocking reads", pipe_reads),
                       ("fib(30), CPU only", cpu_bound)]:
        v, iv, s = switches(work)
        print(f"{name:<22}{v:>10}{iv:>13}{s:>9.3f}")
