"""How a context switch works: a toy CPU, then a real ping-pong.

Part 1 is a model, not the kernel: one CPU with a few registers,
and three tasks. A and A2 are threads of one process (same page
table); B is another process. switch() does what a kernel does:
save the running task's registers into its PCB, load the next
task's registers, and change the page-table base only if the
address space changes.

Part 2 is real: two processes bounce one byte over a pipe, and
getrusage counts how often the parent blocked (a voluntary switch).
"""
import time
from multiprocessing import Pipe, Process
from resource import RUSAGE_SELF, getrusage

CPU = {"pc": 0, "sp": 0, "r0": 0, "ptbr": ""}


class PCB:                          # what the kernel keeps per task
    def __init__(self, name, pc, sp, space):
        self.name = name
        self.regs = {"pc": pc, "sp": sp, "r0": 0, "ptbr": space}


def run(steps):                     # user code: count, advance pc
    for _ in range(steps):
        CPU["r0"] += 1
        CPU["pc"] += 4              # one 4-byte instruction


def switch(cur, nxt):
    cur.regs = dict(CPU)            # 1. save registers into PCB
    new_space = nxt.regs["ptbr"] != CPU["ptbr"]
    CPU.update(nxt.regs)            # 2. load next task's registers
    return new_space                # 3. new page table? (process)


def show(event, cur):
    r = CPU
    print(f"{event:<14} {cur:<3} pc={r['pc']:#x} sp={r['sp']:#x}"
          f" r0={r['r0']} ptbr={r['ptbr']}")


def toy():
    a = PCB("A", 0x400, 0x7f00, "P1")
    a2 = PCB("A2", 0x480, 0x6f00, "P1")     # thread: same space
    b = PCB("B", 0x800, 0x9f00, "P2")       # other process
    CPU.update(a.regs)
    plan = [(a, 3), (a2, 2), (b, 1), (a, 1)]
    for i, (task, steps) in enumerate(plan):
        if i:
            prev = plan[i - 1][0]
            moved = switch(prev, task)
            kind = "new page table" if moved else "same page table"
            print(f"  switch {prev.name}->{task.name}: {kind}")
        run(steps)
        show(f"ran {steps} step(s)", task.name)
    print(f"A's saved r0 survived: A resumed at 3, now {CPU['r0']}")


def echo(conn, n):
    for _ in range(n + 1):          # +1 for the warm-up message
        conn.send_bytes(conn.recv_bytes())


def ping_pong(n=5_000):              # real processes, real switches
    a, b = Pipe()
    child = Process(target=echo, args=(b, n)); child.start()
    a.send_bytes(b"x"); a.recv_bytes()  # warm-up: wait for child
    before = getrusage(RUSAGE_SELF).ru_nvcsw
    t = time.perf_counter()
    for _ in range(n):
        a.send_bytes(b"x")
        a.recv_bytes()              # blocks: parent gives up CPU
    dt = time.perf_counter() - t
    after = getrusage(RUSAGE_SELF).ru_nvcsw
    child.join()
    print(f"{n:,} round trips: {dt / n * 1e6:.1f} us each,"
          f" parent blocked {after - before:,} times")


if __name__ == "__main__":
    toy()
    print()
    ping_pong()
