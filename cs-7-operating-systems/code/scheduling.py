"""CPU scheduling: FCFS, SJF and round robin on three jobs.

Jobs A = 8, B = 4, C = 1 time units of CPU, all arriving at t = 0
(in the order A, B, C). Each policy returns a Gantt chart as a list
of (job, start, end) slices, and we score it by average turnaround
(finish - arrival) and average response (first run - arrival).
"""
from collections import deque

JOBS = {"A": 8, "B": 4, "C": 1}     # name -> CPU burst, arrive at 0


def fcfs(jobs):
    t, out = 0, []
    for name, burst in jobs.items():        # arrival order
        out.append((name, t, t + burst))
        t += burst
    return out


def sjf(jobs):
    order = sorted(jobs, key=jobs.get)      # shortest burst first
    return fcfs({n: jobs[n] for n in order})


def rr(jobs, q):
    left, ready = dict(jobs), deque(jobs)
    t, out = 0, []
    while ready:
        n = ready.popleft()
        run = min(q, left[n])               # one quantum or less
        out.append((n, t, t + run))
        t += run
        left[n] -= run
        if left[n]:
            ready.append(n)                 # back of the queue
    return out


def score(gantt):
    first, done = {}, {}
    for n, s, e in gantt:
        first.setdefault(n, s)
        done[n] = e
    k = len(done)
    return sum(done.values()) / k, sum(first.values()) / k, done, first


def show(label, gantt):
    turn, resp, done, first = score(gantt)
    bar = " ".join(f"{n}{s}-{e}" for n, s, e in gantt)
    print(f"{label:6} {bar}")
    print(f"       finish {done}  first run {first}")
    print(f"       avg turnaround {turn:.2f}  avg response {resp:.2f}")


if __name__ == "__main__":
    show("FCFS", fcfs(JOBS))
    show("SJF", sjf(JOBS))
    show("RR q=2", rr(JOBS, 2))
    show("RR q=1", rr(JOBS, 1))
