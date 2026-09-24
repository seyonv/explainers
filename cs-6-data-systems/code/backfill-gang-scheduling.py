"""Backfilling and gang scheduling on an 8-GPU cluster.

Part 1: FCFS vs EASY backfilling. One job (J0) already holds 4 GPUs
until t = 4 h; four jobs are queued at t = 0 in order J1..J4. EASY
reserves the earliest start ("shadow time") for the job at the head
of the queue, then lets a later job jump ahead only if it fits now
AND either ends by the shadow time or uses GPUs the head job won't
need then. Runtimes are the users' estimates and assumed exact.

Part 2: gang scheduling. The FfDL paper's example (Jayaram et al.,
Middleware 2019): 4 machines x 2 GPUs, 4 jobs of 2 learners, each
learner needs a whole machine. Placing pods one at a time can give
every job half of what it needs; placing a job all-or-nothing can't.
Python 3.10, standard library only.
"""
TOTAL = 8
RUNNING = [("J0", 4, 4)]                 # (name, gpus, ends at t)
QUEUE = [("J1", 8, 2), ("J2", 2, 3),     # (name, gpus, hours)
         ("J3", 2, 6), ("J4", 4, 2)]


def simulate(backfill, greedy=False):  # greedy: no reservation
    run = list(RUNNING)                  # (name, gpus, end)
    queue, start, t = list(QUEUE), {}, 0
    while queue:
        free = TOTAL - sum(g for _, g, _ in run)
        while queue and queue[0][1] <= free:      # FCFS part
            n, g, d = queue.pop(0)
            run.append((n, g, t + d)); start[n] = t; free -= g
        for job in list(queue) if greedy else []:
            if job[1] <= free:           # any job that fits starts
                queue.remove(job); free -= job[1]
                run.append((job[0], job[1], t + job[2]))
                start[job[0]] = t
        if backfill and queue:
            head_g = queue[0][1]         # reserve for the head job
            shadow, avail = t, free
            for _, g, end in sorted(run, key=lambda r: r[2]):
                if avail >= head_g:
                    break
                shadow, avail = end, avail + g
            extra = avail - head_g       # spare GPUs at shadow
            for job in queue[1:]:
                n, g, d = job
                ends_ok = t + d <= shadow
                if g <= free and (ends_ok or g <= extra):
                    queue.remove(job)
                    run.append((n, g, t + d)); start[n] = t
                    free -= g
                    if not ends_ok:
                        extra -= g
        t = min(end for _, _, end in run)         # next finish
        run = [r for r in run if r[2] > t]
    return start


def idle_gpu_hours(start, until):
    jobs = {n: (g, d) for n, g, d in QUEUE}
    busy = sum(g * min(e, until) for _, g, e in RUNNING)
    for n, t0 in start.items():
        g, d = jobs[n]
        busy += g * max(0, min(t0 + d, until) - t0)
    return TOTAL * until - busy


def gang_demo():
    machines = 4                         # 2 GPUs each
    jobs = {j: 2 for j in "ABCD"}        # learners needed per job
    # one pod at a time, interleaved across jobs (the bad order)
    placed = {j: 0 for j in jobs}
    pods = [j for _ in range(2) for j in jobs]   # A B C D A B C D
    for j in pods[:machines]:
        placed[j] += 1
    running = [j for j in jobs if placed[j] == jobs[j]]
    idle = sum(2 for j in jobs if 0 < placed[j] < jobs[j])
    # gang: a job is placed only if all its learners fit
    free, gang = machines, []
    for j, need in jobs.items():
        if need <= free:
            gang.append(j); free -= need
    return running, idle, gang


if __name__ == "__main__":
    for name, bf, gr in (("FCFS", False, False),
                         ("EASY backfill", True, False),
                         ("greedy", False, True)):
        s = simulate(bf, gr)
        waits = [s[n] for n, _, _ in QUEUE]
        print(f"{name:14} starts {s}")
        print(f"{'':14} waits {waits}  avg {sum(waits)/len(waits)}"
              f"  idle GPU-h before t=4: {idle_gpu_hours(s, 4)}")
    running, idle, gang = gang_demo()
    print("pod-by-pod: running jobs", running,
          f"| {idle} of 8 GPUs held idle")
    print("gang:       running jobs", gang, "| 2 jobs queued whole")
