"""Spreading work: competing consumers vs log partitions.

24 resize jobs sit in a backlog. Job i goes to partition i % 6.
Every job takes 1 tick (0.1 s: one core does 10 conversions/s)
except job 4, a huge image that takes 10 ticks (illustrative).
Four consumers share the work in two ways:

  competing: one shared queue; any idle consumer takes the next job.
  partitions: Kafka-style range assignment of 6 partitions to 4
              consumers; each consumer works through its partitions
              one job at a time, in offset order.
"""
import heapq

P, C, N, SLOW = 6, 4, 24, 4
cost = [10 if i == SLOW else 1 for i in range(N)]


def range_assign(parts, consumers):   # Kafka's RangeAssignor
    q, r = divmod(parts, consumers)
    out, start = [], 0
    for c in range(consumers):
        n = q + (c < r)             # first r get one extra
        out.append(list(range(start, start + n)))
        start += n
    return out


def competing():                  # next job to first idle consumer
    done, free = [0] * N, [(0, c) for c in range(C)]
    for i in range(N):
        t, c = heapq.heappop(free)
        done[i] = t + cost[i]
        heapq.heappush(free, (done[i], c))
    return done


def partitioned(assign):          # one job at a time, in order
    done = [0] * N
    for parts in assign:
        t = 0
        for i in range(N):
            if i % P in parts:
                t += cost[i]
                done[i] = t
    return done


def report(name, done):
    fast = [d for i, d in enumerate(done) if i != SLOW]
    print(f"{name:10} finish {max(done):2}  "
          f"mean of 23 normal {sum(fast)} / 23 = {sum(fast) / 23:.2f}  "
          f"stuck {sum(d > 10 for d in fast)}")


if __name__ == "__main__":
    a = range_assign(P, C)
    print("assignment", a, [len(x) for x in a])
    comp, part = competing(), partitioned(a)
    report("competing", comp)
    report("partitions", part)
    print("per-job finish tick, competing :", comp)
    print("per-job finish tick, partitions:", part)
    print("partition 4 finishes:",
          [part[i] for i in range(N) if i % P == 4])
    print("8 consumers, 6 partitions:", range_assign(P, 8))
    t1, tinf = sum(cost), max(cost)
    print(f"greedy bound T1/P + Tinf = {t1}/{C} + {tinf}"
          f" = {t1 / C + tinf}")
    # with every job 1 tick, the 2,2,1,1 split alone costs time
    cost = [1] * N
    print("all 1 tick: competing", max(competing()),
          "partitions", max(partitioned(a)))
