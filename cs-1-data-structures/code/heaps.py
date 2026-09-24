"""Heaps and priority queues.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md,
Heaps (The Skyline Problem, Rearrange String k Distance Apart,
Course Schedule). The three solutions are the source's algorithms,
lightly renamed; push/pop are written out by hand to show the
sift-up and sift-down that heapq does for you.
"""
import collections
import heapq
import itertools
import sys


def push(heap, x):
    heap.append(x)                   # new leaf at the end
    i = len(heap) - 1
    while i and heap[i] < heap[(i - 1) // 2]:
        p = (i - 1) // 2             # sift up: swap with parent
        heap[i], heap[p] = heap[p], heap[i]
        i = p


def pop(heap):
    top, last = heap[0], heap.pop()  # min is always heap[0]
    if heap:
        heap[0], i = last, 0         # last leaf goes to the root
        while 2 * i + 1 < len(heap):
            c = 2 * i + 1            # sift down: smaller child
            if c + 1 < len(heap) and heap[c + 1] < heap[c]:
                c += 1
            if heap[i] <= heap[c]:
                break
            heap[i], heap[c] = heap[c], heap[i]
            i = c
    return top


def schedule_course(courses):
    days, taken = 0, []              # taken: max-heap of -duration
    for duration, last_day in sorted(courses, key=lambda c: c[1]):
        days += duration
        heapq.heappush(taken, -duration)
        if days > last_day:          # over: drop the longest course
            days += heapq.heappop(taken)
    return len(taken)


def get_skyline(buildings):
    live = [(0, sys.maxsize)]        # (-height, right edge)
    skyline = [[0, 0]]
    events = sorted(itertools.chain(*[[(l, -h, r), (r, 0, 0)]
                                      for l, r, h in buildings]))
    for x, neg_h, right in events:
        while x >= live[0][1]:       # lazily drop ended buildings
            heapq.heappop(live)
        if neg_h:
            heapq.heappush(live, (neg_h, right))
        height = -live[0][0]
        if skyline[-1][1] != height:
            skyline.append([x, height])
    return skyline[1:]


def rearrange_string(s, k):
    heap = [(-n, ch) for ch, n in collections.Counter(s).items()]
    heapq.heapify(heap)
    cooling = collections.deque()    # last k - 1 letters used
    out = []
    while heap:
        n, ch = heapq.heappop(heap)  # most remaining letter
        out.append(ch)
        cooling.append((n + 1, ch))
        if len(cooling) >= k:
            n, ch = cooling.popleft()
            if n < 0:
                heapq.heappush(heap, (n, ch))
    return "".join(out) if len(out) == len(s) else ""


def trace_push(values):
    heap = []
    for x in values:
        heap.append(x)
        i, swaps = len(heap) - 1, []
        while i and heap[i] < heap[(i - 1) // 2]:
            p = (i - 1) // 2
            swaps.append(f"{heap[i]}<->{heap[p]}")
            heap[i], heap[p] = heap[p], heap[i]
            i = p
        print(f"push {x}: swaps {', '.join(swaps) or '-':<14} {heap}")
    return heap


def trace_schedule(courses):
    days, taken = 0, []
    for duration, last_day in sorted(courses, key=lambda c: c[1]):
        days += duration
        heapq.heappush(taken, -duration)
        line = f"take {duration:>4} (by {last_day:>4}): days {days:>4}"
        if days > last_day:
            dropped = -heapq.heappop(taken)
            days -= dropped
            line += f" > {last_day}, drop {dropped} -> {days}"
        print(line, sorted(-d for d in taken))
    return len(taken)


def is_heap(h):
    return all(h[(i - 1) // 2] <= h[i] for i in range(1, len(h)))


def ok_gap(r, k):
    return all(r[i] not in r[i + 1:i + k] for i in range(len(r)))


def brute_schedule(courses):
    best = 0
    for r in range(len(courses) + 1):
        for sub in itertools.combinations(courses, r):
            days = 0
            ok = True
            for d, last in sorted(sub, key=lambda c: c[1]):
                days += d
                ok = ok and days <= last
            if ok:
                best = r
    return best


if __name__ == "__main__":
    import random

    a = [5, 2, 9, 1, 5, 6]
    h = trace_push(a)
    print("pop ->", pop(h), h)
    q = []
    for x in a:
        heapq.heappush(q, x)
    print("heapq pushes:", q)
    print("heapq pop ->", heapq.heappop(q), q)
    b = a[:]
    heapq.heapify(b)
    print("heapify:", b)

    courses = [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
    print(trace_schedule(courses))
    print(trace_schedule([[5, 5], [4, 6], [2, 6]]))  # illustrative

    print(get_skyline([[2, 9, 10], [3, 7, 15], [5, 12, 12],
                       [15, 20, 10], [19, 24, 8]]))
    print(rearrange_string("aabbcc", 3), repr(rearrange_string(
        "aaabc", 3)), rearrange_string("aaadbbcc", 2))

    rng = random.Random(0)
    for _ in range(1000):
        xs = [rng.randint(0, 20) for _ in range(rng.randint(0, 30))]
        mine = []
        for x in xs:
            push(mine, x)
        assert is_heap(mine)
        assert [pop(mine) for _ in xs] == sorted(xs)
        cs = [[rng.randint(1, 9), rng.randint(1, 20)]
              for _ in range(rng.randint(0, 7))]
        assert schedule_course(cs) == brute_schedule(cs)
        s = "".join(rng.choice("abc") for _ in range(rng.randint(0, 7)))
        k = rng.randint(0, 3)
        r = rearrange_string(s, k)
        assert (r != "") == any(ok_gap(p, k) for p in
                                itertools.permutations(s)) or s == ""
        assert r == "" or sorted(r) == sorted(s) and ok_gap(r, k)
    print("1,000 random cases: push/pop sorted, schedule and"
          " rearrange match brute force")
