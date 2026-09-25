"""Two pointers and intervals.

Three patterns the source guide uses without naming them:
  1. opposite ends  - two-sum on a sorted array
  2. same direction - count pairs within distance d (the counting
                      loop inside the source's kth-pair-distance)
  3. intervals      - merge overlapping intervals; minimum meeting
                      rooms by sweeping start/end events
"""
import itertools
import random


def two_sum_sorted(a, target, trace=None):
    lo, hi = 0, len(a) - 1
    while lo < hi:
        s = a[lo] + a[hi]
        if trace is not None:
            trace.append((lo, hi, s))
        if s == target:
            return lo, hi
        if s < target:
            lo += 1           # a[lo] + anything left of hi is too small
        else:
            hi -= 1           # a[hi] + anything right of lo is too big
    return None


def pairs_within(a, d, trace=None):
    """Pairs i < j of sorted a with a[j] - a[i] <= d."""
    count, j = 0, 0
    for i in range(len(a)):
        while j < len(a) and a[j] - a[i] <= d:
            j += 1            # j only moves forward: O(n) in total
        count += j - i - 1    # partners of i are a[i+1 .. j-1]
        if trace is not None:
            trace.append((i, j, j - i - 1, count))
    return count


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:      # overlaps the last one
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def min_rooms(meetings, trace=None):
    """Half-open [start, end): a room freed at t is reusable at t."""
    events = sorted([(s, +1) for s, _ in meetings] +
                    [(e, -1) for _, e in meetings])  # -1 sorts first
    rooms = best = 0
    for t, delta in events:
        rooms += delta
        best = max(best, rooms)
        if trace is not None:
            trace.append((t, delta, rooms, best))
    return best


# --- brute-force checks -------------------------------------------

def _pairs_brute(a, d):
    return sum(1 for x, y in itertools.combinations(a, 2)
               if abs(x - y) <= d)


def _rooms_brute(meetings):
    ts = {t for m in meetings for t in m}
    return max((sum(s <= t < e for s, e in meetings) for t in ts),
               default=0)


def _merge_brute(intervals):
    covered = set()
    for s, e in intervals:
        covered |= {x / 2 for x in range(2 * s, 2 * e + 1)}
    return covered


def _covered(merged):
    return {x / 2 for s, e in merged for x in range(2 * s, 2 * e + 1)}


def self_check(trials=1000, seed=0):
    rng = random.Random(seed)
    for _ in range(trials):
        n = rng.randint(0, 12)
        a = sorted(rng.randint(0, 40) for _ in range(n))
        t = rng.randint(0, 80)
        got = two_sum_sorted(a, t)
        want = any(x + y == t for x, y in itertools.combinations(a, 2))
        assert (got is not None) == want
        if got:
            assert a[got[0]] + a[got[1]] == t
        d = rng.randint(0, 20)
        assert pairs_within(a, d) == _pairs_brute(a, d)
        iv = []
        for _ in range(n):
            s = rng.randint(0, 30)
            iv.append([s, s + rng.randint(1, 8)])
        m = merge(iv)
        assert _covered(m) == _merge_brute(iv)
        assert all(m[k][1] < m[k + 1][0] for k in range(len(m) - 1))
        assert min_rooms(iv) == _rooms_brute(iv)
    return trials


if __name__ == "__main__":
    a = [3, 9, 10, 27, 38, 43, 82]
    tr = []
    print("two_sum_sorted(a, 48) ->", two_sum_sorted(a, 48, tr))
    for lo, hi, s in tr:
        move = "found" if s == 48 else ("lo += 1" if s < 48 else
                                        "hi -= 1")
        print(f"  lo={lo} hi={hi}  {a[lo]} + {a[hi]} = {s}  {move}")
    print(f"  {len(tr)} sums checked vs {len(a) * (len(a) - 1) // 2}"
          " pairs for brute force")

    tr = []
    print("pairs_within(a, 10) ->", pairs_within(a, 10, tr))
    for i, j, k, c in tr:
        print(f"  i={i} (a={a[i]})  j stops at {j}  +{k}  total {c}")

    iv = [[1, 3], [2, 6], [8, 10], [15, 18]]
    print("iv =", iv)
    print("merge(iv) ->", merge(iv))

    rooms = [[0, 30], [5, 10], [15, 20], [10, 15]]
    print("rooms =", rooms)
    tr = []
    print("min_rooms(rooms) ->", min_rooms(rooms, tr))
    for t, dlt, r, b in tr:
        print(f"  t={t:>2} {'start' if dlt > 0 else 'end  '}"
              f"  in use {r}  max {b}")
    print("self_check:", self_check(), "random inputs agree with"
          " brute force")
