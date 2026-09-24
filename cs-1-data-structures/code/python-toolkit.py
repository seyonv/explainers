"""The Python interview toolkit: one tool per hot operation.

Source: ljeng/cheat-sheet, coding-algorithms/coding.md, heading "C++".
That heading is empty (a "Count the Repetitions" stub), so this file
is supplemented from the Python docs and the Python wiki's
TimeComplexity page. Running example from the course:
a = [5, 2, 9, 1, 5, 6], s = "ADOBECODEBANC".
"""
import bisect
import heapq
from collections import Counter, OrderedDict, defaultdict, deque

a = [5, 2, 9, 1, 5, 6]
s = "ADOBECODEBANC"


def toolkit_demo():
    q = deque(a)
    q.append(7)                  # O(1) at the right
    first = q.popleft()          # O(1); list.pop(0) is O(n)
    print("deque popleft:", first, list(q))

    print("3 smallest:", heapq.nsmallest(3, a))
    h = [-x for x in a]          # heapq is a min-heap only,
    heapq.heapify(h)             # so store -x for a max-heap
    print("max:", -heapq.heappop(h))

    b = sorted(a)                # [1, 2, 5, 5, 6, 9]
    print("bisect_left(b, 5):", bisect.bisect_left(b, 5))
    bisect.insort(b, 4)          # O(log n) search + O(n) shift
    print("after insort 4:", b)

    print("Counter:", Counter(s).most_common(3))
    groups = defaultdict(list)   # missing key -> new empty list
    for x in a:
        groups[x % 2].append(x)
    print("by parity:", dict(groups))

    seen = set(a)                # O(1) average `in`
    print("9 in seen:", 9 in seen, "| 3 in seen:", 3 in seen)
    print("joined:", "-".join(map(str, a)))  # not s += ...


def lru_demo():
    """OrderedDict keeps order and can move a key to the end in O(1)."""
    cache = OrderedDict()
    for k in "abca":
        cache[k] = cache.get(k, 0) + 1
        cache.move_to_end(k)     # most recently used goes last
    cache.popitem(last=False)    # evict the least recently used
    print("LRU after a,b,c,a then evict:", list(cache))


def pop0_shifts(n):
    """Element moves done by n calls of list.pop(0) on a list of n."""
    return sum(k - 1 for k in range(n, 0, -1))


def measured_ratios():
    """Ratios of the series' measured pairs (Apple M3, Python 3.10)."""
    list_s, set_s = 0.232, 24.6e-6      # 1,000 lookups, n = 10**5
    concat_s, join_s = 0.468, 0.0073    # 10**5 pieces
    print(f"list vs set membership: {list_s / set_s:,.0f}x")
    print(f"  per lookup: {list_s / 1000 * 1e6:.0f} us"
          f" vs {set_s / 1000 * 1e9:.1f} ns")
    print(f"  expected scan length: {10**5 // 2:,} items; "
          f"{list_s / 1000 / (10**5 / 2) * 1e9:.1f} ns per item")
    print(f"s = s + ... vs join: {concat_s / join_s:.0f}x")


if __name__ == "__main__":
    toolkit_demo()
    lru_demo()
    n = 10**5
    print(f"pop(0) x {n:,}: {pop0_shifts(n):,} shifts"
          f" = n(n-1)/2 = {n * (n - 1) // 2:,}")
    print(f"popleft x {n:,}: 0 shifts")
    measured_ratios()
