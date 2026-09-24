"""Back-of-the-envelope estimation, with Python's own overhead.

Source: ljeng/cheat-sheet, large-scale-design/system-design/
resource-estimation-real-systems.md. Recomputes the source's
estimates (Mississippi, 2M nodes in 128 MB, n^3 square roots,
large numbers, Little's law), then asks the 2M-node question of
Python itself using sys.getsizeof (CPython 3.10, 64-bit), and
sizes TinyURL. TinyURL inputs are illustrative; the source gives
none. The 50 ms latency and the 6x factor on TinyURL are
illustrative too.
"""
import sys
import tracemalloc

MB = 10**6
DAY, YEAR = 86_400, 365.25 * 86_400


def littles_law(rate=None, time=None, count=None):
    """L = lambda * W. Pass any two; returns the third."""
    if count is None:
        return rate * time
    if rate is None:
        return count / time
    return count / rate


class Node:                      # the source's node: int + pointer
    __slots__ = ("val", "next")

    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt


def python_node_bytes():
    """Bytes per linked-list node in CPython, from getsizeof."""
    obj = sys.getsizeof(Node(1000))  # header + 2 slots
    num = sys.getsizeof(1000)        # the int it points to
    ref = 8                          # the list slot pointing at it
    return obj, num, ref, obj + num + ref


def measured_node_bytes(n=2_000_000):
    """Cross-check with tracemalloc: allocate n real nodes."""
    tracemalloc.start()
    nodes = [Node(i + 1000) for i in range(n)]
    used, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del nodes
    return used / n


if __name__ == "__main__":
    print("-- the source's estimates, recomputed")
    miss = 1 * (1 / 250) * 120
    exact = 1 * (20 / 5280) * 120
    print(f"Mississippi  1 x 1/250 x 120 = {miss:.2f} mi^3/day"
          f"  (20/5280 mi deep: {exact:.3f})")
    for size in (8, 48):
        mb = 2_000_000 * size / MB
        print(f"2M nodes x {size:>2} B = {mb:>3.0f} MB"
              f" = {mb / 128:.1%} of 128 MB")
    print(f"n^3 sqrt, n=1000: 1e9 x 200 ns = {1e9 * 200e-9:.0f} s")
    print(f"1e6 s = {1e6 / DAY:.1f} days;"
          f" 1e9 s = {1e9 / YEAR:.1f} years")
    ft3 = 5e9 * 231 / 1728       # US gallon = 231 in^3
    print(f"5B gal blood = {ft3:.3g} ft^3,"
          f" cube side {ft3 ** (1 / 3):.0f} ft")
    print(f"Concorde / snail = 2000 / 0.005 = {2000 / 0.005:,.0f}x")

    print("-- Little's law, L = lambda x W")
    print("wine: W =", littles_law(rate=25, count=150), "years")
    print("club: lambda =", littles_law(time=3, count=60), "per hour")
    print("line: W =", littles_law(rate=20, count=20), "hour")

    print("-- Python's own overhead (CPython 3.10, 64-bit)")
    print("getsizeof: int", sys.getsizeof(1000),
          "float", sys.getsizeof(1.5),
          "tuple(a, b)", sys.getsizeof((1, 2)),
          "empty dict", sys.getsizeof({}))
    obj, num, ref, per = python_node_bytes()
    print(f"one node = {obj} obj + {num} int + {ref} ref = {per} B")
    print(f"2M nodes = {2_000_000 * per / MB:.0f} MB > 128 MB")
    print(f"tracemalloc: {measured_node_bytes():.1f} B per node")

    class Plain:                 # no __slots__: a __dict__ per node
        def __init__(self, val):
            self.val, self.next = val, None
    p = Plain(1000)
    other = {
        "tuple (val, next)": sys.getsizeof((1000, None)) + num + ref,
        "plain class": sys.getsizeof(p) + sys.getsizeof(p.__dict__)
        + num + ref,
    }
    for name, b in other.items():
        print(f"{name}: {b} B/node, 2M = {2_000_000 * b / MB:.0f} MB")

    print("-- TinyURL (illustrative inputs, 30-day month)")
    writes = 100_000_000 / (30 * DAY)
    reads = writes * 100
    records = 100_000_000 * 12 * 5
    print(f"writes/s {writes:.1f}  reads/s {reads:,.0f}")
    print(f"records {records:,}  storage {records * 500 / 1e12:.0f} TB")
    flight = littles_law(rate=reads, time=0.05)
    print(f"in flight at 50 ms: {flight:.0f}")
    print(f"derated 6x: {reads * 6:,.0f} reads/s")
