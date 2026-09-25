"""Limit order book: the source guide's math spec turned into a system.

Prices live on a tick grid P = {k * delta}, so the engine stores the
integer k ("ticks"), never a float. Submission time t becomes a
sequence number. Each side is a dict tick -> FIFO deque of orders plus
a heap of ticks (negated for bids), so the best price is the heap top.
Run: python3 order-book-to-system.py   (Python 3.10, stdlib only)
"""
import heapq
from collections import deque
from itertools import count

TICK = 0.01  # delta, the tick size (illustrative)


class Book:
    def __init__(self):
        self.levels = {"B": {}, "S": {}}  # tick -> deque of orders
        self.heaps = {"B": [], "S": []}   # -tick for bids, tick asks
        self.orders = {}                  # id -> [id, side, tick, qty]
        self.seq = count(1)               # t becomes a counter

    def best(self, side):                 # b(t) or a(t)
        h, lv = self.heaps[side], self.levels[side]
        while h:
            tick = -h[0] if side == "B" else h[0]
            q = lv.get(tick)
            while q and q[0][3] == 0:     # skip cancelled orders
                q.popleft()
            if q:
                return tick
            lv.pop(tick, None)            # empty level: drop lazily
            heapq.heappop(h)
        return None

    def submit(self, side, qty, tick=None):   # tick=None: market
        oid, opp, fills = next(self.seq), "SB"[side == "S"], []
        while qty:
            b = self.best(opp)
            if b is None or (tick is not None and
                             (b > tick if side == "B" else b < tick)):
                break                     # nothing left that crosses
            o = self.levels[opp][b][0]    # oldest order at best price
            n = min(qty, o[3])
            o[3] -= n
            qty -= n
            fills.append((o[0], b, n))
        if qty and tick is not None:      # rest the remainder
            lv = self.levels[side]
            if tick not in lv:
                lv[tick] = deque()
                heapq.heappush(self.heaps[side],
                               -tick if side == "B" else tick)
            o = [oid, side, tick, qty]
            lv[tick].append(o)
            self.orders[oid] = o
        return oid, fills

    def cancel(self, oid):                # O(1): mark, skip later
        self.orders[oid][3] = 0

    def depth(self, side):
        rows = []
        for tick in sorted(self.levels[side], reverse=side == "S"):
            ids = [(o[0], o[3]) for o in self.levels[side][tick]
                   if o[3]]
            if ids:
                rows.append((tick, ids))
        return rows


def px(tick):
    return f"{tick * TICK:.2f}"


def start_book():
    bk = Book()
    for side, qty, price in [("S", 60, 10.01), ("S", 100, 10.02),
                             ("S", 200, 10.03), ("B", 200, 10.00),
                             ("S", 40, 10.01), ("B", 150, 9.99)]:
        bk.submit(side, qty, round(price / TICK))  # ids #1..#6
    return bk


def show(bk, title):
    print(title)
    for tick, ids in bk.depth("S"):
        print("  ask", px(tick), ids)
    for tick, ids in sorted(bk.depth("B"), reverse=True):
        print("  bid", px(tick), ids)
    a, b = bk.best("S"), bk.best("B")
    print(f"  best bid {px(b)}  best ask {px(a)}  "
          f"spread {a - b} tick(s) = {px(a - b)}")


def report(fills, arrival):
    qty = sum(n for _, _, n in fills)
    cost = sum(t * n for _, t, n in fills) * TICK
    for oid, t, n in fills:
        print(f"  fill #{oid}: {n:3} @ {px(t)}")
    avg = cost / qty
    slip = avg - arrival * TICK
    print(f"  {qty} shares, cost {cost:.2f}, avg {avg:.4f}, "
          f"slippage {slip:.4f}/share = {slip * qty:.2f}")


def brute_best(bk, side):
    live = [o[2] for o in bk.orders.values() if o[1] == side and o[3]]
    if not live:
        return None
    return max(live) if side == "B" else min(live)


def fuzz(trials=2000):
    import random
    rng = random.Random(1)
    for _ in range(trials):
        bk = Book()
        for _ in range(30):
            r = rng.random()
            if r < 0.15 and bk.orders:
                bk.cancel(rng.choice(list(bk.orders)))
            else:
                side = rng.choice("BS")
                tick = None if r < 0.3 else rng.randint(995, 1005)
                bk.submit(side, rng.randint(1, 50), tick)
            for s in "BS":
                assert bk.best(s) == brute_best(bk, s)
            b, a = bk.best("B"), bk.best("S")
            assert b is None or a is None or b < a  # never crossed
    return trials


if __name__ == "__main__":
    bk = start_book()
    show(bk, "Start (orders #1..#6, [(id, qty)] in FIFO order):")

    print("\nMarket buy 250 (#7) walks the asks:")
    bk = start_book()
    arrival = bk.best("S")
    oid, fills = bk.submit("B", 250)
    report(fills, arrival)
    show(bk, "After:")

    print("\nMarket buy 50 fits in the best level:")
    bk = start_book()
    arrival = bk.best("S")
    report(bk.submit("B", 50)[1], arrival)

    print("\nLimit buy 50 @ 10.00 rests behind #4:")
    bk = start_book()
    print("  fills", bk.submit("B", 50, 1000)[1])
    print("  bid 10.00 queue", bk.depth("B")[-1])

    print("\nLimit buy 250 @ 10.02 crosses, then rests 50:")
    bk = start_book()
    oid, fills = bk.submit("B", 250, 1002)
    report(fills, 1001)
    show(bk, "After:")

    print("\nCancel #1 (head of 10.01):")
    bk = start_book()
    bk.cancel(1)
    show(bk, "After:")

    print("\nCancel #4 (the whole 10.00 level):")
    bk = start_book()
    bk.cancel(4)
    show(bk, "After:")

    n = fuzz()
    print(f"\nfuzz: best() matches brute force in {n} random books")
