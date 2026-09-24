"""A minimal reverse-mode autodiff tape (AAD), checked and stretched.

1. Var: each value remembers its parents and the local derivative
   to each one. backward() orders the graph topologically and runs
   the chain rule in reverse: adjoint(parent) += adjoint(child) *
   d child / d parent.
2. Check f(x, y) = x*y + sin(x) at (2, 3) against finite differences.
3. Bump-and-reprice cost (1 + N valuations) vs one forward + one
   backward pass.
4. A strict unary chain (the source's case) with sqrt(n)
   checkpointing: same gradient, far fewer stored values.
5. A digital payoff: the pathwise gradient is 0; smoothing and the
   likelihood-ratio trick recover Delta.
6. SegTape: the source's tape as a doubly linked list of fixed-size
   segments: append, drop the oldest, cut everything downstream.
"""
import math
import random


class Var:
    def __init__(self, val, parents=()):
        self.val = val
        self.parents = parents      # tuple of (Var, local gradient)
        self.grad = 0.0             # the adjoint, filled by backward

    def __add__(self, o):
        return Var(self.val + o.val, ((self, 1.0), (o, 1.0)))

    def __mul__(self, o):
        return Var(self.val * o.val, ((self, o.val), (o, self.val)))


def sin(a):
    return Var(math.sin(a.val), ((a, math.cos(a.val)),))


def backward(out):
    order, seen = [], set()

    def visit(v):                   # DFS post-order = topological
        if id(v) not in seen:
            seen.add(id(v))
            for p, _ in v.parents:
                visit(p)
            order.append(v)

    visit(out)
    out.grad = 1.0                  # seed: d out / d out
    for v in reversed(order):       # successors before predecessors
        for p, local in v.parents:
            p.grad += v.grad * local  # chain rule, accumulated
    return order


def f_plain(x, y):
    return x * y + math.sin(x)


# ---- 4. unary chain with sqrt(n) checkpoints ----------------------

DT = 0.001


def step(x):                        # one link of the chain
    return x + DT * math.sin(x)


def dstep(x):                       # its local derivative
    return 1 + DT * math.cos(x)


def grad_full_tape(x0, n):
    tape = [x0]
    for _ in range(n):
        tape.append(step(tape[-1]))  # store every value
    g = 1.0
    for k in range(n - 1, -1, -1):
        g *= dstep(tape[k])
    return tape[-1], g, len(tape), n


def grad_checkpointed(x0, n):
    s = math.isqrt(n)               # segment length ~ sqrt(n)
    ckpts, x, steps = [], x0, 0
    for k in range(n):
        if k % s == 0:
            ckpts.append(x)         # keep only segment starts
        x = step(x)
        steps += 1
    out, g, peak = x, 1.0, 0
    for c in range(len(ckpts) - 1, -1, -1):
        lo = c * s
        seg = [ckpts[c]]            # recompute this segment
        for _ in range(lo, min(lo + s, n) - 1):
            seg.append(step(seg[-1]))
            steps += 1
        peak = max(peak, len(ckpts) + len(seg))
        for xk in reversed(seg):
            g *= dstep(xk)
    return out, g, peak, steps


# ---- 5. digital payoff 1{S_T > K} ---------------------------------

def digital_deltas(s0=100.0, k=100.0, vol=0.2, t=1.0, n=200_000,
                   eps=1.0, seed=0):
    rng = random.Random(seed)
    sq = vol * math.sqrt(t)
    path = smooth = lr = 0.0
    for _ in range(n):
        z = rng.gauss(0.0, 1.0)
        st = s0 * math.exp(-0.5 * sq * sq + sq * z)
        dst = st / s0               # d S_T / d S_0
        path += 0.0 * dst           # d 1{S_T > K} / d S_T = 0 a.e.
        u = (st - k) / eps          # sigmoid of width eps
        sig = 1 / (1 + math.exp(-u)) if u > -700 else 0.0
        smooth += sig * (1 - sig) / eps * dst
        lr += (st > k) * z / (s0 * sq)  # payoff * d log p / d S_0
    d2 = (math.log(s0 / k) - 0.5 * sq * sq) / sq
    exact = math.exp(-0.5 * d2 * d2) / math.sqrt(2 * math.pi)
    exact /= s0 * sq
    return path / n, smooth / n, lr / n, exact


# ---- 6. the source's tape: linked segments of a unary chain -------

class Segment:
    def __init__(self, start):
        self.start = start          # global index of first node
        self.vals = []              # inputs to each op, in order
        self.ckpt = None            # kept when vals are freed
        self.prev = self.next = None


class SegTape:
    """Append-only; drop oldest; cut a suffix; walk in reverse."""

    def __init__(self, seg_len):
        self.seg_len, self.head = seg_len, None
        self.tail, self.n = None, 0

    def append(self, x):            # O(1)
        t = self.tail
        if t is None or len(t.vals) == self.seg_len:
            s = Segment(self.n)
            s.ckpt, s.prev = x, t
            if t:
                t.next = s
            else:
                self.head = s
            self.tail = t = s
        t.vals.append(x)
        self.n += 1

    def drop_oldest(self):          # O(1): sliding window
        self.head = self.head.next
        self.head.prev = None

    def truncate_after(self, i):    # O(1) + O(seg_len): barrier hit
        s = self.tail
        while s.start > i:
            s = s.prev
        del s.vals[i - s.start + 1:]
        s.next, self.tail, self.n = None, s, i + 1

    def segments_reversed(self):    # backward walks tail -> head
        s = self.tail
        while s:
            yield s
            s = s.prev


if __name__ == "__main__":
    x, y = Var(2.0), Var(3.0)
    a = x * y
    b = sin(x)
    out = a + b
    order = backward(out)
    print("tape (forward order):")
    names = {id(x): "x", id(y): "y", id(a): "a=x*y",
             id(b): "b=sin(x)", id(out): "f=a+b"}
    for v in order:
        print(f"  {names[id(v)]:9} val={v.val:.6f} "
              f"adjoint={v.grad:.6f}")
    h = 1e-6
    fx = (f_plain(2 + h, 3) - f_plain(2 - h, 3)) / (2 * h)
    fy = (f_plain(2, 3 + h) - f_plain(2, 3 - h)) / (2 * h)
    print(f"f = {out.val:.6f}")
    print(f"tape:  df/dx = {x.grad:.9f}  df/dy = {y.grad:.9f}")
    print(f"FD:    df/dx = {fx:.9f}  df/dy = {fy:.9f}")
    print(f"exact: df/dx = {3 + math.cos(2):.9f}  df/dy = 2")
    print(f"|tape - FD| = {abs(x.grad - fx):.1e}, "
          f"{abs(y.grad - fy):.1e}")

    print("\nvaluations for all N sensitivities:")
    for n_par in (2, 100, 500):
        print(f"  N={n_par:4}: bump {1 + n_par:4} (one-sided), "
              f"{2 * n_par:5} (central); AAD 1 fwd + 1 bwd")

    n = 10_000
    o1, g1, st1, fw1 = grad_full_tape(0.5, n)
    o2, g2, st2, fw2 = grad_checkpointed(0.5, n)
    print(f"\nunary chain, n = {n:,}, x0 = 0.5")
    print(f"  full tape:    dx_n/dx0 = {g1:.12f}  "
          f"stored {st1:,}  forward steps {fw1:,}")
    print(f"  checkpointed: dx_n/dx0 = {g2:.12f}  "
          f"stored {st2:,}  forward steps {fw2:,}")
    print(f"  same output: {o1 == o2}, |diff grad| = {abs(g1-g2):.1e}")

    p, s, l, e = digital_deltas()
    print("\ndigital call Delta, S0=K=100, vol 20%, T=1, r=0, "
          "200k paths:")
    print(f"  pathwise AD      {p:.5f}")
    print(f"  sigmoid, eps=1   {s:.5f}")
    print(f"  likelihood ratio {l:.5f}")
    print(f"  exact (closed)   {e:.5f}")

    tp = SegTape(seg_len=3)
    for k in range(12):
        tp.append(k)
    tp.drop_oldest()
    tp.truncate_after(7)
    segs = list(tp.segments_reversed())[::-1]
    print("\nSegTape: 12 appends, seg_len 3, drop oldest, cut after 7:")
    print("  " + " <-> ".join(str(s.vals) for s in segs))
