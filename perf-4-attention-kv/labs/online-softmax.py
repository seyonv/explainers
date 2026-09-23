"""Online softmax lab: Milakov & Gimelshein, "Online normalizer calculation for softmax" (arXiv 1805.02867).

Run: python3 labs/online-softmax.py   (numpy + stdlib, CPU, a few seconds)

Prints:
  1. why naive softmax (Alg 1) overflows and safe softmax (Eq 2) does not
  2. the worked example: Alg 3 traced on x = [3, 1, 5, 2], then the (+) merge of [3, 1] and [5, 2]
  3. Alg 1 / 2 / 3 and the parallel (+) version on a Llama-3.1-8B-sized vector (V = 128,256): max abs diff
  4. memory accesses per element for every variant, counted by instrumented code
  5. Alg 4, fused online softmax + top-k (K = 5), checked against np.argsort
  6. our recomputation: bytes and time of the sampling softmax at V = 128,256, FP32, on V100 / M3 / H100
  7. our recomputation: one attention score row of length N, and the full N x N matrix it avoids
"""
import math
import numpy as np

V = 128_256          # Llama-3.1-8B vocabulary (course facts)
FP32 = 4             # bytes per logit


class Mem:
    """A vector in 'global memory' that counts every element loaded or stored."""
    def __init__(self, data):
        self.data = data
        self.loads = 0
        self.stores = 0

    def load(self, i):
        self.loads += 1
        return self.data[i]

    def store(self, i, v):
        self.stores += 1
        self.data[i] = v


def naive(x, y):                         # Alg 1: 2 loads + 1 store per element
    n = len(x.data); d = 0.0
    for j in range(n):
        d += math.exp(x.load(j))
    for i in range(n):
        y.store(i, math.exp(x.load(i)) / d)


def safe(x, y):                          # Alg 2: 3 loads + 1 store per element
    n = len(x.data); m = -math.inf
    for k in range(n):
        m = max(m, x.load(k))
    d = 0.0
    for j in range(n):
        d += math.exp(x.load(j) - m)
    for i in range(n):
        y.store(i, math.exp(x.load(i) - m) / d)


def online_normalizer(x, n, trace=None):  # Alg 3, lines 1-6
    m, d = -math.inf, 0.0
    for j in range(n):
        xj = x.load(j)
        m_new = max(m, xj)
        d = d * math.exp(m - m_new) + math.exp(xj - m_new)
        m = m_new
        if trace is not None:
            trace.append((j + 1, xj, m, d))
    return m, d


def online(x, y):                        # Alg 3: 2 loads + 1 store per element
    n = len(x.data)
    m, d = online_normalizer(x, n)
    for i in range(n):
        y.store(i, math.exp(x.load(i) - m) / d)


def merge(a, b):                         # Eq 4: [m_i, d_i] (+) [m_j, d_j]
    (mi, di), (mj, dj) = a, b
    M = max(mi, mj)
    return M, di * math.exp(mi - M) + dj * math.exp(mj - M)


def partial(chunk):                      # what one thread block would compute for its chunk (vectorised)
    m = float(np.max(chunk))
    return m, float(np.sum(np.exp(chunk - m)))


def tree_reduce(pairs):                  # parallel evaluation of Eq 3: pairwise, like a GPU reduction
    while len(pairs) > 1:
        nxt = [merge(pairs[k], pairs[k + 1]) for k in range(0, len(pairs) - 1, 2)]
        if len(pairs) % 2:
            nxt.append(pairs[-1])
        pairs = nxt
    return pairs[0]


def online_topk_fused(x, K):             # Alg 4: 1 load per element, stores only K values
    n = len(x.data); m, d = -math.inf, 0.0
    u = [-math.inf] * (K + 1); p = [-1] * (K + 1)
    for j in range(n):
        xj = x.load(j)
        m_new = max(m, xj)
        d = d * math.exp(m - m_new) + math.exp(xj - m_new)
        m = m_new
        u[K], p[K] = xj, j
        k = K - 1
        while k >= 0 and u[k] < u[k + 1]:
            u[k], u[k + 1] = u[k + 1], u[k]
            p[k], p[k + 1] = p[k + 1], p[k]
            k -= 1
    return [math.exp(u[i] - m) / d for i in range(K)], p[:K]


print("== 1. Naive softmax overflows (Eq 1 / Alg 1); safe softmax (Eq 2) does not ==")
print(f"largest x with finite exp(x): fp32 {math.log(np.finfo(np.float32).max):.2f}, fp64 {math.log(np.finfo(np.float64).max):.2f}")
x = np.array([1000.0, 999.0, 998.0], dtype=np.float32)
with np.errstate(over="ignore", invalid="ignore"):
    e = np.exp(x)
    print(f"x = {x.tolist()}  exp(x) = {e.tolist()}  naive y = {(e / e.sum()).tolist()}")
s = np.exp(x - x.max())
print(f"safe: exp(x - 1000) = {[round(float(v), 6) for v in s]}  sum = {s.sum():.6f}  y = {[round(float(v), 6) for v in s / s.sum()]}")

print("\n== 2. Worked example: Alg 3 on x = [3, 1, 5, 2] ==")
xs = [3.0, 1.0, 5.0, 2.0]
tr = []
m4, d4 = online_normalizer(Mem(list(xs)), 4, tr)
print(" j  x_j  m_j   d_j = d_{j-1} * e^(m_{j-1}-m_j) + e^(x_j-m_j)")
prev_m, prev_d = -math.inf, 0.0
for j, xj, m, d in tr:
    resc = 0.0 if prev_d == 0 else prev_d * math.exp(prev_m - m)
    print(f" {j}  {xj:3.0f}  {m:3.0f}   {prev_d:.6f} * e^({'-inf' if prev_m == -math.inf else f'{prev_m - m:+.0f}'}) + e^({xj - m:+.0f}) "
          f"= {resc:.6f} + {math.exp(xj - m):.6f} = {d:.6f}")
    prev_m, prev_d = m, d
ref = sum(math.exp(v - max(xs)) for v in xs)
print(f"check (Theorem 1): m_V = max x = {max(xs):.0f};  sum e^(x_j - 5) = {ref:.6f}")
print(f"y = e^(x - 5) / {d4:.6f} = {[round(math.exp(v - m4) / d4, 6) for v in xs]}")
left = online_normalizer(Mem([3.0, 1.0]), 2)
right = online_normalizer(Mem([5.0, 2.0]), 2)
mm, dd = merge(left, right)
print(f"[3,1] -> (m, d) = ({left[0]:.0f}, {left[1]:.6f});  [5,2] -> ({right[0]:.0f}, {right[1]:.6f})")
print(f"(+): max(3,5) = {mm:.0f};  {left[1]:.6f}*e^(3-5) + {right[1]:.6f}*e^(5-5) = "
      f"{left[1] * math.exp(-2):.6f} + {right[1]:.6f} = {dd:.6f}  (sequential: {d4:.6f})")
print(f"bounds 1 <= d_j <= j hold: {all(1 <= d <= j for j, _, _, d in tr)}")

print(f"\n== 3. Same answer, V = {V:,} (Llama-3.1-8B vocab), synthetic logits ~ N(0, 3^2), seed 0 ==")
rng = np.random.default_rng(0)
logits = (rng.standard_normal(V) * 3).astype(np.float32)
ref64 = np.exp(logits.astype(np.float64) - logits.max()); ref64 /= ref64.sum()
results = {}
for name, fn in (("naive  (Alg 1)", naive), ("safe   (Alg 2)", safe), ("online (Alg 3)", online)):
    xm, ym = Mem(logits.astype(np.float64).tolist()), Mem([0.0] * V)
    fn(xm, ym)
    results[name] = np.array(ym.data)
safe_y = results["safe   (Alg 2)"]
for name, yv in results.items():
    print(f"{name}: max |y - safe| = {np.max(np.abs(yv - safe_y)):.2e}   max |y - fp64 numpy ref| = {np.max(np.abs(yv - ref64)):.2e}")
m_seq, d_seq = online_normalizer(Mem(logits.astype(np.float64).tolist()), V)
for chunks in (132, 1024):
    parts = [partial(c) for c in np.array_split(logits.astype(np.float64), chunks)]
    m_par, d_par = tree_reduce(parts)
    y_par = np.exp(logits.astype(np.float64) - m_par) / d_par
    print(f"parallel (+) over {chunks:4d} chunks: m = {m_par:.4f} (sequential {m_seq:.4f}), d = {d_par:.6f} (sequential {d_seq:.6f}), "
          f"max |y - safe| = {np.max(np.abs(y_par - safe_y)):.2e}")
# float32 end to end: the realistic GPU dtype
x32 = logits
s32 = np.exp(x32 - x32.max()); s32 = s32 / s32.sum(dtype=np.float32)
parts32 = [(np.float32(c.max()), np.exp(c - c.max()).sum(dtype=np.float32)) for c in np.array_split(x32, 132)]
M = max(p[0] for p in parts32)
D = np.float32(sum(np.float32(d * np.exp(np.float32(m - M))) for m, d in parts32))
o32 = np.exp(x32 - M) / D
print(f"fp32 end to end, safe vs online-(+) over 132 chunks: max abs diff = {np.max(np.abs(o32 - s32)):.2e} "
      f"(largest y = {s32.max():.2e}, fp32 epsilon = {np.finfo(np.float32).eps:.2e})")

print("\n== 4. Memory accesses per element (counted, V = 4,000) ==")
small = rng.standard_normal(4000).tolist()
rows = []
for name, fn in (("naive  (Alg 1)", naive), ("safe   (Alg 2)", safe), ("online (Alg 3)", online)):
    xm, ym = Mem(list(small)), Mem([0.0] * 4000)
    fn(xm, ym)
    rows.append((name, xm.loads / 4000, ym.stores / 4000))
for name, l, s_ in rows:
    print(f"{name}: {l:.0f} loads + {s_:.0f} store = {l + s_:.0f} accesses/element")
print(f"safe / online = 4 / 3 = {4 / 3:.2f}x fewer accesses (paper: 1.33x)")
xm = Mem(list(small)); online_topk_fused(xm, 5)
print(f"top-k (K=5): safe then top-k = 4 + 1 = 5;  online then top-k = 3 + 1 = 4;  fused Alg 4 = {xm.loads / 4000:.0f} load, "
      f"stores only K = 5 values -> 5x fewer than safe unfused (paper: 2.5x fusion x 2x online)")

print("\n== 5. Alg 4, fused online softmax + top-k, K = 5, on the V = 128,256 logits ==")
vals, idx = online_topk_fused(Mem(logits.astype(np.float64).tolist()), 5)
ref_idx = np.argsort(-safe_y, kind="stable")[:5]
print(f"Alg 4 indices    : {idx}")
print(f"np.argsort(safe) : {ref_idx.tolist()}   same: {idx == ref_idx.tolist()}")
print(f"Alg 4 probs      : {[f'{v:.6e}' for v in vals]}")
print(f"max |prob - safe[idx]| = {max(abs(v - safe_y[i]) for v, i in zip(vals, idx)):.2e}")

print("\n== 6. Our recomputation: Llama-3.1-8B sampling softmax, V = 128,256 FP32 logits, one sequence ==")
vec = V * FP32
print(f"one pass over the logits = {V:,} x 4 B = {vec:,} B = {vec / 1e6:.3f} MB")
bw = {"V100 PCIe (paper's GPU)": 900e9, "Apple M3 (reader's Mac)": 100e9, "H100 SXM (running example)": 3.35e12}
for label, acc in (("safe, 4 accesses", 4), ("online, 3 accesses", 3), ("safe + top-k unfused, 5", 5), ("online + top-k fused, 1", 1)):
    t = "  ".join(f"{k.split(' (')[0]} {acc * vec / b * 1e6:6.2f} us" for k, b in bw.items())
    print(f"{label:26s} {acc * vec / 1e6:.3f} MB   {t}")
batch = 59
print(f"at batch {batch} (59 sequences of 8k, course facts), H100: safe {4 * vec * batch / 3.35e12 * 1e6:.1f} us vs online "
      f"{3 * vec * batch / 3.35e12 * 1e6:.1f} us, saves {vec * batch / 3.35e12 * 1e6:.1f} us per decode step")
lm_head = V * 4096 * 2
print(f"for scale: the LM head weights it follows are {V:,} x 4096 x 2 B = {lm_head / 1e9:.2f} GB "
      f"-> {lm_head / 3.35e12 * 1e3:.3f} ms on H100; softmax is {4 * vec / lm_head * 100:.2f}% of that traffic at batch 1")
print(f"the whole batch-1 decode step is ~4.8 ms (16.06 GB / 3.35 TB/s): safe softmax is {4 * vec / 3.35e12 / 4.8e-3 * 100:.3f}% of it")

print("\n== 7. Our recomputation: attention rows. One head, fp32 scores, N = 8,192 ==")
N = 8192
row = N * FP32
full = N * N * FP32
print(f"one score row  = {N:,} x 4 B = {row / 1024:.0f} KiB (fits in the 256 KB SMEM of one SM)")
print(f"full N x N     = {N:,}^2 x 4 B = {full / 1e6:.1f} MB per head per layer; x 32 heads = {32 * full / 1e9:.2f} GB per layer")
print(f"safe softmax over a stored score matrix: write S, 3 reads, write P = 5 x {full / 1e6:.1f} MB = {5 * full / 1e6:.0f} MB per head "
      f"-> {5 * full / 3.35e12 * 1e6:.0f} us on H100")
print("online + (+): stream K in blocks, keep (m, d) per row in registers -> S is never written (FlashAttention, next card)")

# Try this:
# 1. x = np.array([1000., 1001.], dtype=np.float64) through naive(): does fp64 save you? (exp limit is 709.78)
# 2. Change `chunks` to 7 or 100_000 in section 3: the (+) result stays equal to rounding, in any grouping (associativity).
# 3. Set K = 30 in section 5 and time online_topk_fused: the insertion loop grows with K, the paper's "1.4x at K = 30".
