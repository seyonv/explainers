"""Reduction cost lab: Harris, "Optimizing Parallel Reduction in CUDA", slides 16-38.

Run: python3 labs/reduction-cost.py   (stdlib + numpy, CPU, under a second)

What it does (a simulator, not a GPU benchmark):
  1. Replays the slides' 7-row table and turns each bandwidth into % of the G80's 86.4 GB/s.
  2. Walks the in-block tree for a 128-thread block, step by step: active threads, active
     warps, idle lanes. Once for kernel 3/4's shape ("load one, then tree") and once with
     "first add during load" (kernel 4 onward: each thread adds two elements while loading).
  3. Counts loop iterations executed by warps with and without unrolling the last warp
     (kernels 4 vs 5). This is our model of "loop overhead", not a SASS instruction count.
  4. Runs the slides' complexity numbers for N = 2^22: steps, operations, and Brent's cost
     (processors x time) for several thread counts P.
  5. Checks with numpy that the tree, the cascaded version and a warp-shuffle tree all give
     the exact sum.
"""
import math
import numpy as np

G80_BW = 86.4  # GB/s, slides: 384-bit bus, 900 MHz DDR
N = 2 ** 22    # the slides' 4M ints
BYTES = N * 4

# ---------------------------------------------------------------- 1. the slides' table
TABLE = [  # kernel, name, ms, GB/s, step speedup (as printed), kind
    (1, "interleaved addressing, divergent branching", 8.054, 2.083, None, "-"),
    (2, "interleaved addressing, bank conflicts", 3.456, 4.854, 2.33, "algorithmic"),
    (3, "sequential addressing", 1.722, 9.741, 2.01, "algorithmic"),
    (4, "first add during global load", 0.965, 17.377, 1.78, "algorithmic"),
    (5, "unroll last warp", 0.536, 31.289, 1.8, "code"),
    (6, "completely unrolled (templates)", 0.381, 43.996, 1.41, "code"),
    (7, "multiple elements per thread", 0.268, 62.671, 1.42, "algorithmic"),
]
print("1. Harris's table (G80, 2^22 ints) and % of the 86.4 GB/s peak")
print(f"   {'#':>2} {'ms':>6} {'GB/s':>7} {'% peak':>7} {'step':>6} {'cumul (t1/t)':>13}  {'bytes/time':>10}")
for k, name, ms, gbs, step, kind in TABLE:
    pct = 100 * gbs / G80_BW
    cum = TABLE[0][2] / ms
    ours = BYTES / (ms * 1e-3) / 1e9
    s = f"{step:.2f}x" if step else "-"
    print(f"   {k:>2} {ms:>6.3f} {gbs:>7.3f} {pct:>6.1f}% {s:>6} {cum:>12.2f}x  {ours:>9.2f}  {name}")
print(f"   kernel 7 on 32M elements: 73 GB/s = {100 * 73 / G80_BW:.1f}% of peak")
print(f"   bytes read: 2^22 x 4 B = {BYTES:,} B = {BYTES / 1e6:.2f} MB (bytes/time column is our recomputation)")
alg = 2.33 * 2.01 * 1.78 * 1.42
code = 1.8 * 1.41
print(f"   algorithmic steps 2.33 x 2.01 x 1.78 x 1.42 = {alg:.2f}x   (slides: 11.84x)")
print(f"   code steps        1.8 x 1.41               = {code:.2f}x    (slides: 2.54x)")
print(f"   all six step speedups multiplied = {alg * code:.2f}x vs 8.054/0.268 = {8.054 / 0.268:.2f}x (slides: 30.04x; rounding)")
print()

# ---------------------------------------------------------------- 2. idle lanes per step
B = 128  # threads per block, as in the slides


def tree_steps(first_add):
    """Rows: (label, active threads, active warps, adds). One block of B threads."""
    rows = []
    if first_add:
        rows.append(("load: a[i] + a[i+128]", B, B // 32, B))
    s = B // 2
    while s > 0:
        rows.append((f"s = {s}", s, math.ceil(s / 32), s))
        s //= 2
    return rows


print("2. One 128-thread block (4 warps): who is busy at each step")
for first_add in (False, True):
    rows = tree_steps(first_add)
    elems = 2 * B if first_add else B
    title = "kernel 4+: first add during load" if first_add else "kernel 3: load one element, then tree"
    print(f"   {title}  ({elems} elements per block, {N // elems:,} blocks for 2^22)")
    for label, act, warps, adds in rows:
        print(f"     {label:<22} active threads {act:>3}/128  warps with work {warps}/4  idle lanes {100 * (1 - act / B):5.1f}%")
    tot_adds = sum(r[3] for r in rows)
    slots = len(rows) * B
    print(f"     adds {tot_adds}, steps {len(rows)}, busy lane-steps {tot_adds}/{slots} = {100 * tot_adds / slots:.1f}%")
print()

# ---------------------------------------------------------------- 3. loop overhead, kernels 4 vs 5
steps_all = int(math.log2(B))                     # s = 64 .. 1 -> 7 iterations
k4_iters = steps_all * (B // 32)                  # every warp runs every iteration (loop + if + __syncthreads)
k5_loop = sum(1 for s in [64, 32, 16, 8, 4, 2, 1] if s > 32) * (B // 32)
k5_warp = 6                                       # warpReduce: 6 unrolled adds, one warp, no loop, no sync
print("3. Loop iterations executed by warps (our model of loop overhead), 128-thread block")
print(f"   kernel 4: {steps_all} iterations x 4 warps = {k4_iters} warp-iterations, each with compare, branch, __syncthreads")
print(f"   kernel 5: loop only while s > 32 -> 1 x 4 warps = {k5_loop} warp-iterations, then 6 straight-line adds in warp 0")
print(f"   __syncthreads per block: kernel 4 = {steps_all}, kernel 5 = 1, kernel 6 = 1 (and no loop counter at all)")
print()

# ---------------------------------------------------------------- 4. complexity and Brent
D = int(math.log2(N))
ops = sum(2 ** (D - s) for s in range(1, D + 1))
print(f"4. Tree reduction of N = 2^{D} = {N:,}")
print(f"   steps = log2 N = {D};  operations = sum 2^(D-S), S=1..{D} = {ops:,} = N-1 (work-efficient)")


def time_exact(P):
    """Each thread first adds ceil(N/P) elements serially, then a tree over P partial sums."""
    per = math.ceil(N / P)
    return (per - 1) + math.ceil(math.log2(P)) if P > 1 else N - 1


print("   Brent: cost = P x time.  Two time models:")
print("     big-O  time = N/P + log2 N              (the slides' O(N/P + log N), constants = 1)")
print("     exact  time = (ceil(N/P) - 1) + ceil(log2 P)   (serial adds, then the tree)")
print(f"   {'P':>24} {'elems/thread':>12} {'bigO time':>10} {'bigO cost':>14} {'exact time':>10} {'exact cost':>14} {'cost / N':>9}")
Ps = [("N", N), ("N/log2 N", N / D), ("2^17", 2 ** 17), ("2^13 = 64x128", 2 ** 13), ("2^10", 2 ** 10), ("1", 1)]
for label, P in Ps:
    Pi = math.ceil(P)
    bt = N / P + D
    bc = P * bt
    et = time_exact(Pi)
    ec = Pi * et
    print(f"   {label:>14} {Pi:>9,} {math.ceil(N / Pi):>12,} {bt:>10.1f} {bc:>14,.0f} {et:>10,} {ec:>14,} {ec / N:>8.2f}x")
print(f"   P = N:         cost = N log2 N = {N:,} x {D} = {N * D:,}")
print(f"   P = N/{D} = {N / D:,.2f}: big-O cost = {N / D:,.0f} x ({D} + {D}) = {N / D * 2 * D:,.0f} = 2N")
print(f"   ratio of costs: {N * D / (N / D * 2 * D):.1f}x less machine-time for the same sum")
print(f"   the slides' G80 config at 2^22: 64 blocks x 128 threads = 8,192 threads -> {N // 8192} elements each"
      f"; at 32M = 2^25: {2 ** 25 // 8192:,} each")
print()

# ---------------------------------------------------------------- 5. correctness check in numpy
rng = np.random.default_rng(0)
x = rng.integers(0, 100, size=N, dtype=np.int64)


def block_tree(v):
    v = v.copy()
    s = len(v) // 2
    while s > 0:
        v[:s] += v[s:2 * s]   # sequential addressing: thread tid < s adds tid + s
        s //= 2
    return v[0]


blk = x[: 2 * B]
k4 = block_tree(blk[:B] + blk[B:])               # first add during load
P = 64 * B                                        # kernel 7: 64 blocks of 128 threads, grid-stride
grid = 2 * P                                      # each loop trip reads a[i] and a[i+blockSize]
partial = x.reshape(-1, grid).sum(axis=0)         # element i goes to the slot (i mod grid)
per_thread = np.zeros(P, dtype=np.int64)
for b in range(64):                               # map slot -> (block, tid) as in the kernel
    base = b * 2 * B
    per_thread[b * B:(b + 1) * B] = partial[base:base + B] + partial[base + B:base + 2 * B]
block_sums = [block_tree(per_thread[b * B:(b + 1) * B]) for b in range(64)]
lanes = x[:32].copy()                             # __shfl_down_sync tree over one warp
for off in (16, 8, 4, 2, 1):
    shifted = np.concatenate([lanes[off:], lanes[32 - off:]])  # lane X reads lane X+off (past 31: its own value)
    lanes = lanes + shifted  # only lane 0's final value is used
print("5. Exact-sum checks (numpy)")
print(f"   kernel 4 block (256 elements): tree {k4} == numpy {blk.sum()}: {k4 == blk.sum()}")
print(f"   kernel 7 (64 blocks, grid-stride, 2^22): {sum(block_sums):,} == numpy {x.sum():,}: {sum(block_sums) == x.sum()}")
print(f"   warp shuffle tree (32 lanes, offsets 16,8,4,2,1): lane 0 = {lanes[0]} == {x[:32].sum()}: {lanes[0] == x[:32].sum()}")

# Try this:
# - Set B = 256 or 512 in section 2: the "first add during load" step is still 100% busy,
#   but the tree gets one more mostly-idle step for every doubling.
# - Add ("2^20", 2**20) and ("2^15", 2**15) to Ps: cost stays near N until P passes N/log N, then grows.
# - Change the 64 in section 5 to 256 blocks: the sum is still exact; on a GPU the choice only changes speed.
