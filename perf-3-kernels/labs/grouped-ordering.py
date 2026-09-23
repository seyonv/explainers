"""Grouped launch ordering for a tiled matmul (a calculator, no GPU needed).

Reproduces the "90 vs 54 blocks" example from Triton tutorial 03
(https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
and generalises it.

Model: C = A @ B is cut into a grid of output tiles (num_pid_m x num_pid_n).
Program `pid` computes one tile. It walks K in K_BLOCKS steps, loading one
A block (row strip pid_m) and one B block (column strip pid_n) per step.
The first W programs in launch order run at the same time (W = number of
programs resident at once, e.g. one per SM). We count the DISTINCT A and B
blocks they touch: if L2 keeps a block after its first load, that is how
many blocks must come from DRAM. Each program still copies all of its own
2 x K_BLOCKS blocks into its own shared memory, whatever the order.
Section 3 counts whole K strips: that is everything the wave touches over its
life, an upper bound on what L2 must hold at once (programs that start
together also move through K roughly together, so the live set is smaller).

pid -> (pid_m, pid_n) uses the tutorial's code, line for line.
GROUP_SIZE_M = 1 gives exactly the row-major order.

Run: python3 labs/grouped-ordering.py
"""
import math

import numpy as np


def tile_of(pid, num_pid_m, num_pid_n, GROUP_SIZE_M):
    # Triton tutorial 03 (MIT License, (c) 2018-2020 Philippe Tillet, 2020-2022 OpenAI)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def footprint(W, num_pid_m, num_pid_n, g):
    """Distinct A row strips and B column strips touched by programs 0..W-1."""
    tiles = [tile_of(p, num_pid_m, num_pid_n, g) for p in range(W)]
    assert len(set(tiles)) == W  # every pid gets its own tile
    return len({m for m, _ in tiles}), len({n for _, n in tiles})


def rowmajor_tile(pid, num_pid_n):
    return pid // num_pid_n, pid % num_pid_n


# sanity: GROUP_SIZE_M = 1 is row-major, and the mapping is a permutation
for pid in range(32 * 56):
    assert tile_of(pid, 32, 56, 1) == rowmajor_tile(pid, 56)
assert len({tile_of(p, 30, 7, 8) for p in range(30 * 7)}) == 30 * 7  # uneven last group

print("0) Pointer arithmetic, tiny example: A is 8 x 8 row-major (stride_am = 8, stride_ak = 1),")
print("   BLOCK_SIZE_M = BLOCK_SIZE_K = 4, program pid_m = 1")
stride_am, stride_ak, BMx, BKx, pid_m = 8, 1, 4, 4, 1
offs_am = pid_m * BMx + np.arange(BMx)
offs_k = np.arange(BKx)
a_offs = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
print("   offs_am =", offs_am.tolist(), " offs_k =", offs_k.tolist())
print("   a_ptrs - a_ptr, k-step 0:", a_offs.tolist())
print("   after a_ptrs += BLOCK_SIZE_K * stride_ak:", (a_offs + BKx * stride_ak).tolist())
Mx, Kx, BMt, BKt = 1000, 1000, 128, 64
steps = -(-Kx // BKt)
last = Kx - (steps - 1) * BKt
print(f"   K tail: K = {Kx}, BLOCK_SIZE_K = {BKt} -> cdiv = {steps} steps; last step keeps {last} of {BKt} columns,"
      f" masks {BKt - last} (loaded as 0.0)")
tiles_m = -(-Mx // BMt)
first_last = (tiles_m - 1) * BMt
wrapped = (first_last + np.arange(BMt)) % Mx
n_wrap = int((first_last + np.arange(BMt) >= Mx).sum())
print(f"   M edge: M = {Mx}, BLOCK_SIZE_M = {BMt} -> {tiles_m} row tiles; the last covers rows {first_last}..{first_last+BMt-1};"
      f" '% M' wraps {n_wrap} of them to rows {wrapped[-n_wrap]}..{wrapped[-1]} (store is masked)")

print("\n1) The tutorial's 9 x 9 example (9 x 9 output blocks, K = 9 blocks, first 9 outputs)")
KB = 9
for name, g in [("row-major (GROUP_SIZE_M=1)", 1), ("grouped   (GROUP_SIZE_M=3)", 3)]:
    r, c = footprint(9, 9, 9, g)
    print(f"   {name}: {r} A strips x {KB} + {c} B strips x {KB} = {r*KB} + {c*KB} = {(r+c)*KB} distinct blocks")
print(f"   each program reads 2 x {KB} = {2*KB} blocks into its own SRAM: 9 x {2*KB} = {9*2*KB} block copies either way")
print("   grid, first 9 programs, g=3 (pid -> (m, n)):",
      " ".join(f"{p}:{tile_of(p, 9, 9, 3)}" for p in range(9)))

print("\n2) Generalisation: distinct blocks loaded by the first W programs")
print("   grid 32 x 32 output tiles, K = 32 blocks (e.g. 4096^2 with 128 x 128 tiles)")
GM = GN = KB = 32
gs = [1, 2, 4, 8, 16, 32]
print(f"   {'W':>5} " + "".join(f"{'g='+str(g):>12}" for g in gs) + "   best g ~ sqrt(W)")
for W, label in [(9, "tutorial"), (40, "T4, 40 SMs"), (132, "H100, 132 SMs")]:
    cells = []
    for g in gs:
        r, c = footprint(W, GM, GN, g)
        cells.append(f"{(r+c)*KB:>6} ({(r+c)*KB/W:4.1f})")
    print(f"   {W:>5} " + "".join(f"{x:>12}" for x in cells) + f"   sqrt={math.sqrt(W):.1f}  [{label}]")
print("   cell = distinct blocks (blocks per output tile). g=1 is row-major; g=32 is column-major.")
for W in (40, 132):
    r1, c1 = footprint(W, GM, GN, 1)
    r8, c8 = footprint(W, GM, GN, 8)
    print(f"   W={W}: row-major {r1} rows + {c1} cols = {(r1+c1)*KB} blocks; g=8 {r8} rows + {c8} cols = {(r8+c8)*KB}"
          f" -> {(r1+c1)/(r8+c8):.2f}x fewer")

print("\n3) Running example: Llama-3.1-8B prefill up-projection on one H100 (our recomputation)")
M, N, K = 4096, 14336, 4096       # 4,096 tokens x D=4096  @  4096 x F=14336
BM, BN, BYTES = 128, 256, 2       # tutorial's first config, BF16
W = 132                           # one program per SM (assumption)
L2 = 50e6
gm, gn = M // BM, N // BN
a_strip = BM * K * BYTES          # one row strip of A, all of K
b_strip = K * BN * BYTES          # one column strip of B, all of K
print(f"   tiles: {M}/{BM} x {N}/{BN} = {gm} x {gn} = {gm*gn:,} output tiles; waves of {W}: {gm*gn/W:.1f}")
print(f"   A strip = {BM} x {K} x 2 B = {a_strip/2**20:.0f} MiB ; B strip = {K} x {BN} x 2 B = {b_strip/2**20:.0f} MiB")
flops_wave = W * 2 * BM * BN * K
no_reuse = W * (a_strip + b_strip)
print(f"   no reuse at all: {W} x ({a_strip/2**20:.0f} + {b_strip/2**20:.0f}) MiB = {no_reuse/1e6:,.0f} MB")
for name, g in [("row-major", 1), ("GROUP_SIZE_M=8", 8), ("GROUP_SIZE_M=16", 16)]:
    r, c = footprint(W, gm, gn, g)
    b = r * a_strip + c * b_strip
    fit = "fits in" if b <= L2 else "exceeds"
    print(f"   {name:<16}: {r:>2} A strips + {c:>2} B strips = {r}x1 + {c}x2 MiB = {b/2**20:.0f} MiB = {b/1e6:.1f} MB"
          f"  ({fit} 50 MB L2)  FLOP per DRAM byte = {flops_wave/b:.0f}")
print(f"   FLOPs per wave = {W} x 2 x {BM} x {BN} x {K} = {flops_wave:.3e};  H100 ridge = 989e12/3.35e12 = {989e12/3.35e12:.0f}")
tile_ai = BM * BN / (BM + BN)  # 2*BM*BN*BK flops / ((BM+BN)*BK*2 bytes)
print(f"   one program alone, every block from DRAM: 2*BM*BN*BK / ((BM+BN)*BK*2 B) = BM*BN/(BM+BN) = {tile_ai:.1f} FLOP/B")

print("\n4) The tutorial's 16 CUDA autotune configs: tile intensity and a rough shared-memory estimate")
CFG = [(128, 256, 64, 3, 8), (64, 256, 32, 4, 4), (128, 128, 32, 4, 4), (128, 64, 32, 4, 4),
       (64, 128, 32, 4, 4), (128, 32, 32, 4, 4), (64, 32, 32, 5, 2), (32, 64, 32, 5, 2),
       (128, 256, 128, 3, 8), (256, 128, 128, 3, 8), (256, 64, 128, 4, 4), (64, 256, 128, 4, 4),
       (128, 128, 128, 4, 4), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 32, 64, 4, 4)]
print(f"   {'BM':>4}{'BN':>5}{'BK':>5}{'stg':>4}{'warps':>6}{'thr':>5}{'out/thr':>8}{'FLOP/B':>8}{'SMEM est':>10}  T4 (64 KB)")
for i, (bm, bn, bk, st, nw) in enumerate(CFG):
    smem = st * (bm * bk + bk * bn) * 2  # fp16; Triton's real allocation can differ
    fit = "fits" if smem < 64 * 1024 else ("borderline" if smem == 64 * 1024 else "too big")
    print(f"   {bm:>4}{bn:>5}{bk:>5}{st:>4}{nw:>6}{nw*32:>5}{bm*bn//(nw*32):>8}{bm*bn/(bm+bn):>8.1f}{smem/1024:>8.0f} KB"
          f"  {fit}")
print("   SMEM est = num_stages x (BM*BK + BK*BN) x 2 bytes (fp16). A rough estimate, not Triton's exact number;")
print("   the autotuner skips configs that fail to compile for lack of shared memory.")

# Try this:
# 1. In section 2, add W = 264 (two programs per H100 SM). Which g wins now, and is it near sqrt(264) ~ 16?
# 2. In section 3, set M = 512 (a smaller prefill): gm = 4 < 8, so the last group is short -- how much does g=8 still save?
# 3. Set M = 128 (decode-sized batch): gm = 1, and every order gives the same footprint. Why?
