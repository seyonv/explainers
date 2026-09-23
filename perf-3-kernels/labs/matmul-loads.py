"""Loads per result for siboehm's SGEMM kernels 1-5 (a calculator, no GPU needed).

Reproduces the memory-access counts in "How to Optimize a CUDA Matmul Kernel for
cuBLAS-like Performance: a Worklog" (siboehm, 2022) for C = A @ B with
M = N = K = 4092, fp32, and extends them to FLOP/byte and total bytes.

Model: a thread block owns a BM x BN tile of C and walks K in steps of BK.
Each step it copies a BM x BK slice of A and a BK x BN slice of B from global
memory (GMEM) into shared memory (SMEM). Each thread owns TM x TN results and,
per dotIdx, reads TM values of As and TN values of Bs into registers.

  GMEM loads per result = K * (BM + BN) / (BM * BN)   (BK, TM, TN cancel out)
  SMEM loads per result = K * (TM + TN) / (TM * TN)

These are loads *issued* by the kernel. Real DRAM traffic is lower whenever L1
or L2 catches a repeat; this calculator assumes no cache hits at all.

Run: python3 labs/matmul-loads.py
"""
from fractions import Fraction as F

K = M = N = 4092
BYTES = 4  # fp32
FLOP_PER_RESULT = 2 * K  # K fused multiply-adds, 2 FLOPs each
MIN_BYTES = 3 * M * N * BYTES + M * N * BYTES  # read A, B, C once + write C once


def threads(BM, BN, TM, TN):
    return (BM * BN) // (TM * TN)


def gmem_per_result(BM, BN):
    return F(K * (BM + BN), BM * BN)


def smem_per_result(TM, TN, tiled=True):
    return F(K * (TM + TN), TM * TN) if tiled else F(0)


def gmem_per_result_loopwise(BM, BN, BK, TM, TN):
    """The article's way: outer-loop iterations x loads per thread per iteration, / results per thread."""
    iters = F(K, BK)
    loads_per_thread_per_iter = F(BM * BK + BK * BN, threads(BM, BN, TM, TN))
    return iters * loads_per_thread_per_iter / (TM * TN)


def as_k_fraction(x):
    f = x / K
    return "K" if f == 1 else f"{f.numerator}K" if f.denominator == 1 else (
        f"K/{f.denominator}" if f.numerator == 1 else f"{f.numerator}K/{f.denominator}")


# name: (BM, BN, BK, TM, TN, uses SMEM)
KERNELS = {
    "1 naive":            (1, 1, 1, 1, 1, False),   # no sharing: every thread loads its own row and column
    "2 coalesced":        (1, 1, 1, 1, 1, False),   # same loads as 1; only the warp's address pattern changes
    "3 SMEM 32x32":       (32, 32, 32, 1, 1, True),
    "4 1D blocktile":     (64, 64, 8, 8, 1, True),
    "5 2D blocktile":     (128, 128, 8, 8, 8, True),
}

print(f"C = A @ B, M = N = K = {K}, fp32. Results = {M*N:,}. FLOPs per result = 2K = {FLOP_PER_RESULT:,}")
print(f"Minimum GMEM traffic (read A, B, C once, write C once): {MIN_BYTES/1e6:.0f} MB\n")
print(f"{'kernel':<16}{'threads':>8}{'GMEM/result':>13}{'= floats':>10}{'SMEM/result':>13}"
      f"{'FLOP/B (GMEM)':>15}{'total GMEM':>12}{'x minimum':>11}")
for name, (BM, BN, BK, TM, TN, smem) in KERNELS.items():
    g = gmem_per_result(BM, BN)
    if smem:
        assert g == gmem_per_result_loopwise(BM, BN, BK, TM, TN)
    s = smem_per_result(TM, TN, smem)
    ai = F(FLOP_PER_RESULT) / (g * BYTES)
    total = g * BYTES * M * N
    t = threads(BM, BN, TM, TN) if smem else 1024
    print(f"{name:<16}{t:>8}{as_k_fraction(g):>13}{float(g):>10.1f}{(as_k_fraction(s) if s else '-'):>13}"
          f"{float(ai):>15.2f}{float(total)/1e9:>10.2f} GB{float(total)/MIN_BYTES:>10.0f}x")

print("\nFLOP/B counts A and B loads only. Adding C's read + write (8 B per result):")
for name, (BM, BN, BK, TM, TN, smem) in KERNELS.items():
    b = gmem_per_result(BM, BN) * BYTES + 8
    print(f"  {name:<16}{FLOP_PER_RESULT / float(b):7.2f} FLOP/B")

# Kernel 1 vs 2: 32-byte sectors one warp touches in one step of the k-loop (no cache).
print("\nOne warp, one k-step: distinct 32 B sectors touched (kernel 1 vs 2)")
SECTOR = 32


def sectors(addrs):
    return len({a // SECTOR for a in addrs})


i = 0
for name, rows, cols in [("1 naive", range(32), [0] * 32),        # threadIdx.x -> row x; one column y
                         ("2 coalesced", [0] * 32, range(32))]:   # one row x; threadIdx.x % 32 -> column y
    a = [(r * K + i) * BYTES for r in rows]
    b = [(i * N + c) * BYTES for c in cols]
    sa, sb = sectors(a), sectors(b)
    useful = len(set(a)) * BYTES + len(set(b)) * BYTES
    print(f"  {name:<12} A: {sa:>2} sectors  B: {sb:>2} sectors  total {sa+sb:>2} = {(sa+sb)*SECTOR:>4} B moved"
          f" for {useful} distinct useful bytes")

# Kernel 3's shared memory and block sizes
print("\nKernel 3 SMEM per block: 2 x 32 x 32 x 4 B =", 2 * 32 * 32 * 4, "B")
print("Kernel 3 per thread per 32-wide tile: 2 GMEM loads, 64 SMEM loads, 32 FMAs -> 2 SMEM loads per FMA")

# The article's measured numbers (RTX A6000) and the ratios the card quotes
print("\nArticle (A6000): GFLOP/s 309.0 -> 1986.5 -> 2980.3;  GMEM throughput 15 -> 110 GB/s")
print(f"  kernel 2 / kernel 1 = {1986.5/309.0:.2f}x   kernel 3 / kernel 2 = {2980.3/1986.5:.2f}x"
      f"   (text's ~2200 / 1986.5 = {2200/1986.5:.2f}x)")
print(f"  GMEM throughput 110 / 15 = {110/15:.1f}x   Volta shared vs global 12080 / 750 = {12080/750:.1f}x")
print(f"  kernel 2 at 1986.5 GFLOP/s and 0.25 FLOP/B issues {1986.5e9/0.25/1e12:.2f} TB/s of thread-level loads;"
      f" the profiler saw 110 GB/s -> caches/broadcast serve ~{1 - 110e9/(1986.5e9/0.25):.1%} of it")
print(f"  kernel 1 no-cache ceiling on the A6000: 0.25 x 768 GB/s = {0.25*768:.0f} GFLOP/s < measured 309.0")

# Kernel 3 occupancy on the A6000 (the article's calculation; occupancy.py has the full calculator)
regs_warp = -(-37 * 32 // 256) * 256
print(f"\nKernel 3 occupancy (A6000): smem {102400 // (8192 + 1024)} blocks, threads {1536 // 1024} block,"
      f" regs 37*32={37*32} -> {regs_warp}/warp x 32 warps = {regs_warp*32} of 65536 -> {65536 // (regs_warp*32)} block"
      f"  => 32/48 warps = {32/48:.1%}")

# Upper bound from GMEM-level intensity if every load went to DRAM: min(peak, AI x bandwidth)
print("\nCeiling if every issued load reached DRAM: min(peak, FLOP/B x bandwidth), TFLOP/s")
GPUS = [("RTX A6000 fp32", 30e12, 768e9), ("T4 fp32", 8.1e12, 320e9), ("H100 SXM bf16 tensor", 989e12, 3.35e12)]
for g, peak, bw in GPUS:
    row = [min(peak, float(F(FLOP_PER_RESULT) / (gmem_per_result(BM, BN) * BYTES)) * bw) / 1e12
           for (BM, BN, BK, TM, TN, s) in KERNELS.values()]
    print(f"  {g:<22} ridge {peak/bw:6.1f} FLOP/B   k1 {row[0]:6.2f}  k3 {row[2]:6.2f}  k4 {row[3]:6.2f}  k5 {row[4]:6.2f}")

k3_h100 = min(989e12, 8 * 3.35e12)
print(f"  Llama-3.1-8B prefill, 1,000 tokens (2 x 8.03e9 x 1000 FLOPs) at kernel 3's H100 ceiling:"
      f" {2*8.03e9*1000/k3_h100*1e3:.0f} ms vs {2*8.03e9*1000/989e12*1e3:.1f} ms at 989 TFLOP/s"
      f" ({k3_h100/989e12:.1%} of peak)")

# Try this:
# 1. Set kernel 3's tile to 64x64 (BM=BN=BK=64): GMEM/result halves to K/32, SMEM/result stays 2K.
# 2. Kernel 5 with TM=TN=16: SMEM/result drops to K/8 -- but that's 256 accumulators per thread (registers!).
# 3. Change K to 1024 and watch every count scale linearly while FLOP/B stays the same.
