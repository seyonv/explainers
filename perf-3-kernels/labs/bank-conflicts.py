"""Shared-memory bank-conflict counter and the transpose blog's table.

A simulator, not a benchmark: it runs on the CPU and needs no GPU.
Model (Harris, "Using Shared Memory in CUDA C/C++"): shared memory has 32 banks,
successive 32-bit words live in successive banks, so

    bank = (byte_address // 4) % 32

A warp's request is served in "wavefronts". Threads that hit the same word share
it (broadcast); different words in the same bank are serialized. So the number
of wavefronts is the largest count of *distinct words* any one bank must deliver.
The unavoidable minimum is ceil(bytes requested / 128), since 32 banks x 4 B = 128 B
per wavefront. "Conflict degree" = wavefronts / minimum.

Run: python3 labs/bank-conflicts.py
"""
import math

import numpy as np

BANKS = 32
WORD = 4  # bytes per bank word


def wavefronts(byte_addrs, elem_bytes):
    """Wavefronts needed for one warp request of elements starting at byte_addrs."""
    words = set()
    for a in byte_addrs:
        for w in range(elem_bytes // WORD):  # an 8-byte double covers 2 words
            words.add(a // WORD + w)
    per_bank = np.zeros(BANKS, dtype=int)
    for w in words:
        per_bank[w % BANKS] += 1
    need = int(per_bank.max())
    minimum = math.ceil(len(byte_addrs) * elem_bytes / (BANKS * WORD))
    return need, minimum, per_bank


def warp_access(width, elem_bytes, pattern, fixed=0):
    """Byte addresses of 32 threads reading tile[rows][width].

    row:    thread t reads tile[fixed][t]   (what the load phase does)
    column: thread t reads tile[t][fixed]   (what transposeCoalesced's store phase does)
    """
    t = np.arange(32)
    if pattern == "row":
        idx = fixed * width + t
    else:
        idx = t * width + fixed
    return [int(i) * elem_bytes for i in idx]


def report(label, width, elem_bytes):
    print(f"\n{label}: tile[32][{width}] of {elem_bytes}-byte elements "
          f"({32 * width * elem_bytes:,} B)")
    for pattern in ("row", "column"):
        worst = (0, 1)
        for fixed in range(32):  # every row / every column; report the worst
            need, minimum, _ = wavefronts(warp_access(width, elem_bytes, pattern, fixed), elem_bytes)
            if need / minimum > worst[0] / worst[1]:
                worst = (need, minimum)
        need, minimum = worst
        tag = "conflict-free" if need == minimum else f"{need // minimum}-way conflict"
        print(f"  {pattern:6s} read by one warp: {need:2d} wavefronts, minimum {minimum}"
              f" -> {tag}")


print("=" * 72)
print("1. Bank index of tile[i][j]: word = i*width + j, bank = word % 32")
print("=" * 72)
for width in (32, 33):
    banks = [(i * width + 0) % BANKS for i in range(32)]
    print(f"  tile[32][{width}], column j=0, rows i=0..31 -> banks {banks[:8]} ... {banks[-2:]}"
          f"  ({len(set(banks))} distinct)")
    _, _, per_bank = wavefronts(warp_access(width, 4, "column", 0), 4)
    print(f"    requests per bank: max {per_bank.max()}, banks used {int((per_bank > 0).sum())}")

print("\n" + "=" * 72)
print("2. Worst-case conflict degree per warp (32 threads)")
print("=" * 72)
report("float, the blog's transposeCoalesced", 32, 4)
report("float, the blog's fix (pad by 1)", 33, 4)
report("float, a 64-wide tile", 64, 4)
report("float, 64-wide padded by 1", 65, 4)
report("double, 32-wide", 32, 8)
report("double, 32-wide padded by 1 double", 33, 8)

print("\nXOR swizzle instead of padding: tile[32][32] float, element (i,j) stored at column j ^ i")
t = np.arange(32)
worst = max(wavefronts([int(i * 32 + (j ^ i)) * 4 for i in t], 4)[0] for j in range(32))
print(f"  column read tile[t][j ^ t] by one warp: {worst} wavefront(s), no extra bytes (4,096 B)")

print("\n" + "=" * 72)
print("3. Harris transpose table (1024x1024 float, GB/s, ECC on) and our ratios")
print("=" * 72)
table = {  # kernel: (Tesla M2050, Tesla K20c), from the blog
    "copy": (105.2, 136.0),
    "copySharedMem": (104.6, 152.3),
    "transposeNaive": (18.8, 55.3),
    "transposeCoalesced": (51.3, 97.6),
    "transposeNoBankConflicts": (99.5, 144.3),
}
nbytes = 2 * 1024 * 1024 * 4  # read + write, 4 B per float
print(f"  bytes moved per kernel = 2 x 1024 x 1024 x 4 = {nbytes:,} B")
print(f"  {'kernel':26s}{'M2050':>8s}{'time us':>9s}{'K20c':>8s}{'time us':>9s}")
for k, (m, kk) in table.items():
    print(f"  {k:26s}{m:8.1f}{nbytes / (m * 1e9) * 1e6:9.1f}{kk:8.1f}{nbytes / (kk * 1e9) * 1e6:9.1f}")


def ratio(a, b):
    return tuple(table[a][g] / table[b][g] for g in range(2))


steps = [
    ("naive -> coalesced (shared memory)", "transposeCoalesced", "transposeNaive"),
    ("coalesced -> padded (no conflicts)", "transposeNoBankConflicts", "transposeCoalesced"),
    ("naive -> padded (total)", "transposeNoBankConflicts", "transposeNaive"),
]
print("\n  step multipliers            M2050    K20c")
for label, a, b in steps:
    m, k = ratio(a, b)
    print(f"  {label:36s}{m:5.2f}x  {k:5.2f}x")
print("\n  as a share of a copy         M2050    K20c")
for label, a, b in [
    ("naive / copy", "transposeNaive", "copy"),
    ("padded / copy", "transposeNoBankConflicts", "copy"),
    ("padded / copySharedMem", "transposeNoBankConflicts", "copySharedMem"),
    ("copySharedMem / copy", "copySharedMem", "copy"),
]:
    m, k = ratio(a, b)
    print(f"  {label:28s}{m * 100:6.1f}%  {k * 100:6.1f}%")
fastest = [max(table["copy"][g], table["copySharedMem"][g]) for g in range(2)]
print("  padded / fastest copy       "
      + "  ".join(f"{table['transposeNoBankConflicts'][g] / fastest[g] * 100:6.1f}%" for g in range(2))
      + "   <- the blog's 'about 95%'")

# Try this:
# 1. report("float, 48-wide", 48, 4): 48 = 16 x 3, so how many threads share a bank?
# 2. Change a column read to tile[t][t] (the diagonal) in warp_access; does width 32 conflict?
# 3. Break the swizzle: use (j ^ (i // 2)) instead of (j ^ i) and see the conflict come back.
