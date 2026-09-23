"""The CUDA execution model, as arithmetic.

Given a 1D launch (N elements, one per thread, B threads per block) and a GPU's
limits, print: blocks, warps, how many blocks can be resident at once, how many
"waves" that makes, which block / warp / lane handles a few sample elements,
and how many warp passes an if/else costs for a few per-lane conditions.

Occupancy-free: this ignores registers and shared memory, which can only lower
the resident-block count (see the occupancy card). Stdlib only.

Run: python3 labs/execution-model.py
"""
import math

N = 1 << 20          # 1,048,576 floats (the size in Harris's "Even Easier Intro")
BLOCK = 256          # threads per block
WARP = 32

# (SMs, max resident threads/SM, max resident blocks/SM)
# CUDA Programming Guide 5.1, Compute Capabilities table: CC 7.5 = 1024 / 16, CC 9.0 = 2048 / 32
GPUS = {
    "T4 (CC 7.5)":     (40, 1024, 16),
    "H100 SXM (CC 9.0)": (132, 2048, 32),
}

SAMPLES = [(0, 0), (3, 37), (1023, 255), (4095, 255)]  # (blockIdx.x, threadIdx.x)


def launch(n, block):
    blocks = math.ceil(n / block)
    warps_per_block = math.ceil(block / WARP)
    return blocks, warps_per_block, blocks * warps_per_block


def residency(block, sms, max_threads, max_blocks):
    per_sm = min(max_threads // block, max_blocks)
    in_flight = per_sm * sms
    return per_sm, in_flight


def where(bx, tx, block, blocks_in_flight):
    i = bx * block + tx
    warp_in_block = tx // WARP
    lane = tx % WARP
    global_warp = bx * math.ceil(block / WARP) + warp_in_block
    wave = bx // blocks_in_flight  # idealised: blocks handed out in index order
    return i, warp_in_block, lane, global_warp, wave


def passes(cond):
    """cond: 32 booleans, one per lane. Each branch with >= 1 active lane costs one pass."""
    taken = sum(cond)
    p = (taken > 0) + (taken < len(cond))
    used = len(cond)  # every lane does useful work in exactly one pass
    return p, taken, used / (p * WARP)


def main():
    blocks, wpb, warps = launch(N, BLOCK)
    print(f"Launch: N = {N:,} elements, {BLOCK} threads/block")
    print(f"  blocks  = ceil({N:,} / {BLOCK}) = {blocks:,}")
    print(f"  warps   = {blocks:,} blocks x {wpb} warps/block = {warps:,}")
    print(f"  threads = {blocks * BLOCK:,}")
    # Harris's add kernel: y[i] = x[i] + y[i] -> read x, read y, write y = 12 B per element
    nbytes = 3 * 4 * N
    print(f"  bytes moved = 3 x 4 B x {N:,} = {nbytes:,} B")
    print(f"  T4 floor at 320 GB/s = {nbytes / 320e9 * 1e6:.1f} us;  Harris measured 47,520 ns -> "
          f"{nbytes / 47_520e-9 / 1e9:.1f} GB/s (T4, Even Easier Intro)\n")

    for name, (sms, mt, mb) in GPUS.items():
        per_sm, fl = residency(BLOCK, sms, mt, mb)
        waves = blocks / fl
        print(f"{name}: {sms} SMs, {mt} threads/SM ({mt // WARP} warps), max {mb} blocks/SM")
        print(f"  resident blocks/SM = min({mt} // {BLOCK}, {mb}) = {per_sm}")
        print(f"  blocks in flight   = {per_sm} x {sms} = {fl:,}  ({fl * BLOCK:,} threads)")
        print(f"  waves              = {blocks:,} / {fl:,} = {waves:.2f}  -> {math.ceil(waves)} rounds, "
              f"last one {blocks - (math.ceil(waves) - 1) * fl} blocks ({(blocks - (math.ceil(waves) - 1) * fl) / fl:.0%} full)")
        print()

    fl_t4 = residency(BLOCK, *GPUS["T4 (CC 7.5)"])[1]
    fl_h = residency(BLOCK, *GPUS["H100 SXM (CC 9.0)"])[1]
    print("Index mapping  i = blockIdx.x * blockDim.x + threadIdx.x")
    print(f"  {'block':>6} {'thread':>6} {'i':>9} {'warp in blk':>11} {'lane':>4} {'global warp':>11} {'T4 wave':>7} {'H100 wave':>9}")
    for bx, tx in SAMPLES:
        i, w, lane, gw, wv = where(bx, tx, BLOCK, fl_t4)
        wv_h = bx // fl_h
        print(f"  {bx:>6} {tx:>6} {i:>9,} {w:>11} {lane:>4} {gw:>11,} {wv:>7} {wv_h:>9}")
    print("  (wave = block // blocks-in-flight: an idealised in-order hand-out; CUDA guarantees no order or SM)\n")

    print("Divergence: if (cond) {A} else {B}, one warp, A and B one pass each (illustrative cost model)")
    lanes = range(WARP)
    patterns = {
        "all true":            [True] * WARP,
        "alternating lanes":   [l % 2 == 0 for l in lanes],
        "first 16 lanes":      [l < 16 for l in lanes],
        "warp-aligned (tid<128), warp 1": [32 + l < 128 for l in lanes],
    }
    for name, cond in patterns.items():
        p, taken, util = passes(cond)
        print(f"  {name:<32} taken {taken:>2}/32  passes {p}  lane utilisation {util:.0%}")

    # Block-level contrast: same half/half split, lane-wise vs warp-wise
    for name, f in [("tid % 2 == 0", lambda t: t % 2 == 0), ("(tid / 32) % 2 == 0", lambda t: (t // WARP) % 2 == 0)]:
        total = sum(passes([f(w * WARP + l) for l in lanes])[0] for w in range(wpb))
        print(f"  block of {BLOCK}, cond {name:<20} -> {total} warp passes for {wpb} warps")


if __name__ == "__main__":
    main()

# Try this:
# BLOCK = 1024  -> T4 fits 1 block/SM (40 in flight, 25.6 waves again); H100 fits 2 (264 in flight).
# BLOCK = 96    -> 3 warps/block; T4 hits min(1024//96=10, 16) = 10 blocks/SM: 960 of 1,024 thread slots used.
# BLOCK = 32    -> T4's 16-blocks/SM cap binds: only 512 of 1,024 thread slots can be filled.
