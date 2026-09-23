"""Occupancy calculator + Little's-law calculator (card: occupancy.html).

Part 1 reproduces siboehm's kernel-3 occupancy calculation (A6000, 66%), then runs
the same kernel on a T4 and an H100, and checks Volkov's GTX480 matmul table and the
CUDA Best Practices Guide example. Part 2 reproduces Volkov's Little's-law numbers
(576 ops/SM, ~100 KB in flight) and recomputes bytes in flight for an H100.

Pure Python, runs in well under a second: python3 labs/occupancy.py
"""
import math

# ---------------------------------------------------------------- per-SM limits
# threads, warps, regs, smem (bytes), blocks: CUDA Programming Guide 5.1 "Compute
# Capabilities" tables 30-31. reserve = shared memory CUDA keeps per block.
# reg_unit: registers per warp are rounded up to this (256 on CC >= 7; siboehm's table
# and the Best Practices Guide). warp_gran: warps per block are rounded up to this
# before counting registers (siboehm's "warp allocation granularity 4").
GPUS = {
    "A6000 (CC 8.6)": dict(threads=1536, warps=48, regs=65536, smem=102400,
                           blocks=16, reserve=1024, reg_unit=256, warp_gran=4),
    "T4 (CC 7.5)":    dict(threads=1024, warps=32, regs=65536, smem=65536,
                           blocks=16, reserve=0, reg_unit=256, warp_gran=4),
    "H100 (CC 9.0)":  dict(threads=2048, warps=64, regs=65536, smem=228 * 1024,
                           blocks=32, reserve=1024, reg_unit=256, warp_gran=4),
    "V100 (CC 7.0)":  dict(threads=2048, warps=64, regs=65536, smem=96 * 1024,
                           blocks=32, reserve=0, reg_unit=256, warp_gran=4),
    # Fermi GF100. reg_unit=64 and warp_gran=2 are our assumption (older occupancy
    # calculator values, not stated by Volkov); with them the blocks/SM match his slides.
    "GTX480 (CC 2.0)": dict(threads=1536, warps=48, regs=32768, smem=48 * 1024,
                            blocks=8, reserve=0, reg_unit=64, warp_gran=2),
}


def ceil_to(x, m):
    return -(-x // m) * m


def occupancy(gpu, threads_per_block, regs_per_thread, smem_per_block, verbose=False):
    g = GPUS[gpu]
    warps_per_block = -(-threads_per_block // 32)
    lim_threads = g["threads"] // threads_per_block
    lim_warps = g["warps"] // warps_per_block
    smem_block = smem_per_block + g["reserve"]
    lim_smem = g["smem"] // smem_block if smem_block else g["blocks"]
    regs_per_warp = ceil_to(regs_per_thread * 32, g["reg_unit"])
    regs_block = regs_per_warp * ceil_to(warps_per_block, g["warp_gran"])
    lim_regs = g["regs"] // regs_block
    lims = {"threads": min(lim_threads, lim_warps), "registers": lim_regs,
            "shared memory": lim_smem, "max blocks": g["blocks"]}
    blocks = min(lims.values())
    active = blocks * warps_per_block
    occ = active / g["warps"]
    if verbose:
        print(f"  {gpu}: {threads_per_block} threads/block, {regs_per_thread} regs/thread, "
              f"{smem_per_block} B smem/block")
        print(f"    shared memory: ({smem_per_block} + {g['reserve']} reserve) = {smem_block} B/block;"
              f" {g['smem']} / {smem_block} = {g['smem'] / smem_block:.2f} -> {lim_smem} blocks")
        print(f"    threads:       {g['threads']} / {threads_per_block} = "
              f"{g['threads'] / threads_per_block:.2f} -> {lim_threads} blocks")
        print(f"    registers:     {regs_per_thread} x 32 = {regs_per_thread * 32} -> round up to "
              f"{g['reg_unit']}: {regs_per_warp}/warp x {ceil_to(warps_per_block, g['warp_gran'])} warps"
              f" = {regs_block}/block; {g['regs']} / {regs_block} = {g['regs'] / regs_block:.2f}"
              f" -> {lim_regs} blocks")
        print(f"    max blocks/SM: {g['blocks']}")
        binding = [k for k, v in lims.items() if v == blocks]
        print(f"    => {blocks} block(s) x {warps_per_block} warps = {active} / {g['warps']} warps"
              f" = {occ:.1%} theoretical occupancy (limited by {', '.join(binding)})")
    return blocks, active, occ


print("=" * 72)
print("PART 1  Occupancy = active warps / max warps, per SM")
print("=" * 72)
print("\nsiboehm kernel 3 (SMEM cache-blocking): 1024 threads, 37 regs, 8192 B smem")
for gpu in ["A6000 (CC 8.6)", "T4 (CC 7.5)", "H100 (CC 9.0)"]:
    occupancy(gpu, 1024, 37, 8192, verbose=True)
print("  (siboehm writes 32/48 as '66%'; exactly it is 66.7%)")

print("\nCUDA Best Practices Guide 11.1.1 check (CC 7.0, 37 regs/thread):")
for tpb, want in [(128, "75%, 12 blocks"), (320, "63%, 4 blocks")]:
    b, a, o = occupancy("V100 (CC 7.0)", tpb, 37, 0)
    print(f"  {tpb:4d}-thread blocks -> {b:2d} blocks, {o:.1%}   (guide: {want})")

print("\nVolkov GTC 2010 matmul on GTX480 (32x32 tile, 8 KB smem per block):")
print("  outputs/thread  threads  regs  blocks/SM  occupancy   Volkov's slide")
for outs, tpb, regs, gf, slide in [(1, 1024, 21, 242, "67%, 1 block"),
                                   (2, 512, 28, 341, "67%, 2 blocks"),
                                   (4, 256, 41, 427, "50%, 3 blocks"),
                                   (8, 128, 63, 485, "33%, 4 blocks")]:
    b, a, o = occupancy("GTX480 (CC 2.0)", tpb, regs, 8192)
    print(f"  {outs:14d}  {tpb:7d}  {regs:4d}  {b:9d}  {o:9.1%}   {slide}, {gf} Gflop/s")

print("\n" + "=" * 72)
print("PART 2  Little's law: needed parallelism = latency x throughput")
print("=" * 72)
# Volkov slide 11 / 28: GF100 arithmetic
lat, cores = 18, 32
need = lat * cores
print(f"\nGF100 arithmetic: {lat} cycles x {cores} ops/cycle/SM = {need} ops in flight per SM")
print("  Volkov's GTX480 experiment, threads/SM for 100% of peak (read off his charts):")
for ilp, thr in [(1, 576), (2, 320), (3, 256), (4, 192)]:
    print(f"    ILP={ilp}: {thr:4d} threads = {thr // 32:2d} warps = {thr / 1536:5.1%} occupancy;"
          f" ops in flight = {thr} x {ilp} = {thr * ilp}")

# Volkov slide 28: memory
clk = 1.4e9           # GTX480 shader clock (Volkov uses 1.4 GHz on slide 45)
bw = 177.4e9          # GTX480 pin bandwidth, slide 31
lat_cyc = 800         # "< 800 cycles (?)", slide 28
bpc = bw / clk
inflight = lat_cyc * bpc
print(f"\nGTX480 memory: {bw / 1e9:.1f} GB/s / {clk / 1e9:.1f} GHz = {bpc:.1f} B/cycle;"
      f" x {lat_cyc} cycles = {inflight / 1e3:.0f} KB in flight (whole GPU)")
print(f"  per SM (15 SMs): {inflight / 15 / 1e3:.2f} KB")
print(f"  at 4 B/thread: {inflight / 4:,.0f} threads needed; GPU holds 15 x 1536 = {15 * 1536:,}")
print(f"  at 100 B/thread: {inflight / 100:,.0f} threads")
for occ_pct, n, vec in [(8, 8, 16), (4, 14, 16)]:
    thr = round(occ_pct / 100 * 1536 / 32) * 32
    per_sm = thr * n * vec
    print(f"  {n} float4/thread at {occ_pct}% occupancy ~ {thr} threads/SM x {n * vec} B ="
          f" {per_sm / 1e3:.1f} KB/SM x 15 = {per_sm * 15 / 1e3:.0f} KB in flight")

# H100: our recomputation with Luo et al. 2024 (H800 PCIe global latency, Table IV)
h_lat_cyc, h_clk = 478.8, 1.755e9
h_lat = h_lat_cyc / h_clk
h_bw, sms = 3.35e12, 132
h_inflight = h_lat * h_bw
per_sm = h_inflight / sms
print(f"\nH100 (our recomputation): latency {h_lat_cyc} cycles / {h_clk / 1e9:.3f} GHz ="
      f" {h_lat * 1e9:.0f} ns")
print(f"  bytes in flight = {h_lat * 1e9:.0f} ns x {h_bw / 1e12:.2f} TB/s ="
      f" {h_inflight / 1e3:.0f} KB (whole GPU); / {sms} SMs = {per_sm / 1e3:.2f} KB per SM")
for label, b in [("4 B (one float)", 4), ("16 B (one float4)", 16), ("128 B (8 x float4)", 128)]:
    thr = per_sm / b
    print(f"  {label:20s}: {thr:6.0f} threads/SM = {thr / 32:5.1f} warps = {thr / 2048:5.1%} occupancy")

# Try this:
# 1. occupancy("H100 (CC 9.0)", 256, 64, 48 * 1024, verbose=True)  # a typical big-tile kernel
# 2. Change 37 regs to 32 in the kernel-3 call: does the A6000 fit a second block? (no: threads)
# 3. Double h_lat_cyc (loaded memory latency is higher than idle): how many warps/SM now?
