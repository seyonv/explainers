"""Memory coalescing: count the 32-byte sectors one warp touches.

A warp is 32 threads that issue one load (or store) together. The memory
system serves it in aligned 32-byte sectors (CUDA Best Practices Guide 10.2.1:
"a number of transactions equal to the number of 32-byte transactions
necessary to service all of the threads of the warp"). This script lays out the
32 addresses for an access pattern, counts the distinct 32 B sectors and
128 B cache lines they fall in, and reports the useful-byte efficiency:

    efficiency = bytes the threads asked for / (32 B x sectors moved)

Lane k reads element  offset + k * stride  of an array whose start is
256-byte aligned (cudaMalloc guarantees at least 256 B).

Runs on any CPU in well under a second:  python3 labs/coalescing.py
"""

WARP = 32
SECTOR = 32
LINE = 128


def warp_access(offset=0, stride=1, elem=4, warp=WARP):
    """Return (sectors, lines, useful_bytes, efficiency) for one warp request."""
    addrs = [(offset + k * stride) * elem for k in range(warp)]
    sectors, lines = set(), set()
    for a in addrs:
        for b in range(a, a + elem):          # every byte this lane touches
            sectors.add(b // SECTOR)
            lines.add(b // LINE)
    useful = len(set(addrs)) * elem
    return len(sectors), len(lines), useful, useful / (len(sectors) * SECTOR)


def row(label, **kw):
    s, l, u, e = warp_access(**kw)
    print(f"{label:<34}{s:>8}{l:>7}{u:>8}{s*SECTOR:>9}{e:>9.1%}")
    return s, e


HDR = f"{'pattern':<34}{'sectors':>8}{'lines':>7}{'useful':>8}{'moved':>9}{'eff':>9}"

print("One warp, 32 lanes, 4-byte floats. Bytes: useful = asked for, moved = 32 B x sectors\n")

print("1) Offset sweep, stride 1:  lane k reads a[offset + k]")
print(HDR)
for off in range(0, 33):
    row(f"offset {off:>2}", offset=off)

print("\n2) Stride sweep, offset 0:  lane k reads a[k * stride]")
print(HDR)
for st in (1, 2, 4, 8, 16, 32):
    row(f"stride {st:>2}", stride=st)

print("\n3) The worked-example patterns")
print(HDR)
key = {}
key["stride1"] = row("stride 1 (coalesced)", stride=1)
key["off1"] = row("offset 1 (misaligned)", offset=1)
key["stride2"] = row("stride 2", stride=2)
key["stride32"] = row("stride 32", stride=32)

print("\n4) Matrix transpose, 1024 x 1024 floats, block 32 x 8 (Harris transpose blog)")
print("   A warp is one row of threadIdx.x = 0..31, so lanes differ only in x.")
print(HDR)
W = 1024
rd, _ = row("copy / naive read: idata[y*W + x]", stride=1)
wr_copy, _ = row("copy write: odata[y*W + x]", stride=1)
wr_naive, _ = row(f"naive write: odata[x*W + y]", stride=W)
print(f"   naive write stride = {W} floats = {W*4} B between neighbouring lanes")
copy_bytes = (rd + wr_copy) * SECTOR
naive_bytes = (rd + wr_naive) * SECTOR
print(f"   per warp step: copy moves {rd}+{wr_copy} = {rd+wr_copy} sectors = {copy_bytes} B;"
      f" naive moves {rd}+{wr_naive} = {rd+wr_naive} sectors = {naive_bytes} B")
print(f"   naive / copy DRAM traffic if no cache merges partial writes = {naive_bytes/copy_bytes:.2f}x")
print(f"   source (Harris, ECC on): M2050 copy/naive = 105.2/18.8 = {105.2/18.8:.2f}x;"
      f" K20c 136.0/55.3 = {136.0/55.3:.2f}x")

print("\n5) Element size matters (offset 0)")
print(HDR)
row("bf16 (2 B), stride 1", elem=2)
row("bf16 (2 B), stride 16", elem=2, stride=16)
row("float4 (16 B), stride 1", elem=16)

print("\n6) What 12.5% sector efficiency would cost (illustrative)")
eff = key["stride32"][1]
H100_BW, WEIGHTS = 3.35e12, 16.06e9
base_ms = WEIGHTS / H100_BW * 1e3
bad_ms = WEIGHTS / eff / H100_BW * 1e3
print(f"   H100 decode step, Llama-3.1-8B BF16: {WEIGHTS/1e9:.2f} GB / 3.35 TB/s = {base_ms:.2f} ms"
      f"  ({1e3/base_ms:.0f} tokens/s)")
print(f"   at {eff:.1%} efficiency: {WEIGHTS/eff/1e9:.1f} GB moved -> {bad_ms:.1f} ms"
      f"  ({1e3/bad_ms:.0f} tokens/s)")
T4_BW = 320e9
print(f"   T4: useful bandwidth ceiling at {eff:.1%} = 320 x {eff:.3f} = {T4_BW*eff/1e9:.0f} GB/s")

# Try this:
# 1) row("offset 8", offset=8) vs offset 1: why are multiples of 8 floats (32 B) free?
# 2) warp_access(stride=3) and stride=5: odd strides spread over more sectors than you'd guess.
# 3) warp_access(offset=1, warp=16): the pre-Fermi half-warp view (C870 coalesced per 16 threads).
