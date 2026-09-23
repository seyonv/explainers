"""Mooncake's store-vs-recompute inequality (FAST'25, section 2.2), as a calculator.

Prefill of n tokens costs (Eq 1)   flops(n) = l * (a*n^2*d + b*n*d^2).
Reusing a cached prefix of p tokens saves flops(p) but must first move
p * l * (2*d/gqa) * s bytes of KV into the GPU. Reuse wins on TTFT when
the load time is shorter than the compute time it replaces (Eq 2):

    p*l*(2d/gqa)*s / B  <  l*(a*p^2*d + b*p*d^2) / G
    <=>  B/G  >  2*d*s / (gqa*(a*p*d + b*d^2))
    <=>  B    >  B_min = 2*s*G / (gqa*(a*p + b*d))      (d cancels once)

Part 1 reproduces the paper's two numbers (LLaMA3-70B, prefix 8192):
6 GB/s on 8xA800 and 19 GB/s on 8xH800. Part 2 applies the same inequality
to the course's running example (Llama-3.1-8B on one H100). Part 3 is the
paper's capacity arithmetic (section 5.3.1) and our per-tier token counts.

Run: python3 labs/store-vs-recompute.py   (stdlib only, < 1 s)
"""


def b_min(l, d, a, b, gqa, s, G, p):
    """Minimum KV load bandwidth (bytes/s) for reuse of a p-token prefix to beat recompute."""
    return 2 * s * G / (gqa * (a * p + b * d))


def flops(l, d, a, b, n):
    """Eq 1: prefill FLOPs for n tokens."""
    return l * (a * n * n * d + b * n * d * d)


def kv_bytes(l, d, gqa, s, p):
    """KV bytes for p tokens: p * l * (2d/gqa) * s."""
    return p * l * (2 * d / gqa) * s


def part1():
    print("== Part 1: the paper's example (FAST'25 Table 1: LLaMA3-70B) ==")
    m = dict(l=80, d=8192, a=4, b=22, gqa=8, s=2)
    p = 8192
    for name, G, note in [("8xA800", 8 * 312e12, "paper: 6 GB/s"),
                          ("8xH800", 8 * 989e12, "paper: 19 GB/s (G = 8 x 989 TF is our assumption)")]:
        B = b_min(G=G, p=p, **m)
        t_re = flops(m["l"], m["d"], m["a"], m["b"], p) / G
        kv = kv_bytes(m["l"], m["d"], m["gqa"], m["s"], p)
        print(f"{name}: G = {G/1e15:.3f} PFLOP/s, B_min at p=8192 = {B/1e9:.2f} GB/s   [{note}]")
        print(f"   recompute 8192 tokens: {flops(m['l'], m['d'], m['a'], m['b'], p)/1e15:.3f} PFLOP"
              f" -> {t_re*1e3:.0f} ms at 100% of G;  KV to load: {kv/1e9:.2f} GB"
              f" -> {kv/B*1e3:.0f} ms at B_min")
    print(f"KV per token, LLaMA3-70B: {kv_bytes(80, 8192, 8, 2, 1):,.0f} B = "
          f"{kv_bytes(80, 8192, 8, 2, 1)/1024:.0f} KiB (paper: 320 KB)")
    print(f"Bytes of bandwidth needed per FLOP at p=8192: {b_min(G=1, p=p, **m):.3e} B/FLOP")
    print()


def part2(prefixes=(1024, 8192, 32768)):
    print("== Part 2: our recomputation for Llama-3.1-8B BF16 on one H100 SXM ==")
    # a = 4 as in the paper (QK^T and PV, 2 FLOPs per MAC, no causal halving).
    # b = 2 * (weights per layer) / d^2 = 2 * 13 = 26 by the same convention:
    # Q,O 2d^2 + K,V 2*d*(d/4) + MLP 3*d*14336 = 13 d^2 per layer.
    d, F = 4096, 14336
    per_layer = 2 * d * d + 2 * d * (d // 4) + 3 * d * F
    b = 2 * per_layer / d**2
    m = dict(l=32, d=d, a=4, b=b, gqa=4, s=2)
    G = 989e12
    print(f"matmul weights per layer = {per_layer/1e6:.2f}M = {per_layer/d**2:.2f} d^2 (norms excluded)  -> b = {b:.1f}")
    print(f"KV per token: {kv_bytes(32, d, 4, 2, 1):,.0f} B (course: 131,072 B = 128 KiB)")
    links = [("PCIe Gen5 x16, one direction (NVIDIA: 128 GB/s total)", 64e9),
             ("400 Gbps NIC", 400e9 / 8),
             ("100 Gbps NIC", 100e9 / 8)]
    print(f"{'prefix':>7} {'B_min GB/s':>11} {'KV GB':>7} {'recompute ms':>13} "
          f"{'load @64 ms':>12} {'load @50 ms':>12}")
    for p in prefixes:
        B = b_min(G=G, p=p, **m)
        kv = kv_bytes(32, d, 4, 2, p)
        t_re = flops(32, d, 4, b, p) / G
        print(f"{p:>7} {B/1e9:>11.2f} {kv/1e9:>7.3f} {t_re*1e3:>13.1f} "
              f"{kv/64e9*1e3:>12.1f} {kv/50e9*1e3:>12.1f}")
    for name, bw in links:
        worst = b_min(G=G, p=prefixes[0], **m)
        print(f"  {name}: {bw/1e9:.1f} GB/s = {bw/worst:.1f}x the 1k requirement")
    print(f"Bytes of bandwidth needed per FLOP at p=8192: {b_min(G=1, p=8192, **m):.3e} B/FLOP")
    print(f"Upper limit as p -> 0 (the a*p term vanishes): {b_min(G=G, p=0, **m)/1e9:.2f} GB/s")
    print()


def part3():
    print("== Part 3: capacity (FAST'25 section 5.3.1, plus our per-tier counts) ==")
    kv70 = 320 * 1024        # LLaMA3-70B, 320 KB read as KiB (= l*2*(d/gqa)*s)
    kv8 = 128 * 1024         # Llama-3.1-8B
    TB = 1e12
    print(f"1 TB of DRAM at 320 KiB/token: {TB/kv70/1e6:.2f}M tokens (paper: 'about 3 million')")
    print(f"1 TB of DRAM at 128 KiB/token: {TB/kv8/1e6:.2f}M tokens (Llama-3.1-8B)")
    print(f"50M tokens at 320 KiB/token:  {50e6*kv70/1e12:.1f} TB  -> / 1 TB per node = "
          f"{50e6*kv70/TB:.1f} nodes (paper: 'at least 20 nodes'; it doesn't show its arithmetic)")
    print(f"50M tokens at 128 KiB/token:  {50e6*kv8/1e12:.2f} TB")
    free_hbm = 63.9e9
    print(f"H100 free HBM 63.9 GB at 128 KiB/token: {free_hbm/kv8/1e3:.0f}k tokens; "
          f"x8 GPUs: {8*free_hbm/kv8/1e6:.2f}M tokens")
    print(f"40 GB transfer (LLaMA3-70B, 128k tokens): 128k x 320 KiB = {131072*kv70/1e9:.1f} GB; "
          f"at 87 GB/s {40/87:.2f} s, at 190 GB/s {40/190:.2f} s")
    print(f"8 x 400 Gbps = {8*400/8:.0f} GB/s line rate; 190 GB/s measured = {190/400*100:.0f}%")
    print(f"4 x 200 Gbps = {4*200/8:.0f} GB/s line rate; 87 GB/s measured = {87/100*100:.0f}%")


if __name__ == "__main__":
    part1()
    part2()
    part3()

# Try this:
#   part2(prefixes=(128, 256, 512))  # shorter prefixes need slightly more, never above the p->0 limit
#   in part2, set G = 0.5 * 989e12   # at 50% MFU the requirement halves: slow GPUs favour storing
#   in part1, set gqa=1 (plain MHA)  # 8x more KV bytes per token: B_min rises 8x to ~47 GB/s
