"""Long-context lab: sliding windows (Mistral 7B), Ring Attention, Native Sparse Attention (NSA).

Run: python3 labs/long-context.py   (stdlib only, CPU, instant)

Part 1  Llama-3.1-8B-shaped attention on one H100 SXM at 8K / 32K / 128K context, three ways:
          full attention            stores and reads every token's KV (128 KiB/token)
          4,096 sliding window      every layer keeps only the last 4,096 tokens (hypothetical for Llama)
          NSA-like token budget     reads NSA's loaded-token count per step, s/16 + 1,536 (hypothetical
                                    for Llama); it still STORES every token, because the selection branch
                                    can pick any old block
        KV bytes stored per sequence, how many sequences fit in the 63.94 GB left after BF16 weights,
        and the time to read one sequence's KV at 3.35 TB/s in a decode step.
        Plus: prefill FLOPs for one long prompt (params + attention) at 100% MFU. Our recomputation.

Part 2  Ring Attention overlap condition 4*d*c^2/F >= 4*c*d/B  ->  c >= F/B, and s = 6c.
        Reproduces the paper's table from its own FLOPS and bandwidth columns, then adds H100 (ours).

Part 3  NSA decode memory access: tokens loaded = floor((s-l)/d) + n*l' + w with l=32, d=16, n=16,
        l'=64, w=512. Reproduces the paper's 2048/2560/3584/5632 and 4x/6.4x/9.1x/11.6x.
"""
import math

# ---------------- running example (course _facts.md) ----------------
L, K, H, NQ = 32, 8, 128, 32       # layers, KV heads, head dim, query heads
P = 8.03e9                         # params
W_BYTES = 16.06e9                  # BF16 weights
FREE = 80e9 - W_BYTES              # 63.94 GB
BW = 3.35e12                       # HBM B/s
C = 989e12                         # dense BF16 FLOP/s
KV_TOK = 2 * L * K * H * 2         # 131,072 B = 128 KiB
WINDOW = 4096
CTXS = [8192, 32768, 131072]

# NSA config (arXiv 2502.11089, Sec 4 / 5)
NSA_L, NSA_D, NSA_N, NSA_LP, NSA_W = 32, 16, 16, 64, 512


def nsa_exact(s):
    return (s - NSA_L) // NSA_D + NSA_N * NSA_LP + NSA_W


def nsa_budget(s):
    # the paper's table equals s/16 + 1,536 exactly (their rounding of the formula above)
    return s // NSA_D + NSA_N * NSA_LP + NSA_W


def k(n):
    return f"{n // 1024}K"


def part1():
    print("=== Part 1: Llama-3.1-8B-shaped attention on one H100 (our recomputation) ===")
    print(f"KV per token {KV_TOK:,} B = {KV_TOK // 1024} KiB; free HBM {FREE / 1e9:.2f} GB; HBM {BW / 1e12} TB/s\n")
    print(f"{'context':<9}{'scheme':<22}{'KV stored':>11}{'seqs fit':>10}{'tokens read':>13}{'KV read':>10}{'read time':>11}")
    for s in CTXS:
        rows = [
            ("full attention", s, s),
            ("4,096 window", min(s, WINDOW), min(s, WINDOW)),
            ("NSA-like budget", s, nsa_budget(s)),
        ]
        for name, stored, read in rows:
            b_st = stored * KV_TOK
            b_rd = read * KV_TOK
            fit = math.floor(FREE / b_st)
            print(f"{k(s):<9}{name:<22}{b_st / 1e9:>8.2f} GB{fit:>10}{read:>13,}{b_rd / 1e9:>7.2f} GB"
                  f"{b_rd / BW * 1e3:>8.2f} ms")
        print()
    print("NSA 'KV stored' is a lower bound: it keeps every raw token for selection, plus compressed")
    print("tokens and separate window K/V ('independent keys and values for three branches').")
    print(f"weights read per decode step (batch 1): {W_BYTES / BW * 1e3:.2f} ms, for scale")
    full128 = 131072 * KV_TOK
    print(f"128K full: {131072:,} x {KV_TOK:,} B = {full128 / 2**30:.0f} GiB = {full128 / 1e9:.2f} GB "
          f"-> {FREE / full128:.2f} -> {math.floor(FREE / full128)} sequences")
    win = WINDOW * KV_TOK
    print(f"window:    {WINDOW:,} x {KV_TOK:,} B = {win / 2**20:.0f} MiB = {win / 1e9:.3f} GB "
          f"-> {math.floor(FREE / win)} sequences at any context length")
    print(f"window cut at 32K: {32768 // WINDOW}x (Mistral's '8x'); at 128K: {131072 // WINDOW}x")
    print(f"Mistral theoretical span: W x layers = {WINDOW} x 32 = {WINDOW * 32:,} tokens (~131K)\n")

    print("Prefill of one prompt of T tokens at 100% MFU on one H100 (a floor):")
    print("  params 2*P*T;  attention 4*T^2*H*NQ*L (QK^T and PV, no causal saving)")
    print(f"{'T':>10}{'param FLOPs':>14}{'attn FLOPs':>14}{'attn share':>12}{'time':>10}{'causal':>10}{'KV':>10}")
    for T in [8192, 32768, 131072, 1048576]:
        fp = 2 * P * T
        fa = 4 * T * T * H * NQ * L
        t = (fp + fa) / C
        tc = (fp + fa / 2) / C
        print(f"{T:>10,}{fp:>14.3e}{fa:>14.3e}{fa / (fp + fa) * 100:>11.0f}%{t:>9.1f}s{tc:>9.1f}s"
              f"{T * KV_TOK / 1e9:>7.1f} GB")
    T = 1048576
    print(f"  1M tokens: KV {T * KV_TOK / 1e9:.1f} GB > 80 GB HBM -> cannot hold on one H100; "
          f"split over 8 GPUs (ring / context parallel): {T * KV_TOK / 8 / 1e9:.1f} GB each, "
          f"time /8 = {(2 * P * T + 4 * T * T * H * NQ * L) / C / 8:.0f} s; one decode KV read {T * KV_TOK / BW * 1e3:.1f} ms")


def part2():
    print("\n=== Part 2: Ring Attention overlap, c >= F/B, s = 6c ===")
    print(f"{'host':<34}{'F (TF)':>8}{'B (GB/s)':>10}{'c = F/B':>10}{'s = 6c':>10}{'paper':>18}")
    paper = [
        ("A100 NVLink (paper)", 312, 300, "1.0K / 6.2K"),
        ("A100 InfiniBand (paper)", 312, 12.5, "24.5K / 149.5K"),
        ("TPU v3 (paper)", 123, 112, "1.1K / 6.6K"),
        ("TPU v4 (paper)", 275, 268, "1.0K / 6.2K"),
        ("TPU v5e (paper)", 196, 186, "1.1K / 6.3K"),
        ("H100 NVLink 900 GB/s total (ours)", 989, 900, "-"),
        ("H100 NVLink 450 GB/s one way (ours)", 989, 450, "-"),
    ]
    for name, F, B, pp in paper:
        c = F * 1e12 / (B * 1e9)
        print(f"{name:<34}{F:>8}{B:>10}{c / 1e3:>9.2f}K{6 * c / 1e3:>9.1f}K{pp:>18}")
    print("A100 IB check: 312e12 / 12.5e9 = 24,960 -> 25.0K and 6c = 149.8K; the paper prints 24.5K / 149.5K")
    # GQA note: K and V blocks are K*H wide, not d = NQ*H wide
    c_gqa = 989e12 / 450e9 * (K * H) / (NQ * H)
    print(f"With GQA-4 (Llama KV width 1,024 vs d = 4,096), KV blocks are 4x smaller: c >= {c_gqa:,.0f} "
          f"tokens on H100 at 450 GB/s (our inference, not the paper's formula)")


def part3():
    print("\n=== Part 3: NSA decode, tokens loaded per attention op (l=32, d=16, n=16, l'=64, w=512) ===")
    print(f"{'context':>9}{'formula':>9}{'paper':>8}{'speedup':>9}{'  paper says':>13}")
    paper = {8192: (2048, "4x"), 16384: (2560, "6.4x"), 32768: (3584, "9.1x"), 65536: (5632, "11.6x")}
    for s, (pt, ps) in paper.items():
        print(f"{s:>9,}{nsa_exact(s):>9,}{pt:>8,}{s / pt:>8.1f}x{ps:>13}")
    s = 131072
    print(f"{s:>9,}{nsa_exact(s):>9,}{'-':>8}{s / nsa_budget(s):>8.1f}x   (our extrapolation, budget {nsa_budget(s):,})")
    print("formula: floor((s-32)/16) + 16*64 + 512, e.g. 8,192 -> 510 + 1,024 + 512 = 2,046 (paper rounds to 2,048 = s/16 + 1,536)")


if __name__ == "__main__":
    part1()
    part2()
    part3()

# Try this:
# WINDOW = 1024  -> the window cut at 32K becomes 32x, and 476 sequences fit, at any context length.
# NSA_N = 32     -> 32 selected blocks: budget s/16 + 2,560, so the 64K speedup drops from 11.6x to 9.8x.
# In part2, set B = 50 (one 400 Gb/s NIC per GPU, illustrative) -> an H100 ring over InfiniBand needs c >= 19.8K.
