"""FlashAttention-2 II: how many thread blocks, and how much warp-to-warp traffic?

Two counting models (ours, not measurements):
  1. Thread blocks per attention forward pass.
     FA1 launches one block per (batch, head):          blocks = batch * heads
     FA2 also splits the sequence into row blocks:      blocks = batch * heads * ceil(N / Br)
     We assume ONE resident block per SM, so a "wave" is one block on every SM.
  2. Shared-memory (SMEM) traffic between warps inside one block, per inner-loop
     iteration (one K_j, V_j tile of Bc columns):
     split-K (FA1): each of W warps holds all Br rows of Q and a Bc/W slice of K,V,
                    so each produces a partial Br x d output that must be written to
                    SMEM (fp32), synchronised, read back and added. Row max and row
                    sum (2 fp32 per row per warp) must be combined the same way.
     split-Q (FA2): each warp owns Br/W rows and sees all of K,V: nothing to exchange.
     The K,V tile loads are the same in both schemes, so we leave them out of the
     exchange count and print them only for scale.
Run: python3 labs/fa2-occupancy.py   (stdlib only, < 1 s)
"""
from math import ceil

BR = 128  # query rows per thread block (FA2 picks from {64, 128})


def blocks(batch, heads, n, br=BR):
    fa1 = batch * heads
    fa2 = batch * heads * ceil(n / br)
    return fa1, fa2


def wave_stats(nblocks, sms):
    first = min(nblocks, sms) / sms          # share of SMs busy in the first wave
    waves = nblocks / sms
    full = ceil(waves)
    eff = nblocks / (full * sms)             # average SM use over all waves
    last = nblocks - (full - 1) * sms        # blocks in the last wave
    return first, waves, eff, last


def show(name, batch, heads, n, sms):
    fa1, fa2 = blocks(batch, heads, n)
    print(f"\n{name}: batch {batch} x heads {heads}, N = {n:,}, Br = {BR}, {sms} SMs")
    for label, nb in (("FA1 (batch x heads)", fa1), ("FA2 (+ row blocks)", fa2)):
        first, waves, eff, last = wave_stats(nb, sms)
        print(f"  {label:20s} {nb:6,d} blocks | first wave {first:6.1%} of SMs | "
              f"{waves:6.2f} waves | last wave {last:3d} blocks | avg SM use {eff:6.1%}")


print("=== 1. Thread blocks vs SMs (our model: one resident block per SM) ===")
# FA2 paper benchmark (sec 4.1): A100, 16k tokens total, hidden 2048, d = 128 -> 16 heads
show("A100 paper setup, short seqs", 32, 16, 512, 108)
show("A100 paper setup, long seq", 1, 16, 16384, 108)
# Running example: Llama-3.1-8B prefill on H100 (32 query heads)
show("H100 Llama-3.1-8B prefill", 1, 32, 8192, 132)
show("H100 Llama-3.1-8B prefill, batch 4", 4, 32, 8192, 132)
show("H100 Llama-3.1-8B prefill, batch 8", 8, 32, 8192, 132)
# Decode: one new query token per sequence. Row blocks = ceil(1/128) = 1.
show("H100 Llama-3.1-8B decode, batch 1", 1, 32, 1, 132)
split = 16  # illustrative number of KV chunks (Flash-Decoding style split over keys)
nb = 1 * 32 * split
first, waves, eff, last = wave_stats(nb, 132)
print(f"  split-KV x{split} (illustr.) {nb:6,d} blocks | first wave {first:6.1%} of SMs | "
      f"{waves:6.2f} waves | {8192 // split} keys per chunk")

print("\n=== 2. Warp exchange through SMEM per inner iteration (our model) ===")


def smem(br, bc, d, w, n):
    part_o = w * br * d * 4            # each warp's fp32 partial O
    stats = w * br * 2 * 4             # each warp's row max + row sum
    wr = part_o + stats
    rd = wr                            # everything written is read back to add
    kv = 2 * bc * d * 2                # K_j + V_j tile in bf16 (same for both schemes)
    iters = ceil(n / bc)
    print(f"\nBr={br}, Bc={bc}, d={d}, W={w} warps, N={n:,} -> {iters} iterations per row block")
    print(f"  split-K: write {wr:,} B + read {rd:,} B = {wr + rd:,} B per iteration, >= 2 __syncthreads")
    print(f"           ({part_o:,} B partial O + {stats:,} B row stats written)")
    print(f"           = {(wr + rd) / kv:.1f}x the K,V tile it just loaded ({kv:,} B)")
    print(f"           over the whole row pass: {(wr + rd) * iters / 2**20:,.1f} MiB, {2 * iters} syncs")
    print(f"  split-Q: 0 B exchanged, 0 syncs for the exchange; each warp owns {br // w} rows")
    print(f"  SMEM needed at once for split-K partials: {part_o / 1024:,.0f} KiB "
          f"(H100 max per block: 227 KiB, A100: 163 KiB)")


smem(64, 64, 64, 4, 8192)
smem(128, 64, 128, 4, 8192)
print("  with W = 8 warps, split-Q gives each warp", 128 // 8, "rows")

# Try this:
# - show("H100 prefill, N=32k", 1, 32, 32768, 132): FA2 blocks grow with N, FA1's do not.
# - Set BR = 64: FA2 doubles its blocks; does the last wave get better or worse?
# - smem(128, 128, 128, 8, 8192): more warps make split-K's exchange grow linearly with W.
