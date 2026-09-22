"""KV tricks lab: Scaling Book Part 7, "Tricks for Improving Generation Throughput and Latency".

Run: python3 labs/kv-tricks.py   (stdlib only, CPU, well under a second)

Running example: Llama-3.1-8B shapes on one H100 SXM (course _facts.md).
Start from a hypothetical multi-head (MHA) version of the model and apply one KV trick per row.
For each row:
  KV bytes per token  (averaged over layers for local/global mixes)
  sequences of CONTEXT tokens that fit in the free HBM
  step time = max( (B * KV_seq + W) / HBM_BW ,  2 * P * B / FLOPS )   (book's step-time formula, per sequence-token)
  throughput = B / step
Rows marked "hypothetical" are NOT what Llama-3.1-8B does; they show what the trick would buy on this shape.
"""
import math

# Llama-3.1-8B (course _facts.md)
L, N, K, H = 32, 32, 8, 128
P = 8.03e9                 # params
W_BF16 = 16.06e9           # bytes of bf16 weights
HBM = 80e9
BW = 3.35e12               # H100 HBM3, B/s
FLOPS_BF16 = 989e12
FLOPS_FP8 = 1979e12

CONTEXT = 8192             # tokens per sequence
WINDOW = 4096              # local-attention window (hypothetical)
GLOBAL_EVERY = 6           # 1 global layer in every 6 -> 5 local : 1 global (hypothetical)
KIB = 1024


def kv_per_token(kv_heads, bytes_per_elem):
    """K and V, every layer, one token of context."""
    return 2 * L * kv_heads * H * bytes_per_elem


def local_global_factor(context, window, global_every=GLOBAL_EVERY, layers=L):
    """Average fraction of a full cache kept per layer when local layers keep only `window` tokens."""
    n_global = sum(1 for i in range(layers) if i % global_every == global_every - 1)
    n_local = layers - n_global
    kept = n_global * context + n_local * min(window, context)
    return kept / (layers * context), n_global, n_local


def row(name, kv_tok_stored, kv_tok_read, weights, flops, context=CONTEXT):
    """kv_tok_stored sets how many fit; kv_tok_read sets the bytes each step reads."""
    free = HBM - weights
    kv_seq_stored = kv_tok_stored * context
    B = math.floor(free / kv_seq_stored)
    t_mem = (B * kv_tok_read * context + weights) / BW
    t_flop = 2 * P * B / flops
    step = max(t_mem, t_flop)
    return dict(name=name, kv_tok=kv_tok_stored, B=B, t_mem=t_mem, t_flop=t_flop,
                step=step, tput=B / step if B else 0.0, kv_seq=kv_seq_stored,
                bound="compute" if t_flop > t_mem else "memory")


def ladder(context=CONTEXT, window=WINDOW, mqa=False):
    rows = []
    mha = kv_per_token(N, 2)
    rows.append(row("MHA, bf16 KV (hypothetical)", mha, mha, W_BF16, FLOPS_BF16, context))
    kvh = 1 if mqa else K
    gqa = kv_per_token(kvh, 2)
    rows.append(row(f"+ {'MQA (1 KV head, hypothetical)' if mqa else 'GQA 32:8 (the real model)'}", gqa, gqa, W_BF16, FLOPS_BF16, context))
    fp8 = gqa / 2
    rows.append(row("+ FP8 KV cache", fp8, fp8, W_BF16, FLOPS_BF16, context))
    f, ng, nl = local_global_factor(context, window)
    loc = fp8 * f
    rows.append(row(f"+ local layers, {window}-token window (hypothetical)", loc, loc, W_BF16, FLOPS_BF16, context))
    shared = loc / 2
    # book: shared KVs "may need to be read from HBM multiple times" -> two read models
    rows.append(row("+ share KV across adjacent layers, reads x1 (hypothetical)", shared, shared, W_BF16, FLOPS_BF16, context))
    rows.append(row("  same, but each layer re-reads the shared KV", shared, loc, W_BF16, FLOPS_BF16, context))
    rows.append(row("+ FP8 weights too (8.03 GB, FP8 matmuls)", shared, shared, W_BF16 / 2, FLOPS_FP8, context))
    return rows, (f, ng, nl)


def show(context=CONTEXT, window=WINDOW, mqa=False):
    rows, (f, ng, nl) = ladder(context, window, mqa)
    print(f"\n=== Llama-3.1-8B shape, H100, context {context:,} tokens, local window {window:,} ===")
    print(f"step time at a full HBM, if every stored byte is read once: 80 GB / 3.35 TB/s = {HBM/BW*1e3:.2f} ms")
    print(f"local/global mix: {ng} global + {nl} local layers (1 in {GLOBAL_EVERY} global); "
          f"avg kept fraction = ({ng}*{context} + {nl}*{min(window, context)}) / ({L}*{context}) = {f:.4f}")
    base = rows[0]
    print(f"{'row':58s} {'KV/token':>10s} {'x vs MHA':>9s} {'GB/seq':>7s} {'fit':>5s} "
          f"{'t_mem ms':>9s} {'t_flop ms':>9s} {'tok/s':>9s} {'x tput':>7s}")
    for r in rows:
        print(f"{r['name']:58s} {r['kv_tok']/KIB:8.2f}Ki {base['kv_tok']/r['kv_tok']:9.2f} "
              f"{r['kv_seq']/1e9:7.3f} {r['B']:5d} {r['t_mem']*1e3:9.2f} {r['t_flop']*1e3:9.2f} "
              f"{r['tput']:9.0f} {(r['tput']/base['tput'] if base['tput'] else float('nan')):7.2f}  "
              f"{r['bound'] if r['B'] else 'does not fit'}")
    return rows


def padding(lengths=(1200, 3000, 6500, 8192), slot=CONTEXT, block=16):
    """Illustrative request lengths: padded 8k slots vs ragged reads vs 16-token pages."""
    tok = kv_per_token(K, 2)
    used = sum(lengths)
    padded = slot * len(lengths)
    paged = sum(math.ceil(n / block) * block for n in lengths)
    print(f"\n=== padding (illustrative lengths {lengths}, {slot}-token slots, {block}-token pages) ===")
    print(f"padded slots:  {padded:,} tokens = {padded*tok/1e9:.3f} GB reserved and read")
    print(f"real tokens:   {used:,} tokens = {used*tok/1e9:.3f} GB  ({used/padded:.1%} of the padded bytes)")
    print(f"ragged reads skip {1-used/padded:.1%} of the KV reads, but the 8k slots are still reserved")
    print(f"paged storage: {paged:,} tokens = {paged*tok/1e9:.3f} GB  (waste {paged-used} tokens, "
          f"{(paged-used)/paged:.2%}; at most {block-1} per request)")


if __name__ == "__main__":
    print("KV per token, bf16: MHA 2*L*N*H*2 =", kv_per_token(N, 2), "B;  GQA 2*L*K*H*2 =", kv_per_token(K, 2), "B")
    print(f"free HBM = 80 - 16.06 = {(HBM - W_BF16)/1e9:.2f} GB; weights-only step = {W_BF16/BW*1e3:.2f} ms")
    ridge = FLOPS_BF16 / BW
    print(f"B_crit, bf16 weights + bf16 compute: intensity = 2BP/(2P) = B  -> B_crit = {ridge:.0f}")
    print(f"B_crit, 8-bit weights + bf16 compute: intensity = 2BP/P = 2B -> B_crit = {ridge/2:.0f}  (book, v5e: 240 -> 120)")
    print(f"weights-only step with FP8 weights = {W_BF16/2/BW*1e3:.2f} ms")
    rows = show()
    # log-scale bar widths for the card: log10(x)/log10(100) * 100%
    base = rows[0]
    print("\nbar widths (log scale, 100x = full bar):")
    for r in rows:
        kx = base['kv_tok'] / r['kv_tok']
        tx = r['tput'] / base['tput']
        print(f"  {r['name']:58s} KV {kx:6.2f}x -> {math.log10(kx)/2*100:5.1f}%   tput {tx:5.2f}x -> {math.log10(tx)/2*100:5.1f}%")
    show(context=131072)
    show(mqa=True)
    padding()

# Try this:
# 1. show(context=131072): at 128k a 4k window keeps only 3% of each local layer, so the local/global
#    row wins far more than at 8k, and MHA cannot fit even one sequence.
# 2. show(mqa=True): one KV head instead of 8 shrinks KV another 8x (the Character.ai choice), and the
#    ladder hits the compute roof (t_flop > t_mem): batch no longer buys throughput.
# 3. Change GLOBAL_EVERY to 2 (Gemma-2-style alternation) and WINDOW to 1024 (Gemma 3 / Character.ai span).
