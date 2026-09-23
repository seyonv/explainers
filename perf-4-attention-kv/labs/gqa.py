"""GQA lab: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints" (EMNLP 2023, arXiv 2305.13245).

Run: python3 labs/gqa.py   (numpy, CPU, about a second)

Part 1  Llama-3.1-8B on one H100 SXM with G KV heads (groups) instead of the real 8.
        For G = 32 (MHA), 8 (the real model), 4, 2, 1 (MQA):
          KV bytes per token = 2 * L * G * H * 2 B
          sequences of 8,192 tokens that fit in the 63.94 GB left after BF16 weights
          decode step at that batch = B*KV_seq/BW + max(2*B*P/C, W/BW)   (perf-2 step formula)
          throughput = B / step
          decode attention intensity = r*S/(r+S), r = query heads per group = N/G
        Only G = 8 is the real model; the other rows are hypothetical versions of its shape.
        Our recomputation, not in the paper.

Part 2  A TOY of the paper's checkpoint conversion (section 2.1-2.2): take random MHA K and V
        projection heads, mean-pool them within each group, and measure how far the attention
        output moves BEFORE any uptraining. The paper's 'First' baseline (keep one head per group)
        is printed alongside. Run twice: independent random heads (worst case for pooling) and heads
        sharing 80% of their projection variance (illustrative). This shows the mechanism and the
        direction (fewer groups = more change, mean beats first), not a quality result.
"""
import math
import numpy as np

# ---------------- Part 1: Llama-3.1-8B, H100 (course _facts.md) ----------------
L, N, H = 32, 32, 128          # layers, query heads, head dim
P = 8.03e9                     # params
W = 16.06e9                    # BF16 weight bytes
HBM = 80e9
FREE = HBM - W                 # 63.94 GB
BW = 3.35e12                   # B/s
C = 989e12                     # dense BF16 FLOP/s
CTX = 8192
KIB = 1024


def row(G, ctx=CTX):
    kv_tok = 2 * L * G * H * 2
    kv_seq = kv_tok * ctx
    B = math.floor(FREE / kv_seq)
    t_kv = B * kv_seq / BW
    t_w = W / BW
    t_c = 2 * B * P / C
    step = t_kv + max(t_c, t_w)
    r = N // G
    inten = r * ctx / (r + ctx)
    return dict(G=G, r=r, kv_tok=kv_tok, kv_seq=kv_seq, B=B, t_kv=t_kv, t_w=t_w, t_c=t_c,
                step=step, tput=B / step, inten=inten)


def part1(ctx=CTX):
    print(f"=== Part 1: Llama-3.1-8B shape, H100, {ctx:,}-token sequences, free HBM {FREE/1e9:.2f} GB ===")
    print(f"{'G (KV heads)':<16}{'q/group':>8}{'KV/token':>11}{'KV/seq':>9}{'fit':>6}"
          f"{'KV read':>10}{'weights':>9}{'compute':>9}{'step':>9}{'tok/s':>8}{'intensity':>11}")
    rows = [row(G, ctx) for G in (32, 8, 4, 2, 1)]
    names = {32: "32 (MHA)", 8: "8 (real)", 1: "1 (MQA)"}
    for d in rows:
        print(f"{names.get(d['G'], str(d['G'])):<16}{d['r']:>8}{d['kv_tok']/KIB:>8.0f} KiB"
              f"{d['kv_seq']/1e9:>7.3f} GB{d['B']:>5}"
              f"{d['t_kv']*1e3:>8.2f} ms{d['t_w']*1e3:>7.2f} ms{d['t_c']*1e3:>7.2f} ms"
              f"{d['step']*1e3:>7.2f} ms{d['tput']:>8,.0f}{d['inten']:>9.2f} F/B")
    base = rows[0]["tput"]
    print("\nthroughput vs MHA: " + ", ".join(f"G={d['G']}: {d['tput']/base:.2f}x" for d in rows))
    for d in rows:
        bound = "compute" if d["t_c"] > d["t_w"] else "weights"
        print(f"  G={d['G']:<2} non-KV part of the step is {bound}-bound "
              f"(2*{d['B']}*8.03e9/989e12 = {d['t_c']*1e3:.2f} ms vs 16.06 GB/3.35 TB/s = {d['t_w']*1e3:.2f} ms); "
              f"KV share of step {d['t_kv']/d['step']*100:.0f}%")
    return rows


# ---------------- Part 2: toy mean-pool conversion ----------------
def attention(x, Wq, Wk, Wv, n_heads, n_groups, hd):
    """Causal self-attention; query head h uses KV head h // (n_heads // n_groups). Returns [T, n_heads*hd]."""
    T = x.shape[0]
    q = (x @ Wq).reshape(T, n_heads, hd)
    k = (x @ Wk).reshape(T, n_groups, hd)
    v = (x @ Wv).reshape(T, n_groups, hd)
    per = n_heads // n_groups
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    out = np.empty((T, n_heads, hd))
    for h in range(n_heads):
        g = h // per
        s = q[:, h] @ k[:, g].T / math.sqrt(hd)
        s[mask] = -np.inf
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        out[:, h] = p @ v[:, g]
    return out.reshape(T, n_heads * hd)


def mean_pool(Wkv, n_heads, n_groups, hd):
    """[D, n_heads*hd] -> [D, n_groups*hd]: average the projection heads inside each group (paper 2.2)."""
    D = Wkv.shape[0]
    per = n_heads // n_groups
    return Wkv.reshape(D, n_groups, per, hd).mean(axis=2).reshape(D, n_groups * hd)


def first_head(Wkv, n_heads, n_groups, hd):
    """The paper's 'First' baseline (Fig 4): keep the first original head of each group."""
    D = Wkv.shape[0]
    per = n_heads // n_groups
    return Wkv.reshape(D, n_groups, per, hd)[:, :, 0, :].reshape(D, n_groups * hd)


def part2(seed=0, D=256, n_heads=32, hd=8, T=64, shared=0.0):
    """shared in [0,1): fraction of each head's K/V projection that is common to all heads (0 = independent)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, D))
    Wq = rng.standard_normal((D, n_heads * hd)) / math.sqrt(D) * 2.0
    common_k = rng.standard_normal((D, 1, hd))
    common_v = rng.standard_normal((D, 1, hd))
    own_k = rng.standard_normal((D, n_heads, hd))
    own_v = rng.standard_normal((D, n_heads, hd))
    a, b = math.sqrt(shared), math.sqrt(1 - shared)
    Wk = ((a * common_k + b * own_k) / math.sqrt(D) * 2.0).reshape(D, n_heads * hd)
    Wv = ((a * common_v + b * own_v) / math.sqrt(D)).reshape(D, n_heads * hd)
    ref = attention(x, Wq, Wk, Wv, n_heads, n_heads, hd)
    print(f"\n=== Part 2: TOY mean-pool conversion, before any uptraining "
          f"(random weights, {n_heads} heads x {hd} dims, D={D}, T={T}, seed={seed}, shared={shared}) ===")
    print("relative output change = ||O_converted - O_MHA|| / ||O_MHA||   (0 = identical)")
    print(f"{'groups G':<10}{'q/group':>8}{'KV params kept':>16}{'mean-pool':>11}{'first head':>12}")
    res = {}
    for G in (32, 16, 8, 4, 2, 1):
        rels = []
        for conv in (mean_pool, first_head):
            o = attention(x, Wq, conv(Wk, n_heads, G, hd), conv(Wv, n_heads, G, hd), n_heads, G, hd)
            rels.append(np.linalg.norm(o - ref) / np.linalg.norm(ref))
        res[G] = rels
        print(f"{G:<10}{n_heads // G:>8}{G / n_heads * 100:>15.1f}%{rels[0]:>11.3f}{rels[1]:>12.3f}")
    print(f"mean-pool, G=8 vs G=1: {res[8][0]:.3f} vs {res[1][0]:.3f} "
          "(a toy: direction only; the paper's quality numbers are its Fig 4 and Fig 5)")
    return res


if __name__ == "__main__":
    part1()
    part2(shared=0.0)   # independent random heads: the worst case for pooling
    part2(shared=0.8)   # illustrative: heads that share 80% of their projection variance

# Try this:
#   part1(ctx=32768)          # at 32k MHA fits 3 sequences, the real model 14, MQA 119
#   part2(shared=0.95)        # the more alike the heads, the less pooling changes the output
#   part2(seed=1)             # a different random model: same ordering, slightly different numbers
