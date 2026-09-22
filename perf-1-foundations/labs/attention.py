"""Attention: Q, K, V, heads, and GQA -- the numbers behind attention.html.

Run: python3 labs/attention.py   (numpy only, CPU, < 1 s)

Part 1  a fully computed causal attention for 3 tokens, d = 4
Part 2  multi-head attention with GQA grouping (checks GQA with K = N is MHA)
Part 3  KV-cache bytes per token for Llama-3.1-8B under MHA / GQA-8 / MQA
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # subtract the row max: same result, no overflow
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_attention(Q, K, V):
    T, d = Q.shape
    scores = Q @ K.T                                   # [T, T]: query i . key j
    scaled = scores / np.sqrt(d)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)   # j > i is the future
    masked = np.where(mask, -np.inf, scaled)
    weights = softmax(masked)                          # each row sums to 1
    return scores, scaled, masked, weights, weights @ V


# ---------------- Part 1: the toy example (values are illustrative) ----------------
print("=" * 64)
print("PART 1  causal attention, T = 3 tokens, d = 4")
print("=" * 64)
tokens = ["The", "cat", "sat"]
Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 1, 1, 0]], dtype=float)
K = np.array([[1, 0, 1, 0],
              [0, 1, 1, 0],
              [1, 1, 0, 1]], dtype=float)
V = np.array([[1, 0, 0, 0],
              [0, 2, 0, 0],
              [0, 0, 3, 0]], dtype=float)
print("Q =\n", Q, "\nK =\n", K, "\nV =\n", V)

scores, scaled, masked, W, out = causal_attention(Q, K, V)
print("\nscores = Q K^T  (row = asking token, column = offering token)\n", scores)
print("\nscaled = scores / sqrt(4) = scores / 2\n", scaled)
print("\nmasked (future set to -inf)\n", masked)
print("\nweights = softmax(masked) per row\n", W)
print("row sums:", W.sum(axis=1))
print("\noutput = weights @ V\n", out)
for i, t in enumerate(tokens):
    parts = " + ".join(f"{W[i, j]:.3f}*V[{tokens[j]}]" for j in range(i + 1))
    print(f"  out[{t}] = {parts} = {out[i]}")
# a worked softmax row, for the card
r = scaled[2]
e = np.exp(r)
print("\nrow 'sat' by hand: exp(", r, ") =", e, " sum =", f"{e.sum():.3f}",
      " -> weights", e / e.sum())

# ---------------- Part 2: heads and GQA grouping ----------------
print("\n" + "=" * 64)
print("PART 2  multi-head attention with GQA grouping")
print("=" * 64)


def mha(x, Wq, Wk, Wv, Wo, n_heads, head_dim):
    """Textbook MHA: every query head has its own K and V head."""
    T = x.shape[0]
    q = (x @ Wq).reshape(T, n_heads, head_dim)
    k = (x @ Wk).reshape(T, n_heads, head_dim)
    v = (x @ Wv).reshape(T, n_heads, head_dim)
    outs = [causal_attention(q[:, h], k[:, h], v[:, h])[4] for h in range(n_heads)]
    return np.concatenate(outs, axis=-1) @ Wo


def gqa(x, Wq, Wk, Wv, Wo, n_heads, n_kv, head_dim):
    """GQA: n_heads query heads share n_kv K/V heads; query head h reads kv head h // group."""
    assert n_heads % n_kv == 0
    group = n_heads // n_kv
    T = x.shape[0]
    q = (x @ Wq).reshape(T, n_heads, head_dim)
    k = (x @ Wk).reshape(T, n_kv, head_dim)           # only n_kv heads are ever computed or cached
    v = (x @ Wv).reshape(T, n_kv, head_dim)
    outs = [causal_attention(q[:, h], k[:, h // group], v[:, h // group])[4] for h in range(n_heads)]
    return np.concatenate(outs, axis=-1) @ Wo


rng = np.random.default_rng(0)
T, D, N, H = 16, 64, 8, 8                              # tiny: 8 query heads of 8 dims
x = rng.standard_normal((T, D))
Wq = rng.standard_normal((D, N * H)) / np.sqrt(D)
Wo = rng.standard_normal((N * H, D)) / np.sqrt(N * H)
WkN = rng.standard_normal((D, N * H)) / np.sqrt(D)
WvN = rng.standard_normal((D, N * H)) / np.sqrt(D)

ref = mha(x, Wq, WkN, WvN, Wo, N, H)
same = gqa(x, Wq, WkN, WvN, Wo, N, N, H)
assert np.allclose(ref, same), "GQA with K = N must equal MHA"
print(f"check: GQA with K = N = {N} equals MHA  (max diff {np.abs(ref - same).max():.1e})  OK")

for n_kv in (8, 4, 2, 1):
    Wk = WkN[:, : n_kv * H]
    Wv = WvN[:, : n_kv * H]
    y = gqa(x, Wq, Wk, Wv, Wo, N, n_kv, H)
    kv_floats = 2 * T * n_kv * H
    name = {8: "MHA", 1: "MQA"}.get(n_kv, f"GQA-{n_kv}")
    print(f"  {name:6s} N={N} K={n_kv}  group={N // n_kv}  KV floats cached for {T} tokens = {kv_floats:5d}"
          f"  query heads still {N}, output shape {y.shape}")

# causality check: token 0's output must not change if later tokens change
x2 = x.copy(); x2[5:] = rng.standard_normal((T - 5, D))
a = gqa(x, Wq, WkN[:, :2 * H], WvN[:, :2 * H], Wo, N, 2, H)
b = gqa(x2, Wq, WkN[:, :2 * H], WvN[:, :2 * H], Wo, N, 2, H)
assert np.allclose(a[:5], b[:5]), "causal mask leak"
print("check: changing tokens 5..15 leaves outputs 0..4 unchanged (causal)  OK")

# ---------------- Part 3: Llama-3.1-8B KV bytes per token ----------------
print("\n" + "=" * 64)
print("PART 3  Llama-3.1-8B KV cache: 2 (K and V) x L x K x H x 2 bytes (BF16)")
print("=" * 64)
L, Dm, Nq, Hd = 32, 4096, 32, 128
free_hbm = 80e9 - 16.06e9                          # _facts.md: H100 80 GB minus BF16 weights
seq = 8192
print(f"{'variant':8s} {'KV heads':>8s} {'bytes/token':>12s} {'KiB':>6s} {'8k seq (GB)':>12s}"
      f" {'8k seqs in 63.9 GB':>19s} {'attn params/layer':>18s}")
for name, kv in (("MHA", 32), ("GQA-8", 8), ("MQA", 1)):
    b = 2 * L * kv * Hd * 2
    per_seq = b * seq
    fit = int(free_hbm // per_seq)
    params = 2 * Dm * (Nq + kv) * Hd               # Wq, Wo (N heads) + Wk, Wv (K heads)
    print(f"{name:8s} {kv:8d} {b:12,d} {b // 1024:6d} {per_seq / 1e9:12.3f} {fit:19d} {params / 1e6:17.2f}M")
print(f"  arithmetic: MHA 2*32*32*128*2 = {2*32*32*128*2:,}; GQA-8 2*32*8*128*2 = {2*32*8*128*2:,};"
      f" MQA 2*32*1*128*2 = {2*32*1*128*2:,}")
print(f"  MHA / GQA-8 = {32 // 8}x,  GQA-8 / MQA = {8 // 1}x")

print("\n8,192-token sequences that fit next to the 16.06 GB of BF16 weights (HBM from _facts.md):")
for chip, hbm in (("H100 SXM", 80e9), ("H200", 141e9), ("B200", 192e9)):
    row = [int((hbm - 16.06e9) // (2 * L * kv * Hd * 2 * seq)) for kv in (32, 8, 1)]
    print(f"  {chip:9s} {hbm / 1e9:5.0f} GB  free {(hbm - 16.06e9) / 1e9:6.2f} GB   MHA {row[0]:4d}   GQA-8 {row[1]:4d}   MQA {row[2]:5d}")

print("\nscore matrix entries per head per layer grow as T^2:")
for t in (3, 1024, 8192, 131072):
    print(f"  T = {t:>7,d}: {t * t:>17,d} scores  (causal: {t * (t + 1) // 2:,} are used)")

# Try this:
# 1. Change the group size: in Part 2 set N = 32 and loop n_kv over (32, 8, 4, 1). KV floats shrink
#    by the group size; the query-head count (and so the Q.K^T FLOPs) does not change.
# 2. In Part 1, lengthen the sequence: stack more rows onto Q, K, V (e.g. T = 6) and watch
#    the score matrix become T x T while only the lower triangle survives the mask.
# 3. In Part 3, change H or L (e.g. a 70B model: L = 80, K = 8, H = 128) and recompute bytes/token.
