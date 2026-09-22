"""Sampling and the KV cache: naive recompute vs. a KV cache.

(a) A tiny numpy decoder-only Transformer generates 20 tokens twice: once by
    re-running the whole prefix every step (naive), once with a KV cache.
    It checks the outputs are identical and counts the matmul FLOPs of each.
(b) The same FLOP accounting, analytically, for Llama-3.1-8B
    (1,000-token prompt, n generated tokens) and the KV cache it builds.

Run: python3 labs/kv-cache-sampling.py   (numpy only, CPU, < 1 s)
"""
import numpy as np

# ---------------------------------------------------------------- (a) tiny model
V, D, F, L, NQ, NKV = 64, 32, 96, 2, 4, 2          # vocab, width, MLP, layers, q heads, kv heads
H = D // NQ                                          # head dim
rng = np.random.default_rng(0)
W = [dict(wq=rng.normal(0, D**-0.5, (D, NQ * H)), wk=rng.normal(0, D**-0.5, (D, NKV * H)),
          wv=rng.normal(0, D**-0.5, (D, NKV * H)), wo=rng.normal(0, D**-0.5, (NQ * H, D)),
          w1=rng.normal(0, D**-0.5, (D, F)), w3=rng.normal(0, D**-0.5, (D, F)),
          w2=rng.normal(0, F**-0.5, (F, D))) for _ in range(L)]
EMB = rng.normal(0, 1, (V, D))
HEAD = rng.normal(0, D**-0.5, (D, V))

flops = {"proj_mlp": 0, "attn": 0}


def mm(a, b, kind="proj_mlp"):
    """Matmul that counts 2*m*k*n FLOPs."""
    flops[kind] += 2 * a.shape[0] * a.shape[1] * b.shape[1]
    return a @ b


def norm(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)


def forward(tokens, cache=None, start=0):
    """Process `tokens` (positions start..start+len-1). If cache is given, read
    earlier K/V from it and append the new ones. Returns last-position logits."""
    x = EMB[tokens]
    s = len(tokens)
    for l, w in enumerate(W):
        h = norm(x)
        q = mm(h, w["wq"]).reshape(s, NQ, H)
        k = mm(h, w["wk"]).reshape(s, NKV, H)
        v = mm(h, w["wv"]).reshape(s, NKV, H)
        if cache is not None:
            cache[l]["k"].append(k)
            cache[l]["v"].append(v)
            k = np.concatenate(cache[l]["k"])
            v = np.concatenate(cache[l]["v"])
        out = np.empty((s, NQ, H))
        for i in range(s):                       # causal: query at position start+i sees keys 0..start+i
            t = start + i + 1
            for n in range(NQ):
                g = n // (NQ // NKV)             # GQA: query head n reads KV head g
                sc = mm(q[i, n][None], k[:t, g].T, "attn")[0] / np.sqrt(H)
                p = np.exp(sc - sc.max())
                p /= p.sum()
                out[i, n] = mm(p[None], v[:t, g], "attn")[0]
        x = x + mm(out.reshape(s, NQ * H), w["wo"])
        h = norm(x)
        a, b = mm(h, w["w1"]), mm(h, w["w3"])
        x = x + mm(a / (1 + np.exp(-a)) * b, w["w2"])
    return mm(norm(x[-1:]), HEAD)[0]


prompt = [int(t) for t in rng.integers(0, V, 8)]
N_GEN = 20
GUMBEL = rng.gumbel(size=(N_GEN, V))       # pre-drawn noise: argmax(logits + Gumbel) samples softmax(logits),
                                            # and both runs use the same draws so their samples are comparable

# naive: every step re-processes the whole prefix
flops.update(proj_mlp=0, attn=0)
seq = list(prompt)
naive_logits = []
for _ in range(N_GEN):
    lg = forward(np.array(seq))
    naive_logits.append(lg)
    seq.append(int((lg + GUMBEL[len(naive_logits) - 1]).argmax()))
naive_out, naive_flops = seq[len(prompt):], dict(flops)

# cached: prefill once, then one token per step
flops.update(proj_mlp=0, attn=0)
cache = [dict(k=[], v=[]) for _ in range(L)]
lg = forward(np.array(prompt), cache, 0)                     # prefill
seq = list(prompt)
cached_logits = []
for step in range(N_GEN):
    cached_logits.append(lg)
    seq.append(int((lg + GUMBEL[step]).argmax()))
    if step < N_GEN - 1:                                       # feed the new token back (generation)
        lg = forward(np.array(seq[-1:]), cache, len(seq) - 1)
cached_out, cached_flops = seq[len(prompt):], dict(flops)

assert naive_out == cached_out, "outputs differ!"
assert np.allclose(naive_logits, cached_logits, atol=1e-9)
cache_len = sum(len(c) for c in cache[0]["k"])

print("(a) Tiny decoder: V=%d D=%d F=%d L=%d, %d q heads / %d kv heads, prompt %d, generate %d"
      % (V, D, F, L, NQ, NKV, len(prompt), N_GEN))
print("    generated tokens identical:", naive_out == cached_out, naive_out)
print("    max |logit diff|: %.1e" % np.abs(np.array(naive_logits) - np.array(cached_logits)).max())
print("    %-22s %14s %14s %8s" % ("matmul FLOPs", "naive", "KV cache", "ratio"))
for k, name in [("proj_mlp", "projections + MLP"), ("attn", "attention q.k and p.v")]:
    print("    %-22s %14s %14s %7.1fx" % (name, f"{naive_flops[k]:,}", f"{cached_flops[k]:,}",
                                         naive_flops[k] / cached_flops[k]))
tn, tc = sum(naive_flops.values()), sum(cached_flops.values())
print("    %-22s %14s %14s %7.1fx" % ("total", f"{tn:,}", f"{tc:,}", tn / tc))
print("    KV cache entries per layer at the end: %d (prompt %d + %d generated, last not yet fed back)"
      % (cache_len, len(prompt), N_GEN - 1))

# ---------------------------------------------------------------- (b) Llama-3.1-8B
Lm, Dm, Km, Hm, Nparams = 32, 4096, 8, 128, 8.03e9
KV_BYTES = 2 * Lm * Km * Hm * 2                      # K and V, every layer, bf16
ATT = 4 * Lm * Dm                                    # q.k + p.v FLOPs per (query, key) pair, all layers (N*H = D)
H100_FLOPS, H100_BW, WEIGHTS = 989e12, 3.35e12, 16.06e9


def llama(P, n):
    """P prompt tokens, sample n tokens. Causal attention: query at position t sees t keys."""
    tri = lambda s: s * (s + 1) // 2                  # sum_{t=1..s} t
    naive_tok = sum(P + j for j in range(n))                         # passes over prefixes P..P+n-1
    naive_att = ATT * sum(tri(P + j) for j in range(n))
    cached_tok = P + (n - 1)                                         # prefill + n-1 one-token steps
    cached_att = ATT * tri(P + n - 1)                                # = one triangle over the final length
    return naive_tok, naive_att, cached_tok, cached_att


print("\n(b) Llama-3.1-8B (L=32, D=4096, K=8, H=128, N=8.03B params), matmul FLOPs ~ 2N per token processed")
print("    KV bytes per token = 2*L*K*H*2 = %s B = %d KiB" % (f"{KV_BYTES:,}", KV_BYTES // 1024))
P, n = 1000, 200
nt, na, ct, ca = llama(P, n)
print("\n    Running example: prompt %d, generate %d" % (P, n))
print("    tokens processed           naive %9s   cached %6s   (%.1fx)" % (f"{nt:,}", f"{ct:,}", nt / ct))
print("    projection+MLP FLOPs (2N)  naive %.3e  cached %.3e  (%.1fx)" % (2 * Nparams * nt, 2 * Nparams * ct, nt / ct))
print("    attention FLOPs (4LD*keys) naive %.3e  cached %.3e  (%.1fx)" % (na, ca, na / ca))
print("    total                      naive %.3e  cached %.3e  (%.1fx)"
      % (2 * Nparams * nt + na, 2 * Nparams * ct + ca, (2 * Nparams * nt + na) / (2 * Nparams * ct + ca)))
print("    attention share of total   naive %.1f%%     cached %.1f%%"
      % (100 * na / (2 * Nparams * nt + na), 100 * ca / (2 * Nparams * ct + ca)))
print("    attention per decode token at 1,000 keys: 4*L*D*1000 = %.3e vs 2N = %.3e (%.1f%%)"
      % (ATT * 1000, 2 * Nparams, 100 * ATT * 1000 / (2 * Nparams)))
for T in (P + n - 1, P + n):
    print("    KV cache for %d tokens: %d x 128 KiB = %.1f MB (%.1f MiB)" % (T, T, T * KV_BYTES / 1e6, T * KV_BYTES / 2**20))

# time floors (our recomputation, not in the book): each pass takes at least
# max(FLOPs / peak FLOP/s, bytes read / HBM bandwidth); attention FLOPs ignored (~2%).
# TPU v5e has 16 GB HBM, so 16.06 GB of bf16 weights would not actually fit on one chip:
# the v5e row is "as if it fit", to show the ratio is similar on other hardware.
print("\n    Time floors, one chip (roofline: max of compute and weight+KV reads):")
for chip, fl, bw in [("H100 SXM", H100_FLOPS, H100_BW), ("TPU v5e*", 1.97e14, 8.2e11)]:
    naive_t = sum(max(2 * Nparams * (P + j) / fl, WEIGHTS / bw) for j in range(n))
    pre_t = max(2 * Nparams * P / fl, WEIGHTS / bw)
    dec_t = sum(max(2 * Nparams / fl, (WEIGHTS + (P + j) * KV_BYTES) / bw) for j in range(1, n))
    print("    %-9s naive %6.2f s | cached %.3f s = prefill %5.1f ms + %d decode steps x %5.2f ms"
          " | %.1fx faster (FLOPs fell %.0fx)"
          % (chip, naive_t, pre_t + dec_t, pre_t * 1e3, n - 1, dec_t / (n - 1) * 1e3,
             naive_t / (pre_t + dec_t), nt / ct))
print("    * 16.06 GB of weights doesn't fit one v5e's 16 GB; shown as if it did")

print("\n    As n grows (prompt fixed at %d):" % P)
print("    %8s %14s %10s %8s %12s %12s %8s %10s" % ("n gen", "naive tokens", "cached", "ratio",
                                                     "naive attn", "cached attn", "ratio", "KV cache"))
for n in (100, 1000, 10000):
    nt, na, ct, ca = llama(P, n)
    print("    %8s %14s %10s %7.0fx %12.2e %12.2e %7.0fx %8.2f GB"
          % (f"{n:,}", f"{nt:,}", f"{ct:,}", nt / ct, na, ca, na / ca, ct * KV_BYTES / 1e9))

print("\n    Book's claim, prompt ~0 (generate n from a 1-token start): MLP ratio ~ n/2, attention ratio ~ n/3")
for n in (100, 1000, 10000):
    nt, na, ct, ca = llama(1, n)
    print("    n=%6s  MLP ratio %8.0fx   attention ratio %8.0fx" % (f"{n:,}", nt / ct, na / ca))

# Try this:
# 1. Double n in the "as n grows" loop (e.g. 200 -> 400 -> 800): the token ratio roughly doubles
#    once n >> prompt (O(n^2) vs O(n)); the attention ratio grows the same way (O(n^3) vs O(n^2)).
# 2. Set N_GEN = 40 in part (a): the tiny model's projection+MLP ratio roughly doubles too.
# 3. Set NKV = NQ (plain multi-head attention) in part (a): the attention FLOPs don't change,
#    but each cached token now stores twice as many K/V values -- GQA shrinks the cache, not attention compute.
