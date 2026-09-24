"""moe-router.py - route tokens through a mixture-of-experts router and count what happens.

Course 6 card: "Mixture of experts: routers, top-k and imbalance".
Pure simulator: numpy only, CPU, seeded, runs in a few seconds. Nothing here is measured.

What it does:
  1. One token, three gating orders: Shazeer/Mixtral (top-k then softmax), Switch (softmax
     then pick top-1), DeepSeek-V3 (sigmoid, top-k, then normalise the kept gates).
  2. Parameter counts: Mixtral 8x7B total vs active from its Table 1 shapes (our recomputation).
  3. A batch of T tokens routed top-1 to 8 experts (Switch style), balanced vs skewed router:
     per-expert load, max/mean, dropped tokens at capacity factor 1.0 / 1.25 / 2.0,
     and Switch's auxiliary loss  alpha * N * sum_i f_i * P_i.
  4. Serving: tokens per expert = B*k/E, the batch that makes each expert compute-bound
     (B_crit * E / k), and how many experts a batch touches: P(untouched) = (1 - k/E)^B.

The router is a toy: logit_i(x) = bias_i + Gumbel noise. With Gumbel noise the top-k choice is
exactly "sample k experts without replacement with probabilities softmax(bias)", so the bias
sets the skew. bias = 0 is a perfectly balanced router (what the aux loss aims for);
bias = s * log(Zipf rank weight) is a skewed one (illustrative, not a real model's routing).
"""
import numpy as np

SEED = 0
ALPHA = 1e-2          # Switch Transformer aux-loss coefficient (paper: alpha = 10^-2)
ZIPF_S = 1.0          # skew of the illustrative skewed router
T = 4096              # tokens in one routing batch (illustrative)
H100_BCRIT = 295      # course facts: 989 TFLOPS / 3.35 TB/s
V5E_BCRIT = 240       # Scaling Book, TPU v5e BF16
V5E_INT8_BCRIT = 120  # Scaling Book Part 4 Q8: int8 weights, bf16 FLOPs

rng = np.random.default_rng(SEED)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def zipf_bias(n, s):
    w = 1.0 / np.arange(1, n + 1) ** s
    w = w / w.sum()
    return np.log(w), w


def route(bias, t, k):
    """Return logits [t, E] and the chosen top-k expert ids [t, k]."""
    logits = bias[None, :] + rng.gumbel(size=(t, bias.size))
    top = np.argsort(-logits, axis=1)[:, :k]
    return logits, top


def loads(top, n):
    return np.bincount(top.ravel(), minlength=n)


def touched(bias, b, k, trials=200):
    """Mean fraction of experts that at least one of b tokens routes to."""
    logits = bias + rng.gumbel(size=(trials, b, bias.size))
    top = np.argpartition(-logits, k - 1, axis=2)[:, :, :k].reshape(trials, -1)
    hit = np.zeros((trials, bias.size), dtype=bool)
    np.put_along_axis(hit, top, True, axis=1)
    return hit.mean()


def dropped(load, capacity):
    return int(np.maximum(load - capacity, 0).sum())


def section_gating():
    print("=== 1. One token, three gating orders (illustrative logits, 8 experts) ===")
    logits = np.array([2.0, 1.2, 0.3, -0.5, 1.6, -1.0, 0.1, 0.7])
    print("router logits x.W_g:", logits.tolist())
    # Shazeer 2017 / Mixtral: KeepTopK (rest -inf), then softmax over the kept ones (noise term off).
    k = 2
    top2 = np.argsort(-logits)[:k]
    g = np.zeros_like(logits)
    g[top2] = softmax(logits[top2])
    print(f"Shazeer/Mixtral  top-2 -> softmax:   experts {top2.tolist()}  gates {np.round(g[top2], 3).tolist()}  (sum {g.sum():.3f})")
    # Switch: softmax over all N, take the argmax, output = p_i(x) * E_i(x) (gate not renormalised).
    p = softmax(logits)
    i = int(np.argmax(p))
    print(f"Switch           softmax -> top-1:   expert {i}  gate p_i = {p[i]:.3f}  (the other {1 - p[i]:.3f} of the mass is unused)")
    # Mixtral-style top-2 read off the full softmax, to show why the order matters.
    print(f"   (top-2 of the full softmax would be {np.round(p[top2], 3).tolist()}, sum {p[top2].sum():.3f})")
    # DeepSeek-V3: s_i = sigmoid(u.e_i), keep top-k, divide by the sum of the kept s_i.
    s = 1 / (1 + np.exp(-logits))
    g3 = s[top2] / s[top2].sum()
    print(f"DeepSeek-V3      sigmoid -> top-2 -> normalise: sigmoids {np.round(s[top2], 3).tolist()}  gates {np.round(g3, 3).tolist()}")
    print("   (V3 uses top-8 of 256; top-2 of 8 here only so the three rows compare.)")


def section_params():
    print("\n=== 2. Mixtral 8x7B: total vs active (our recomputation from Table 1 shapes) ===")
    dim, layers, hidden, heads, kv_heads, head_dim, vocab, E, k = 4096, 32, 14336, 32, 8, 128, 32000, 8, 2
    expert = 3 * dim * hidden                           # SwiGLU: w1, w2, w3
    attn = dim * heads * head_dim * 2 + dim * kv_heads * head_dim * 2
    router = dim * E
    emb = 2 * vocab * dim                               # input embedding + untied LM head
    norms = layers * 2 * dim + dim
    total = layers * (E * expert + attn + router) + emb + norms
    active = layers * (k * expert + attn + router) + emb + norms
    print(f"one expert (3 x {dim} x {hidden})     = {expert / 1e6:.1f}M params")
    print(f"attention per layer                  = {attn / 1e6:.1f}M")
    print(f"total  = 32 x (8 experts + attn) + emb = {total / 1e9:.2f}B   (paper: 47B)")
    print(f"active = 32 x (2 experts + attn) + emb = {active / 1e9:.2f}B   (paper: 13B)")
    print(f"naive '8 x 7B' = 56B overcounts: attention and embeddings are shared, not copied 8 times")


def section_balance():
    E, k = 8, 1
    print(f"\n=== 3. {T:,} tokens, top-1 over {E} experts (Switch style); seed {SEED} ===")
    mean = T * k / E
    for name, bias in (("balanced router (bias 0)", np.zeros(E)),
                       (f"skewed router (Zipf s={ZIPF_S})", zipf_bias(E, ZIPF_S)[0])):
        logits, top = route(bias, T, k)
        ld = loads(top, E)
        print(f"-- {name}")
        print(f"   tokens per expert: {ld.tolist()}   mean {mean:.0f}   max/mean {ld.max() / mean:.2f}")
        for cf in (1.0, 1.25, 2.0):
            cap = int(T / E * cf)
            d = dropped(ld, cap)
            slots = cap * E
            pad = slots - (T * k - d)
            print(f"   capacity factor {cf:<4}: capacity {cap:>4}/expert, dropped {d:>4} = {100 * d / T:5.1f}%,"
                  f" empty slots {pad:>5} = {100 * pad / slots:5.1f}% of compute")
        f = ld / T                                      # fraction dispatched (argmax)
        P = softmax(logits).mean(axis=0)                # mean router probability
        loss = ALPHA * E * np.sum(f * P)
        print(f"   aux loss alpha*N*sum(f*P) = {ALPHA} x {E} x {np.sum(f * P):.4f} = {loss:.4f}"
              f"   ({loss / ALPHA:.2f} x the uniform minimum {ALPHA})")
    _, w = zipf_bias(E, ZIPF_S)
    print(f"   analytic check, f = P = Zipf weights {np.round(w, 3).tolist()}:"
          f" alpha*N*sum(w^2) = {ALPHA} x {E} x {np.sum(w * w):.4f} = {ALPHA * E * np.sum(w * w):.4f}")


def section_serving():
    print("\n=== 4. Serving: tokens per expert and B_crit (decode, one MoE layer) ===")
    models = (("Mixtral 8x7B", 8, 2), ("DeepSeek-V3", 256, 8), ("Qwen3-235B-A22B", 128, 8))
    for name, E, k in models:
        print(f"{name:<16} E={E:<3} k={k}:  per-expert tokens = B x {k}/{E} = B / {E / k:g};"
              f"  B_crit x E/k on H100 = {H100_BCRIT} x {E / k:g} = {H100_BCRIT * E / k:,.0f} tokens")
    fp8w = 989e12 / (2 * 3.35e12)
    print(f"DeepSeek-V3 on H100 with FP8 weights but BF16 math: B_crit = 989e12 / (2 x 3.35e12) = {fp8w:.1f}"
          f" -> x 32 = {fp8w * 32:,.0f} (FP8 weights AND FP8 math: 1979e12 / (2 x 3.35e12) = {1979e12 / 6.7e12:.1f}, same as BF16)")
    print(f"DeepSeek-V3 on TPU v5e: BF16 {V5E_BCRIT} x 32 = {V5E_BCRIT * 32:,}; int8 weights {V5E_INT8_BCRIT} x 32 = {V5E_INT8_BCRIT * 32:,} (Scaling Book Part 4 Q8)")
    for B in (64, 256, 1024, 9440):
        print(f"   DeepSeek-V3 at B = {B:>5}: each expert sees {B * 8 / 256:7.1f} tokens on average")

    print("\n=== 5. How many experts does a batch touch?  P(untouched) = (1 - k/E)^B ===")
    for name, E, k, bs in (("Mixtral 8x7B", 8, 2, (1, 2, 4, 8, 16)),
                           ("DeepSeek-V3", 256, 8, (1, 8, 16, 32, 64, 128, 256))):
        print(f"-- {name}: E={E}, k={k}")
        _, wz = zipf_bias(E, ZIPF_S)
        bias = np.log(wz)
        for B in bs:
            analytic = 1 - (1 - k / E) ** B
            sim_u, sim_z = touched(np.zeros(E), B, k), touched(bias, B, k)
            print(f"   B = {B:>4}: touched (uniform, formula) {100 * analytic:6.2f}%   simulated uniform {100 * sim_u:6.2f}%"
                  f"   simulated Zipf s={ZIPF_S} {100 * sim_z:6.2f}%")

    print("\n=== 6. DeepSeek-V3 bytes read per decode step (FP8, 1 byte/param; our inference) ===")
    total, routed = 671e9, 654e9                        # course facts: routed experts ~654B of 671B
    always = total - routed                             # attention, shared experts, dense layers, embeddings
    for B in (1, 8, 32, 64, 128, 256):
        frac = 1 - (1 - 8 / 256) ** B
        gb = (always + frac * routed) / 1e9
        print(f"   B = {B:>3}: {100 * frac:5.1f}% of routed experts touched -> {gb:5.0f} GB read  ({gb / B:6.2f} GB per token)")


if __name__ == "__main__":
    section_gating()
    section_params()
    section_balance()
    section_serving()

# Try this:
#  1. ZIPF_S = 1.0 (one expert takes ~65% of tokens at E=8): drops at capacity factor 2.0 jump
#     to about 40% of the batch and the aux loss roughly triples.
#  2. T = 512: the balanced router's max/mean grows (fewer tokens per expert, noisier counts),
#     so even perfect routing drops tokens at capacity factor 1.0.
#  3. In section 5, change DeepSeek's k to 1: the batch that touches ~all 256 experts grows 8x.
