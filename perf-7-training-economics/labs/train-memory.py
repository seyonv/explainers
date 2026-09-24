"""Where a training step's memory goes: weights, gradients, optimizer state, activations.

A calculator (no measurements). It computes, for Llama 3.1 8B / 70B / 405B:
  - parameter counts from the published shapes (Llama 3 herd, Table 3; vocab 128,256)
  - bytes of training state under four conventions (ZeRO/Playbook 16, Playbook with
    FP32 gradient accumulation 20, Scaling Book 10, DeepSeek-V3 15)
  - activation memory per layer from Korthikanti et al. 2022, Eq. 1:
        sbh * (34 + 5as/h)   bytes, 16-bit activations, no recomputation
    plus the two recompute variants from their Table 2 (t = 1):
        selective (attention core recomputed, what FlashAttention does)  34 * sbh
        full (only each layer's input kept)                                2 * sbh
    The 34 was derived for GPT-style layers (4h GeLU MLP, full multi-head K/V, dropout).
    Llama uses SwiGLU, GQA and no dropout, so for Llama it is an approximation.
  - FLOPs per step: 2 N per token forward + 4 N backward = 6 N per token
  - what fits one 80 GB H100, and the serving footprint of the same 8B model

Run: python3 labs/train-memory.py   (stdlib only, < 1 s)
GB = 1e9 bytes throughout (the Playbook and ZeRO use decimal GB too).
"""

GB = 1e9
H100_HBM = 80e9
H100_BF16 = 989e12            # dense BF16 FLOP/s (course facts)
MFU = 0.40                    # assumed, as in the course's training example

# name: (layers L, hidden h, FFN F, query heads a, KV heads K, head dim, vocab V)
MODELS = {
    "8B":   (32, 4096, 14336, 32, 8, 128, 128256),
    "70B":  (80, 8192, 28672, 64, 8, 128, 128256),
    "405B": (126, 16384, 53248, 128, 8, 128, 128256),
}


def params(L, h, F, a, K, hd, V):
    attn = h * a * hd * 2 + h * K * hd * 2      # W_Q, W_O + W_K, W_V
    mlp = 3 * h * F                              # gate, up, down (SwiGLU)
    return L * (attn + mlp + 2 * h) + 2 * V * h + h   # + 2 RMSNorms/layer, untied embed + head, final norm


# bytes per parameter: (label, weights, grads, master FP32, Adam m + v, FP32 grad accumulator)
CONVENTIONS = [
    ("ZeRO 3.1 / Playbook (K = 12)", 2, 2, 4, 8, 0),
    ("Playbook + FP32 grad acc (Llama 3)", 2, 2, 4, 8, 4),
    ("Scaling Book (bf16 params + fp32 m, v)", 2, 0, 0, 8, 0),
    ("DeepSeek-V3 (Playbook FP8 table)", 1, 2, 4, 4, 4),
]


def act_per_layer(s, b, h, a, mode="none"):
    sbh = s * b * h
    if mode == "none":
        return sbh * (34 + 5 * a * s / h)
    if mode == "selective":
        return sbh * 34
    if mode == "full":
        return sbh * 2
    raise ValueError(mode)


def main():
    P = {k: params(*v) for k, v in MODELS.items()}

    print("1. Parameters (from shapes)")
    for k, n in P.items():
        print(f"   Llama 3.1 {k:>4}: {n:,} params  ({n / 1e9:.2f} B)")

    print("\n2. Training state (no activations), GB, by convention")
    print(f"   {'convention':40s} {'B/param':>7s} " + " ".join(f"{k:>9s}" for k in P))
    for lab, w, g, m32, mv, acc in CONVENTIONS:
        bpp = w + g + m32 + mv + acc
        print(f"   {lab:40s} {bpp:7d} " + " ".join(f"{bpp * n / GB:9.1f}" for n in P.values()))
    n8 = P["8B"]
    print(f"   8B breakdown at 16 B/param: weights {2 * n8 / GB:.2f} + grads {2 * n8 / GB:.2f}"
          f" + master {4 * n8 / GB:.2f} + m {4 * n8 / GB:.2f} + v {4 * n8 / GB:.2f} = {16 * n8 / GB:.1f} GB")
    print(f"   + FP32 grad accumulator {4 * n8 / GB:.2f} -> {20 * n8 / GB:.1f} GB")

    print("\n   Playbook table check (nominal sizes x 16 / x 20):")
    for n in (1e9, 7e9, 70e9, 405e9):
        print(f"   {n / 1e9:>5.0f}B: {16 * n / GB:7.0f} GB / {20 * n / GB:7.0f} GB")

    print("\n3. Activations, Korthikanti Eq. 1 (bytes per layer = sbh(34 + 5as/h))")
    print(f"   GPT-3 check: a=96, s=2048, h=12288 -> 5as/h = {5 * 96 * 2048 / 12288:.0f}")
    L, h, F, a, K, hd, V = MODELS["8B"]
    s, b = 8192, 1
    sbh = s * b * h
    print(f"   Llama 3.1 8B, s={s}, b={b}: sbh = {sbh:,}; 5as/h = {5 * a * s / h:.0f};"
          f" 34 + 5as/h = {34 + 5 * a * s / h:.0f}")
    for mode, desc in (("none", "no recompute, scores stored"),
                       ("selective", "selective / FlashAttention"),
                       ("full", "full recompute (layer inputs)")):
        per = act_per_layer(s, b, h, a, mode)
        print(f"   {desc:32s} {per / GB:7.2f} GB/layer x {L} = {per * L / GB:7.1f} GB")

    print("\n   8B activations vs sequence length (b = 1), GB for all 32 layers")
    print(f"   {'seq':>7s} {'5as/h':>6s} {'none':>8s} {'selective':>10s} {'full':>7s}")
    for s2 in (2048, 4096, 8192, 16384, 32768):
        print(f"   {s2:7d} {5 * a * s2 / h:6.0f} " + " ".join(
            f"{act_per_layer(s2, 1, h, a, m) * L / GB:{w}.1f}"
            for m, w in (("none", 8), ("selective", 10), ("full", 7))))
    print("   (linear in micro-batch b: multiply any column by b)")

    print("\n4. FLOPs per step: 6 N per token (2 N forward + 4 N backward)")
    toks = 8192
    fwd, bwd = 2 * n8 * toks, 4 * n8 * toks
    print(f"   8B, one 8,192-token sequence: forward {fwd:.3g} + backward {bwd:.3g} = {fwd + bwd:.3g} FLOPs")
    t = (fwd + bwd) / (MFU * H100_BF16)
    print(f"   at {MFU:.0%} MFU on one H100 ({H100_BF16 / 1e12:.0f} TF dense): {t:.2f} s"
          f" (forward {fwd / (MFU * H100_BF16):.2f} s, backward {bwd / (MFU * H100_BF16):.2f} s)")

    print("\n5. Does it fit one 80 GB H100? (16 B/param, s = 8192, micro-batch 1)")
    for k, n in P.items():
        L2, h2, F2, a2, *_ = MODELS[k]
        state = 16 * n
        sel = act_per_layer(8192, 1, h2, a2, "selective") * L2
        tot = state + sel
        print(f"   {k:>4}: state {state / GB:8.1f} + activations (selective) {sel / GB:6.1f}"
              f" = {tot / GB:8.1f} GB -> {tot / H100_HBM:6.1f}x one H100")

    print("\n6. Serving the same 8B model (BF16): weights + KV cache")
    kv_tok = 2 * L * K * hd * 2
    kv = kv_tok * 8192
    print(f"   KV per token {kv_tok:,} B ({kv_tok / 1024:.0f} KiB); one 8,192-token sequence {kv / GB:.2f} GB")
    serve = 2 * n8 + kv
    train = 16 * n8 + act_per_layer(8192, 1, h, a, "selective") * L
    print(f"   serving {2 * n8 / GB:.2f} + {kv / GB:.2f} = {serve / GB:.2f} GB"
          f"  vs training {train / GB:.1f} GB  ({train / serve:.1f}x)")

    print("\n7. Gradient accumulation: global batch = micro-batch x grad_acc x DP")
    gbs_tok, seq = 2**22, 4096          # "4M tokens" = 4 x 2^20, as the Playbook's 1,024 implies
    samples = gbs_tok / seq
    print(f"   Playbook: 4M tokens / {seq} = {samples:.0f} sequences;"
          f" mbs 2 x 128 GPUs -> grad_acc {samples / (2 * 128):.0f}; 512 GPUs -> {samples / (2 * 512):.0f}")
    print(f"   Llama 3 405B (Table 4): 32 sequences per DP group x DP 64 = {32 * 64} sequences"
          f" x 8192 = {32 * 64 * 8192:,} tokens (the paper's 16M)")
    one = act_per_layer(8192, 1, h, a, "selective") * L
    print(f"   8B on one GPU, micro-batch 1 at 8k: activations {one / GB:.1f} GB whatever grad_acc is;"
          f" micro-batch 8 would need {8 * one / GB:.1f} GB")


if __name__ == "__main__":
    main()

# Try this:
# 1. Set s, b = 32768, 1 in section 3: the 5as/h term (1,280) swamps the 34, which is why
#    long-context training depends on FlashAttention-style recompute and context parallelism.
# 2. Add ("FP32 everything", 4, 4, 0, 8, 0) to CONVENTIONS: also 16 B/param, which is the
#    Playbook's point that mixed precision does not save state memory.
# 3. Change MFU to 0.30 or 0.50 and watch the per-sequence step time move.
