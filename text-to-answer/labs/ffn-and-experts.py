"""Stop 6: the feed-forward network (FFN). Where Qwen2.5-0.5B keeps its weights, where it spends its
arithmetic, and what one FFN does to one token. Then: what the same model would look like as a mixture
of experts (an illustrative what-if; Qwen2.5-0.5B is dense).
  HF_HUB_OFFLINE=1 python ffn-and-experts.py"""
import torch
from common import load, PROMPT

tok, model = load()
cfg = model.config
d, f, n_layers = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers  # 896, 4864, 24

# 1. Count the weights in each part of the model.
parts = {"embedding": 0, "attention": 0, "FFN": 0, "norms": 0}
for name, p in model.named_parameters():  # lm_head is tied to the embedding, so it isn't listed twice
    if "embed_tokens" in name: parts["embedding"] += p.numel()
    elif "self_attn" in name: parts["attention"] += p.numel()
    elif "mlp" in name: parts["FFN"] += p.numel()
    else: parts["norms"] += p.numel()
total = sum(parts.values())
print(f"Parameters: {total:,}")
for k, v in parts.items():
    print(f"  {k:10s} {v:>12,}  {100 * v / total:5.1f}%")

# 2. Arithmetic per token. A matrix multiply costs 2 FLOPs (one multiply, one add) per weight.
#    Looking up the embedding row costs nothing; the output layer (the same table, used as the
#    unembedding) costs 2 x 151,936 x 896. Attention scores grow with how many tokens came before.
attn_proj = 2 * sum(p.numel() for n, p in model.named_parameters() if "self_attn" in n and n.endswith("weight"))
ffn = n_layers * 2 * 3 * d * f
lm_head = 2 * cfg.vocab_size * d
def attn_scores(ctx):  # q.k for every earlier token, then the weighted sum of values: 2 x ctx x 896 each, per layer
    return n_layers * 2 * 2 * ctx * d
n_ids = len(tok(PROMPT).input_ids)
flops = {"attention projections": attn_proj, f"attention scores ({n_ids} tokens back)": attn_scores(n_ids),
         "FFN": ffn, "output layer": lm_head}
ftot = sum(flops.values())
print(f"\nFLOPs for one token: {ftot / 1e6:,.0f} M")
for k, v in flops.items():
    print(f"  {k:32s} {v / 1e6:8,.1f} M  {100 * v / ftot:5.1f}%")
print(f"Attention scores match the FFN's FLOPs once a token looks back {ffn / attn_scores(1):,.0f} tokens")

# 3. Watch one FFN work on one token. Hooks catch what goes in and out of every layer's FFN.
ids = tok(PROMPT, return_tensors="pt").input_ids
caught, hooks = {}, []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.mlp.register_forward_hook(lambda m, inp, out, i=i: caught.__setitem__(("ffn", i), (inp[0][0], out[0]))))
    hooks.append(layer.mlp.down_proj.register_forward_hook(lambda m, inp, out, i=i: caught.__setitem__(("hidden", i), inp[0][0])))
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
for hk in hooks: hk.remove()

pos, L = 5, n_layers - 1  # " India", last layer
mlp = model.model.layers[L].mlp
x_in, y = caught[("ffn", L)]
with torch.no_grad():
    gate, up = mlp.gate_proj(x_in[pos]), mlp.up_proj(x_in[pos])
    h = torch.nn.functional.silu(gate) * up  # SwiGLU: the gate decides how much of "up" gets through
    y_alone = mlp.down_proj(h)
print(f"\nToken {tok.decode(ids[0, pos])!r}, layer {L}: {d} -> gate {tuple(gate.shape)} and up {tuple(up.shape)}"
      f" -> SwiGLU {tuple(h.shape)} -> down {tuple(y_alone.shape)}")
# Same weights at every position, and no mixing between tokens: running " India" alone gives the same answer.
print(f"FFN on ' India' alone vs inside the sentence: max difference {(y_alone - y[pos]).abs().max():.1e}"
      f" (rounding noise; the values are around {y[pos].abs().max():.0f})")

hidden = caught[("hidden", L)][pos]
mag = hidden.abs().sort(descending=True).values
share = (mag ** 2).cumsum(0) / (mag ** 2).sum()  # share of the total squared activity in the top k units
k50, k90 = int((share < 0.5).sum()) + 1, int((share < 0.9).sum()) + 1
print(f"Of {f:,} hidden units: the top {k50} carry half the activity, the top {k90} carry 90%")
print(f"  biggest |value| {mag[0]:.1f}, median |value| {mag[f // 2]:.2f}")

# How big is the FFN's push compared with the vector it is added to (the residual stream)?
print("\nFFN update size / residual size for ' India', by layer:")
ratios = []
for i in range(n_layers):
    _, y_i = caught[("ffn", i)]
    before = out.hidden_states[i + 1][0, pos] - y_i[pos]  # residual after attention, before the FFN adds in
    ratios.append((y_i[pos].norm() / before.norm()).item())
    if i == L:
        print(f"  layer {L}: residual length {before.norm():.1f}, FFN update length {y_i[pos].norm():.1f}")
print("  " + " ".join(f"{r:.2f}" for r in ratios))
mid = ratios[1:]
print(f"  layers 1-23: {min(mid):.2f} to {max(mid):.2f}; layer 0 (residual is still the raw embedding): {ratios[0]:.1f}")

# 4. Illustrative what-if: turn each of the 24 FFNs into 8 copies (experts) plus a router, top-1 routing.
E = 8
ffn_one_layer = 3 * d * f
router = d * E  # one 896 -> 8 linear layer per block picks the expert
moe_total = total - parts["FFN"] + n_layers * (E * ffn_one_layer + router)
moe_active = total + n_layers * router  # each token still runs exactly one FFN per layer
print(f"\nIllustrative MoE, {E} experts, top-1 (one FFN = {ffn_one_layer:,} weights; {E} of them = {E * ffn_one_layer:,} per layer):")
print(f"  total params  {moe_total:>14,}  ({moe_total / total:.2f}x the dense model)")
print(f"  active/token  {moe_active:>14,}  ({moe_active / total:.4f}x)")
for name, bytes_ in [("float32", 4), ("bfloat16", 2)]:
    print(f"  weights in {name}: dense {total * bytes_ / 1e9:.2f} GB, MoE {moe_total * bytes_ / 1e9:.2f} GB")
