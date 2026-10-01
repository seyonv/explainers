"""Stop 4: attention can't tell first from last.

We rebuild layer 0's attention of Qwen2.5-0.5B by hand from the model's own weights, then feed it
the 7 token vectors of "What is the capital of India?" in two orders:
  original:  What  is  the  capital  of  India  ?
  shuffled:  India  of  capital  the  is  What  ?      (the same 7 vectors, reordered)
and compare what comes out, with and without RoPE (the rotation that marks positions) and with and
without the causal mask (the rule that a token may only look at earlier tokens).

Run: HF_HUB_OFFLINE=1 python text-to-answer/labs/positions.py
"""
import math
import torch
from common import load, PROMPT

tok, model = load(eager_attention=True)
cfg = model.config
layer = model.model.layers[0]
att = layer.self_attn
H, KV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
# The RoPE base (config.json's "rope_theta"); newer transformers versions keep it in rope_parameters.
theta = getattr(cfg, "rope_parameters", None) and cfg.rope_parameters["rope_theta"] or cfg.rope_theta
print(f"layer 0: {H} query heads, {KV} key/value heads, head size {D}, rope_theta = {theta:,.0f}")

ids = tok(PROMPT)["input_ids"]
words = [tok.decode([i]) for i in ids]
print("tokens:", words)

# One rotation speed per pair of numbers in a head: theta_i = rope_theta^(-2i/D), i = 0..D/2-1.
inv_freq = theta ** (-torch.arange(0, D, 2).float() / D)          # 32 speeds, radians per position


def rope(x, pos):
    """Turn every pair of numbers in x by angle pos * speed. x: (tokens, heads, D).
    Qwen (like Llama) pairs dimension i with i + D/2, not with i + 1. It's the same idea with
    the dimensions listed in a different order."""
    ang = pos[:, None].float() * inv_freq[None, :]                 # (tokens, 32)
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]        # (tokens, 1, D)
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    rotated_half = torch.cat([-x2, x1], -1)
    return x * cos + rotated_half * sin


@torch.no_grad()
def attention(token_ids, use_rope, causal):
    """Layer 0's attention for one sequence, written out step by step."""
    x = model.model.embed_tokens(torch.tensor(token_ids))          # (T, 896): one row per token
    h = layer.input_layernorm(x)                                    # RMSNorm before attention
    T = len(token_ids)
    q = att.q_proj(h).view(T, H, D)                                 # q_proj, k_proj, v_proj all have biases
    k = att.k_proj(h).view(T, KV, D)
    v = att.v_proj(h).view(T, KV, D)
    if use_rope:
        pos = torch.arange(T)                                       # slot 0, 1, 2, ...
        q, k = rope(q, pos), rope(k, pos)                           # rotate queries and keys, never values
    k = k.repeat_interleave(H // KV, dim=1)                         # GQA: 7 query heads share each KV head
    v = v.repeat_interleave(H // KV, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(D)      # (heads, T, T)
    if causal:
        scores = scores.masked_fill(torch.ones(T, T).triu(1).bool(), float("-inf"))
    weights = scores.softmax(-1)
    out = torch.einsum("hqk,khd->qhd", weights, v).reshape(T, H * D)
    return att.o_proj(out), weights


# 0. Check the hand-built version against the real model (RoPE on, causal mask on).
captured = {}


def keep_output(module, inputs, output):
    captured["out"] = output[0][0]                                  # (7, 896) for our one sequence


hook = att.register_forward_hook(keep_output)
with torch.no_grad():
    model(torch.tensor([ids]))
hook.remove()
mine, _ = attention(ids, use_rope=True, causal=True)
print(f"\n0. hand-built vs real layer-0 attention: max |difference| = {(mine - captured['out']).abs().max():.1e}"
      f"   (outputs are up to {captured['out'].abs().max():.2f} in size)")

# 1. Shuffle the 7 vectors and compare, under four settings.
perm = [5, 4, 3, 2, 1, 0, 6]                                       # India of capital the is What ?
shuf = [ids[p] for p in perm]
print("shuffled:", [tok.decode([i]) for i in shuf])
print("\n1. after shuffling, does each token get the same output it got before?")
print(f"   {'setting':34} {'max |difference|':>17}   {'tokens unchanged (of 7)':>23}")
for use_rope, causal, name in [(False, False, "no positions, no mask"),
                               (True, False, "RoPE, no mask"),
                               (False, True, "no positions, causal mask"),
                               (True, True, "RoPE + causal mask (the real model)")]:
    a, _ = attention(ids, use_rope, causal)
    b, _ = attention(shuf, use_rope, causal)
    diff = (b - a[perm]).abs()                                      # line each token up with itself
    same = [words[p] for j, p in enumerate(perm) if diff[j].max() < 1e-4]
    print(f"   {name:34} {diff.max().item():17.2e}   {len(same):>23}  {same if 0 < len(same) < 7 else ''}")

# 2. Follow one token: how much attention does " India" pay to " capital" (head 0)?
print("\n2. head 0: how much of ' India' attention goes to ' capital'  (no mask)")
for use_rope in (False, True):
    _, wa = attention(ids, use_rope, causal=False)
    _, wb = attention(shuf, use_rope, causal=False)
    # India is slot 5 originally, slot 0 after shuffling; capital is slot 3, then slot 2
    print(f"   {'RoPE' if use_rope else 'no positions':13} original order {wa[0, 5, 3]:.4f}   shuffled {wb[0, 0, 2]:.4f}")

# 3. RoPE by hand on one pair of numbers: the score depends only on the gap m - n.
i = 2                                                               # pair 2: dimensions 2 and 34
speed = inv_freq[i].item()
with torch.no_grad():
    h = layer.input_layernorm(model.model.embed_tokens(torch.tensor(ids)))
    q_pair = att.q_proj(h).view(7, H, D)[5, 0, [i, i + D // 2]]   # ' India' as a query, head 0
    k_pair = att.k_proj(h).view(7, KV, D)[3, 0, [i, i + D // 2]]   # ' capital' as a key, its KV head


def turn(vec, angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c])


print(f"\n3. one pair (dims {i} and {i + D // 2}), speed {speed:.4f} rad = {math.degrees(speed):.2f} degrees per position")
print(f"   query pair of ' India'  = ({q_pair[0]:.3f}, {q_pair[1]:.3f})")
print(f"   key pair of ' capital'  = ({k_pair[0]:.3f}, {k_pair[1]:.3f})")
print(f"   no rotation: q . k = {float(q_pair @ k_pair):.3f}")
for m, n in [(5, 3), (105, 103), (3, 1), (5, 4), (3, 5)]:
    qm, kn = turn(q_pair, m * speed), turn(k_pair, n * speed)
    print(f"   query at m={m:3}, key at n={n:3} (gap {m - n:+d}): q turned {math.degrees(m * speed) % 360:6.1f} deg,"
          f" k turned {math.degrees(n * speed) % 360:6.1f} deg -> q . k = {float(qm @ kn):.3f}")

# 4. How slow is the slowest pair? (one full turn = 2*pi / speed positions)
print(f"\n4. fastest pair: one turn every {2 * math.pi / inv_freq[0]:.1f} positions;"
      f" slowest pair: one turn every {2 * math.pi / inv_freq[-1]:,.0f} positions")
print(f"   max_position_embeddings in the config: {cfg.max_position_embeddings:,}")
