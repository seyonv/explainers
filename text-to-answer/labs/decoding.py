"""Stops 8-9: the model hands back one score (logit) per vocabulary token; a decoder turns them into one token.
Everything the 'decoding' card shows about Qwen2.5-0.5B is printed by this script.
Run: python decoding.py   (needs torch + transformers; uses labs/common.py)"""
import torch
from common import load, PROMPT

tok, model = load()
show = lambda i: repr(tok.decode([i]))


@torch.no_grad()
def logits_after(text):
    ids = tok(text, return_tensors="pt").input_ids
    return model(ids).logits[0, -1]          # one score per vocabulary row (151,936)


# ---------- 1. Softmax by hand on the step after "...? The capital of India is" ----------
step = PROMPT + " The capital of India is"
z = logits_after(step)
p = torch.softmax(z, -1)                      # softmax over the whole vocabulary
top = torch.topk(z, 8)
print("== softmax by hand (top 8 of", len(z), "logits)")
print("token        logit   logit-max  exp(logit-max)  p (whole vocab)")
for l, i in zip(top.values, top.indices):
    d = (l - top.values[0]).item()
    print(f"{show(i):12} {l.item():6.2f}  {d:8.2f}   {torch.exp(torch.tensor(d)).item():10.4f}     {p[i].item():.4f}")
mass8 = p[top.indices].sum().item()
print(f"top 8 hold {mass8:.3f} of the probability; the other {len(z)-8:,} tokens share {1-mass8:.3f}")
e = torch.exp(top.values - top.values[0])
print("softmax over only these 8 (what you'd get by hand):", [round(x, 3) for x in (e / e.sum()).tolist()])

# ---------- 2. Greedy and temperature on the same logits ----------
print("\n== greedy picks", show(z.argmax()))
print("== temperature: softmax(logits / T)")
for T in [0.01, 0.5, 0.7, 1.0, 1.5]:
    pt = torch.softmax(z / T, -1)
    row = ", ".join(f"{tok.decode([i]).strip() or repr(tok.decode([i]))} {pt[i].item():.3f}" for i in top.indices[:3])
    print(f"T={T:<4}  {row}   outside top 8: {1 - pt[top.indices].sum().item():.4f}")

# ---------- 3. Top-k and top-p: cut the list, renormalise what's left ----------
sp, si = torch.sort(p, descending=True)
print("\n== top-k")
for k in [1, 2, 5]:
    kept = sp[:k] / sp[:k].sum()
    print(f"k={k}: kept {sp[:k].sum().item():.3f} of the mass ->", [f"{tok.decode([i])!r} {q:.3f}" for i, q in zip(si[:k].tolist(), kept.tolist())])
print("== top-p (smallest set whose probabilities add up to at least p)")
cum = torch.cumsum(sp, 0)
for P in [0.8, 0.9, 0.95]:
    n = int((cum < P).sum()) + 1
    print(f"p={P}: keeps {n} tokens, holding {cum[n-1].item():.3f}")

# ---------- 4. Beam search: keep the 4 best partial answers, not just 1 ----------
@torch.no_grad()
def token_logprobs(prefix, cont_ids):
    """log P of each continuation token given everything before it. Their sum is what beam search maximises."""
    ids = tok(prefix, return_tensors="pt").input_ids
    full = torch.cat([ids, torch.tensor([cont_ids])], 1)
    lp = torch.log_softmax(model(full).logits[0, :-1], -1)
    n = ids.shape[1]
    return [lp[n - 1 + j, t].item() for j, t in enumerate(cont_ids)]


def gen(text, **kw):
    x = tok(text, return_tensors="pt")
    out = model.generate(**x, pad_token_id=tok.eos_token_id, **kw)
    return out[:, x.input_ids.shape[1]:]


print("\n== beam search (4 beams) vs greedy, 12 new tokens")
for q in [PROMPT, "What is the capital of Australia?", "What is the largest planet?"]:
    g = gen(q, max_new_tokens=12, do_sample=False)[0].tolist()
    b = gen(q, max_new_tokens=12, do_sample=False, num_beams=4)[0].tolist()
    same = "same" if g == b else "DIFFERENT"
    print(f"{q!r}: {same}")
    for name, s in [("greedy", g), ("beam", b)]:
        lps = token_logprobs(q, s)
        print(f"   {name:6} {tok.decode(s)!r}  log P = {sum(lps):.2f}")
        if same == "DIFFERENT":
            print("          per token:", " ".join(f"{tok.decode([t])!r}:{torch.tensor(l).exp().item():.3f}" for t, l in zip(s, lps)))
q = "What is the largest planet?"
pq = torch.softmax(logits_after(q), -1)
print("first step there:", [f"{show(i)} {pq[i].item():.3f}" for i in torch.topk(pq, 4).indices])
pq = torch.softmax(logits_after(q + " - The answer to this question is:"), -1)
print("where greedy names the planet:", [f"{show(i)} {pq[i].item():.3f}" for i in torch.topk(pq, 4).indices])

# ---------- 5. Long greedy text: does it loop? (end-of-text blocked so it keeps going) ----------
print("\n== greedy, end-of-text blocked")
g = gen(PROMPT, max_new_tokens=400, min_new_tokens=400, do_sample=False)[0].tolist()
print("first 60 tokens:", repr(tok.decode(g[:60])))


def loop_start(ids, min_period=8):
    """Earliest position from which the rest of the text is one block repeated (at least twice)."""
    best = None
    for L in range(min_period, len(ids) // 2):
        i = len(ids) - L
        while i > 0 and ids[i - 1] == ids[i - 1 + L]:
            i -= 1
        if len(ids) - i >= 2 * L and (best is None or i < best[0]):
            best = (i, L)
    return best


start, period = loop_start(g)
print(f"from token {start} of 400 it repeats a {period}-token block:", repr(tok.decode(g[start:start + period])))

# ---------- 6. Sampling: pure (T=1, nothing cut) vs top-p 0.9 + T 0.7, same seeds ----------
for name, kw in [("pure sampling", dict(temperature=1.0, top_p=1.0, top_k=0)),
                 ("top-p 0.9, T 0.7", dict(temperature=0.7, top_p=0.9, top_k=0))]:
    print(f"\n== {name}, 20 tokens")
    for seed in range(4):
        torch.manual_seed(seed)
        print(seed, repr(tok.decode(gen(PROMPT, max_new_tokens=20, do_sample=True, **kw)[0])))
