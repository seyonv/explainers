"""The loop: an LLM writes its answer one token at a time.
Card: https://seyonv.github.io/explainers/text-to-answer/the-loop.html
Run:  python the-loop.py   (needs torch + transformers; see common.py)

Three experiments on Qwen2.5-0.5B (base model, float32, CPU):
  1. greedy generation written out as a plain loop, with each step's probability
  2. timing: the first pass over the question vs each later step, with and without the KV cache
  3. the same question wrapped in different templates: how the first answer token changes
"""
import math, statistics, time
import torch
from common import load, PROMPT

torch.manual_seed(0)
tok, model = load()
EOS = tok.eos_token_id  # <|endoftext|>


def next_probs(ids):
    """One forward pass over the whole sequence; return the probabilities for the next token."""
    with torch.inference_mode():
        logits = model(torch.tensor([ids])).logits[0, -1]
    return torch.softmax(logits, -1)


# ---------------------------------------------------------------- 1. the loop
print("== 1. greedy loop ==")
ids = tok(PROMPT)["input_ids"]
probs_chosen = []
for step in range(20):                  # a length limit, in case the end token never comes
    p = next_probs(ids)
    t = int(p.argmax())                 # greedy: take the most likely token
    probs_chosen.append(p[t].item())
    ids.append(t)                       # append it, then run the model again on the longer sequence
    print(f"step {step+1}: {tok.decode([t])!r:18} p = {p[t].item():.3f}")
    if t == EOS:                        # the model chose to stop
        break
product = math.prod(probs_chosen)
logsum = sum(math.log(q) for q in probs_chosen)
print(f"answer: {tok.decode(ids[7:])!r}")
print(f"product of the 8 answer tokens = {math.prod(probs_chosen[:8]):.3f}; with the end token too = {product:.3f}")
print(f"sum of log-probs = {logsum:.3f}; exp(sum) = {math.exp(logsum):.4f}")
print(f"why logs: 0.5 multiplied 1100 times in float64 = {0.5 ** 1100} (underflow); its log is {1100 * math.log(0.5):.1f}")

# ---------------------------------------------------------------- 2. what the loop costs
print("\n== 2. timing: 32 new tokens after the 7-token question (CPU, float32) ==")
q = torch.tensor([tok(PROMPT)["input_ids"]])
NEW, REPEATS = 32, 5


def run_with_cache():
    """Prefill once, then feed only the newest token; the model reuses stored keys/values."""
    with torch.inference_mode():
        t0 = time.perf_counter()
        out = model(q, use_cache=True)
        prefill = time.perf_counter() - t0
        cache, nxt, steps = out.past_key_values, out.logits[0, -1].argmax().view(1, 1), []
        for _ in range(NEW - 1):
            t0 = time.perf_counter()
            out = model(nxt, past_key_values=cache, use_cache=True)
            steps.append(time.perf_counter() - t0)
            cache, nxt = out.past_key_values, out.logits[0, -1].argmax().view(1, 1)
    return prefill, steps, cache


def run_without_cache():
    """No memory between steps: every step re-reads the whole sequence so far."""
    seq, steps = q, []
    with torch.inference_mode():
        for _ in range(NEW):
            t0 = time.perf_counter()
            out = model(seq, use_cache=False)
            steps.append(time.perf_counter() - t0)
            seq = torch.cat([seq, out.logits[0, -1].argmax().view(1, 1)], 1)
    return steps


run_with_cache(); run_without_cache()   # warm-up run, not counted
pre, dec, tot_c, tot_n, first_n, last_n = [], [], [], [], [], []
for r in range(REPEATS):
    p_, s_, cache = run_with_cache()
    pre.append(p_); dec.append(statistics.median(s_)); tot_c.append(p_ + sum(s_))
    n_ = run_without_cache()
    tot_n.append(sum(n_)); first_n.append(n_[0]); last_n.append(n_[-1])
ms = lambda xs: f"{1000 * statistics.median(xs):.0f} ms ({1000 * min(xs):.0f}-{1000 * max(xs):.0f})"
print(f"median of {REPEATS} runs (min-max in brackets):")
print(f"  with cache:    first pass over 7 tokens {ms(pre)}, each later step {ms(dec)}, total {ms(tot_c)}")
print(f"  without cache: step 1 (7 tokens) {ms(first_n)}, step 32 (38 tokens) {ms(last_n)}, total {ms(tot_n)}")

# How big is the cache? Count the stored key/value numbers per token.
k0 = cache.layers[0].keys if hasattr(cache, "layers") else cache[0][0]
per_token = 2 * len(model.model.layers) * k0.shape[1] * k0.shape[3]   # K and V, every layer, every KV head, head dim
print(f"cache holds {per_token:,} numbers per token = {per_token * 4:,} B in float32, {per_token * 2:,} B in bf16")

# ---------------------------------------------------------------- 3. templates
print("\n== 3. the same question, three wrappers ==")
wrappers = {
    "plain": PROMPT,
    "Q/A": f"Q: {PROMPT}\nA:",
    "chat": tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False),
}
for name, text in wrappers.items():
    ids = tok(text)["input_ids"]
    p = next_probs(ids)
    top = torch.topk(p, 4)
    print(f"[{name}] {len(ids)} tokens; first-token top 4:",
          ", ".join(f"{tok.decode([int(i)])!r} {v:.3f}" for v, i in zip(top.values, top.indices)))
    with torch.inference_mode():
        g = model.generate(torch.tensor([ids]), max_new_tokens=20, do_sample=False, pad_token_id=EOS)
    print(f"   greedy: {tok.decode(g[0, len(ids):])!r}")
