"""Every position is a training example: input/target pairs, the sliding window, and the loss.
Card: https://seyonv.github.io/explainers/build-an-llm/training-pairs.html
Run:  python training-pairs.py   (needs torch + transformers; see common.py)

Prepares you for Raschka ch 2.6 (the sliding-window data loader) and 5.1 (the loss on text).
  1. tokenize a short text; build input/target pairs by shifting one token
  2. slide a window over it: how many chunks at context 8 with stride 8 vs stride 4, and the batch shape
  3. one forward pass on a chunk gives a loss at every position; check it matches n separate passes
  4. the same per-position loss and perplexity on our running question plus its answer
"""
import math, time
import torch
import torch.nn.functional as F
from common import load

tok, model = load()

# A few sentences of our own (not from the book), tokenized with Qwen's tokenizer.
TEXT = ("The baker opened the shop at six. She sold bread, coffee and small lemon cakes. "
        "By noon the cakes were gone, and the bread was almost gone too.")
ids = tok(TEXT)["input_ids"]
N = len(ids)
show = lambda xs: [tok.decode([i]) for i in xs]

# ---------------------------------------------------------------- 1. the pair
print("== 1. tokens and the shifted pair ==")
print(f"{len(TEXT)} characters -> {N} tokens")
CTX = 8                                   # context length: how many tokens the model reads per example
x, y = ids[0:CTX], ids[1:CTX + 1]         # inputs = tokens[0:n], targets = tokens[1:n+1]
print("inputs :", show(x))
print("targets:", show(y))
for t in range(CTX):                      # the n examples hidden inside one pair
    print(f"  reads {''.join(show(x[:t + 1]))!r:45} -> should say {tok.decode([y[t]])!r}")

# ---------------------------------------------------------------- 2. the sliding window
print("\n== 2. sliding window over the token stream ==")


def windows(ids, ctx, stride):
    """Start a chunk every `stride` tokens; each needs ctx inputs plus 1 more token for the last target."""
    starts = list(range(0, len(ids) - ctx, stride))
    X = torch.tensor([ids[s:s + ctx] for s in starts])
    Y = torch.tensor([ids[s + 1:s + ctx + 1] for s in starts])
    return starts, X, Y


for stride in (8, 4):
    starts, X, Y = windows(ids, CTX, stride)
    distinct = len({s + k for s in starts for k in range(1, CTX + 1)})
    print(f"context {CTX}, stride {stride}: starts {starts} -> {len(starts)} chunks, "
          f"{X.numel()} predictions, {distinct} distinct target tokens")
starts, X, Y = windows(ids, CTX, 4)
print("a batch of the first 2 stride-4 chunks has shape", tuple(X[:2].shape), "= (batch, context)")
print("  row 0:", show(X[0].tolist()))
print("  row 1:", show(X[1].tolist()), " <- overlaps row 0 by 4 tokens")


# ---------------------------------------------------------------- 3. one pass, n losses
def per_position_loss(ids):
    """One forward pass over ids; return the loss at each position (predicting the next token)."""
    inp, tgt = torch.tensor([ids[:-1]]), torch.tensor(ids[1:])
    with torch.inference_mode():
        logits = model(inp).logits[0]                       # (positions, vocab): a prediction at every position
    return F.cross_entropy(logits, tgt, reduction="none"), logits


print("\n== 3. one forward pass on the first chunk: a loss at every position ==")
chunk = ids[0:CTX + 1]                                      # 9 tokens -> 8 inputs and 8 targets
loss, logits = per_position_loss(chunk)
for t in range(CTX):
    p = math.exp(-loss[t].item())
    print(f"  pos {t}: reads {t + 1} token(s), target {tok.decode(chunk[t + 1])!r:10} "
          f"p = {p:.4f}  loss = -ln p = {loss[t].item():.2f}")
print(f"mean = training loss for this chunk: {loss.mean().item():.3f}")
with torch.inference_mode():                                # Hugging Face does the shift for you:
    hf = model(torch.tensor([chunk]), labels=torch.tensor([chunk])).loss.item()  # pass all 9, it scores 8
print(f"same number from model(chunk, labels=chunk), which shifts internally: {hf:.3f}")

# The causal mask means position t sees only tokens 0..t, so one pass equals 8 separate passes.
with torch.inference_mode():
    sep = torch.stack([model(torch.tensor([chunk[:t + 1]])).logits[0, -1] for t in range(CTX)])
print(f"max |logit difference| one pass vs 8 prefix passes: {(sep - logits).abs().max().item():.1e}")

print("\n== 3b. the whole text in one pass: does loss fall as context grows? ==")
loss_all, _ = per_position_loss(ids)
for t in range(N - 1):
    print(f"  pos {t:2}: target {tok.decode(ids[t + 1])!r:10} loss = {loss_all[t].item():.2f}")
for a, b in ((0, 8), (8, 16), (16, 24), (24, N - 1)):
    print(f"  mean loss, positions {a}-{b - 1}: {loss_all[a:b].mean().item():.2f}")
print(f"mean over all {N - 1} positions: {loss_all.mean().item():.3f} "
      f"(perplexity {math.exp(loss_all.mean().item()):.1f})")

# The same 32 targets, but cut into a batch of four 8-token chunks (stride 8): every chunk starts from scratch.
starts, X, Y = windows(ids, CTX, 8)
with torch.inference_mode():
    logits = model(X).logits                                # (batch 4, context 8, vocab)
chunked = F.cross_entropy(logits.flatten(0, 1), Y.flatten())  # flatten batch and positions, as the book does
print(f"same 32 targets as a batch {tuple(X.shape)} of separate chunks: mean loss {chunked.item():.3f}")
starts, X, Y = windows(ids, CTX, 1)                         # stride 1: every possible 8-token window
with torch.inference_mode():
    stride1 = F.cross_entropy(model(X).logits.flatten(0, 1), Y.flatten())
print(f"stride 1: batch {tuple(X.shape)}, {X.numel()} predictions, mean loss {stride1.item():.3f}")

# Timing, rough: training scores all positions in one pass; generation must go one token per pass,
# even with a KV cache. Median of 3 runs after a warm-up, CPU, float32.
def one_pass():
    model(torch.tensor([ids[:-1]]))


def token_by_token():                                       # generation-style: 1 new token per pass, cached
    cache = None
    for t in range(N - 1):
        cache = model(torch.tensor([[ids[t]]]), past_key_values=cache, use_cache=True).past_key_values


def median_ms(fn):
    times = []
    for _ in range(4):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return sorted(times[1:])[1] * 1000                      # drop the warm-up, take the middle of 3


with torch.inference_mode():
    a, b = median_ms(one_pass), median_ms(token_by_token)
print(f"time for {N - 1} predictions: 1 pass {a:.0f} ms; {N - 1} cached one-token passes {b:.0f} ms")

# ---------------------------------------------------------------- 4. our running example
print("\n== 4. question + answer, per position ==")
QA = "What is the capital of India? The capital of India is New Delhi."
qa = tok(QA)["input_ids"]
loss_qa, _ = per_position_loss(qa)
for t in range(len(qa) - 1):
    print(f"  pos {t:2}: target {tok.decode(qa[t + 1])!r:10} p = {math.exp(-loss_qa[t].item()):.4f}  "
          f"loss = {loss_qa[t].item():.2f}")
m = loss_qa.mean().item()
q_part, a_part = loss_qa[:6], loss_qa[6:]                   # targets inside the question vs after the '?'
print(f"{len(qa)} tokens -> {len(qa) - 1} predictions; mean loss {m:.3f}, perplexity {math.exp(m):.2f}")
print(f"  question targets (6): mean {q_part.mean().item():.2f}, perplexity {math.exp(q_part.mean().item()):.1f}")
print(f"  answer targets ({len(a_part)}): mean {a_part.mean().item():.2f}, perplexity {math.exp(a_part.mean().item()):.2f}")
print(f"untrained baseline: uniform over the vocab = ln({model.config.vocab_size}) = "
      f"{math.log(model.config.vocab_size):.2f}")
