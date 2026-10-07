"""Card 11, "The loss is surprise at the right answer": the measurements behind it.

Uses the tiny model from tiny.py (it never changes it) and prints:
  1. the loss at each of the 3 positions, and the mean
  2. why -log: products become sums, 1 - p gives almost no signal when the model is confidently wrong
  3. what the number means: uniform guessing, perplexity, and the same baseline at real scale

Run:  python labs/logits-and-loss.py      (needs numpy; run from the llm-math folder or anywhere)"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tiny  # noqa: E402

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
v = tiny.forward(tiny.W)
p, z, T = v["p"], v["z"], tiny.T
tgt = tiny.targets

print("== the loss, position by position")
for t in range(T):
    word = tiny.VOCAB[tgt[t]]
    print(f"position {t}: target {word:4s} p = {p[t, tgt[t]]:.4f}   -ln p = {v['nll'][t]:.4f}"
          f"   model's top pick {tiny.VOCAB[p[t].argmax()]} ({p[t].max():.4f})")
print(f"mean over 3 positions L = {v['loss']:.4f}")

# The same number without ever forming p: -log softmax(z)[t] = logsumexp(z) - z[t].
lse = np.log(np.exp(z[2] - z[2].max()).sum()) + z[2].max()
print(f"\nposition 2 in log space: logsumexp(z) = {lse:.4f}, z[on] = {z[2, 3]:.4f}, difference = {lse - z[2, 3]:.4f}")

print("\n== by hand: -ln of the rounded probability")
print(f"-ln(0.0673) = {-np.log(0.0673):.4f}   (the unrounded p gives {v['nll'][2]:.4f})")

print("\n== why -log (1): a product of probabilities becomes a sum")
probs = p[np.arange(T), tgt]
prod = probs.prod()
print(f"p(cat) x p(sat) x p(on) = {prod:.6f}   -ln of the product = {-np.log(prod):.4f}"
      f"   sum of the three -ln p = {v['nll'].sum():.4f}   / 3 = {v['nll'].sum() / 3:.4f}")
for n in (100, 1024):
    print(f"{n} tokens at p = 0.1 each: product in float64 = {np.float64(0.1) ** n:.3e}"
          f"   sum of -ln p = {n * -np.log(0.1):.1f}")

print("\n== why -log (2): how hard each loss pushes on the right logit, position 2")
pt = p[2, 3]
print(f"cross-entropy  -ln p : loss {-np.log(pt):.4f}  dL/dz[on] = p - 1      = {pt - 1:.4f}")
print(f"plain miss     1 - p : loss {1 - pt:.4f}  dL/dz[on] = -p (1 - p) = {-pt * (1 - pt):.4f}")
for q in (0.5, 0.1, 0.01, 0.001):
    print(f"  p = {q:<6}: -ln p = {-np.log(q):.4f}   1 - p = {1 - q:.4f}")
print(f"  p = 0.99  : -ln p = {-np.log(0.99):.4f}")

print("\n== what the number means")
print(f"uniform guessing over 5 words: ln 5 = {np.log(5):.4f}")
print(f"perplexity e^L = {np.exp(v['loss']):.4f}   (guessing: e^ln5 = 5)")
for t in range(T):
    print(f"  position {t}: e^(-ln p) = 1/p = {1 / probs[t]:.4f}")
W0 = dict(tiny.W, Wout=np.zeros((tiny.D, tiny.V)))
print(f"set Wout to all zeros: every logit 0, loss = {tiny.forward(W0)['loss']:.4f}")

print("\n== at real scale: the same baseline and the output head's size")
for name, d, vocab in (("GPT-2 124M", 768, 50257), ("Qwen2.5-0.5B", 896, 151936)):
    print(f"{name:13s}: Wout {d} x {vocab:,} = {d * vocab:,} weights   uniform loss ln {vocab:,} = {np.log(vocab):.4f}")
