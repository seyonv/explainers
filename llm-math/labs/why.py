"""The "why" measurements for "The math of an LLM": small experiments that show why each equation
has the form it has. Each section prints the numbers one card uses.

Run:  python labs/why.py
Needs: numpy, torch."""
import numpy as np
import torch

import tiny

np.seterr(all="ignore")                   # the lr = 10 and 100 runs overflow on purpose
rng = np.random.default_rng(0)
torch.manual_seed(0)


def section(name):
    print(f"\n== {name}")


# ---------------------------------------------------------------- derivatives-and-chain-rule
section("derivatives: nudge one weight, watch the loss")
W, v0 = tiny.W, tiny.forward(tiny.W)
g, _ = tiny.backward(W, v0)
for key, (i, j) in [("Wout", (0, 1)), ("W1", (0, 0))]:
    for eps in (0.1, 0.01, 0.001):
        W2 = {k: a.copy() for k, a in W.items()}
        W2[key][i, j] += eps
        dl = tiny.forward(W2)["loss"] - v0["loss"]
        print(f"{key}[{i},{j}] += {eps:<6} loss change {dl:+.6f}   change/nudge {dl/eps:+.4f}   gradient {g[key][i, j]:+.4f}")

# ---------------------------------------------------------------- softmax
section("softmax: overflow and the max trick")
z = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(z) / np.exp(z).sum()
print(f"logits {z}  naive exp(1000) -> {np.exp(z[0])}  naive softmax {naive}")
print(f"subtract max -> {z - z.max()}  softmax {tiny.softmax(z)}")
print(f"float32 exp overflows above {np.log(np.finfo(np.float32).max):.2f}")
row = v0["z"][2]
for T in (0.5, 1.0, 2.0):
    print(f"temperature {T}: softmax(z/T) of last position's logits = {tiny.softmax(row / T)}")
print(f"why exp, not just divide by the sum: logits {row} -> z/sum(z) = {row / row.sum()}  (negative 'probabilities')")

# ---------------------------------------------------------------- attention-scores: why divide by sqrt(d)
section("attention scores: the spread of q.k grows like sqrt(d)")
for d in (4, 64, 896):
    q, k = rng.standard_normal((100_000, d)), rng.standard_normal((100_000, d))
    dots = (q * k).sum(1)
    print(f"d = {d:>3}: std of q.k = {dots.std():7.3f}   sqrt(d) = {np.sqrt(d):7.3f}   std after / sqrt(d) = {(dots / np.sqrt(d)).std():.3f}")
d, n = 64, 10
peak_raw, peak_scaled = [], []
for _ in range(2000):
    q, K = rng.standard_normal(d), rng.standard_normal((n, d))
    s = K @ q
    peak_raw.append(tiny.softmax(s).max()); peak_scaled.append(tiny.softmax(s / np.sqrt(d)).max())
print(f"d = 64, 10 keys: average top attention weight, unscaled {np.mean(peak_raw):.3f}  scaled {np.mean(peak_scaled):.3f}")
s = rng.standard_normal((n, d)) @ rng.standard_normal(d)
A = tiny.softmax(s)
print(f"one unscaled row: max weight {A.max():.4f}; softmax slope at the others  A(1-A) max = {(A * (1 - A)).max():.4f}")

# ---------------------------------------------------------------- ffn: why a nonlinearity
section("ffn: two linear layers collapse into one")
x = v0["bn"]
two = (x @ W["W1"]) @ W["W2"]
one = x @ (W["W1"] @ W["W2"])
print(f"W1 @ W2 is one 4x4 matrix:\n{W['W1'] @ W['W2']}")
print(f"largest difference, two layers vs one merged matrix: {np.abs(two - one).max():.1e}")
with_gelu = tiny.gelu(x @ W["W1"]) @ W["W2"]
print(f"with GELU between them, difference from the merged matrix: {np.abs(with_gelu - one).max():.4f}")
for u in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0):
    print(f"GELU({u:+.1f}) = {tiny.gelu(np.array(u)):+.4f}   slope {tiny.gelu_back(np.array(u)):+.4f}   ReLU {max(u, 0):+.1f}")

# ---------------------------------------------------------------- norm-and-residual: init scale and depth
section("why normalise: activations through 24 random layers, d = 896")
d = 896
for std, label in [(1.0, "W ~ N(0, 1)"), (1 / np.sqrt(d), "W ~ N(0, 1/d)")]:
    x = rng.standard_normal(d)
    stds = []
    for _ in range(24):
        x = x @ (rng.standard_normal((d, d)) * std)
        stds.append(x.std())
    print(f"{label:<14} std after layers 1, 2, 3, 24: {stds[0]:.3g}, {stds[1]:.3g}, {stds[2]:.3g}, {stds[-1]:.3g}")

section("why add (residual): gradient reaching the input through a deep stack")
for depth in (4, 24):
    for residual in (False, True):
        torch.manual_seed(1)
        layers = [torch.nn.Linear(64, 64) for _ in range(depth)]
        x = torch.randn(8, 64, requires_grad=True)
        h = x
        for L in layers:
            f = torch.tanh(L(h)) * 0.5
            h = h + f if residual else f
        h.pow(2).mean().backward()
        print(f"depth {depth:>2}, residual {str(residual):<5}: gradient size at the input {x.grad.norm():.2e}")

# ---------------------------------------------------------------- positions-rope
section("RoPE: rotate a pair of numbers by (position x angle)")
theta = 0.5
def rot(vec, pos):
    c, s_ = np.cos(pos * theta), np.sin(pos * theta)
    return np.array([c * vec[0] - s_ * vec[1], s_ * vec[0] + c * vec[1]])
q, k = np.array([1.0, 0.0]), np.array([0.6, 0.8])
for pq, pk in [(0, 0), (1, 0), (3, 2), (7, 6), (2, 0), (5, 3)]:
    print(f"query at {pq}, key at {pk} (distance {pq - pk}): rotated q.k = {rot(q, pq) @ rot(k, pk):+.4f}")
print(f"unrotated q.k = {q @ k:+.4f}")
print("Qwen2.5 base: rope_theta = 1,000,000, head dim 64 -> 32 pairs; pair i turns by position x 1e6^(-2i/64)")
for i in (0, 1, 16, 31):
    print(f"  pair {i:>2}: angle per position {1e6 ** (-2 * i / 64):.3g} rad")

# ---------------------------------------------------------------- gradient-descent-and-adam
section("learning rate: plain SGD on the tiny model, 30 steps")
def train(lr, steps, opt="sgd"):
    Wt = {k: a.copy() for k, a in tiny.W.items()}
    m = {k: np.zeros_like(a) for k, a in Wt.items()}; s2 = {k: np.zeros_like(a) for k, a in Wt.items()}
    losses = [tiny.forward(Wt)["loss"]]
    for t in range(1, steps + 1):
        gt, _ = tiny.backward(Wt, tiny.forward(Wt))
        for k in Wt:
            if opt == "sgd":
                Wt[k] = Wt[k] - lr * gt[k]
            else:
                m[k] = 0.9 * m[k] + 0.1 * gt[k]; s2[k] = 0.999 * s2[k] + 0.001 * gt[k] ** 2
                Wt[k] = Wt[k] - lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(s2[k] / (1 - 0.999**t)) + 1e-8)
        losses.append(tiny.forward(Wt)["loss"])
    return losses
for lr in (0.01, 0.1, 1.0, 10.0, 100.0):
    L = train(lr, 30)
    print(f"SGD  lr {lr:>6}: loss at steps 0, 1, 5, 10, 30 = " + ", ".join(f"{L[i]:.4f}" for i in (0, 1, 5, 10, 30)))
for lr in (0.01, 0.1):
    L = train(lr, 30, "adam")
    print(f"Adam lr {lr:>6}: loss at steps 0, 1, 5, 10, 30 = " + ", ".join(f"{L[i]:.4f}" for i in (0, 1, 5, 10, 30)))

section("Adam's first step moves every weight by about lr, whatever the gradient's size")
g, _ = tiny.backward(tiny.W, v0)
step = {k: tiny.adam_step(tiny.W, g)[k] - tiny.W[k] for k in tiny.W}
for key, idx in [("Wout", (1, 3)), ("Wq", (1, 0)), ("dS-small: Wk", (2, 3))]:
    k = key.split(": ")[-1]
    print(f"{k}{list(idx)}: gradient {g[k][idx]:+.5f}   SGD step (lr 1) {-g[k][idx]:+.5f}   Adam step (lr 0.1) {step[k][idx]:+.5f}")
