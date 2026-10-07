"""The capstone of "The math of an LLM": one full training step of the tiny model, as an answer key.

It prints, in the order you would work on paper:
  1. the forward pass: 14 named stops (Q, K, V count as one), each with its shape and one check number
  2. the loss
  3. the backward pass, in reverse, one check number per gradient
  4. the update: one SGD step and one Adam step, and the loss after each
  5. the PyTorch check
  6. "train more": STEPS Adam steps, then the loss on the training text and on a text it never saw

Every number comes from labs/tiny.py, which this script imports. Nothing here is new math.

Run:  python labs/one-training-step.py
Needs: numpy, torch."""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tiny  # noqa: E402

STEPS = 1        # Adam steps in part 6. The "Change it" box asks you to try 30.

W = tiny.W
v = tiny.forward(W)
g, d = tiny.backward(W, v)
shape = lambda a: " x ".join(str(n) for n in np.shape(a))

print("== 1. forward: follow row 2 (the token 'sat', which has seen all three tokens)")
forward = [  # (name, the array, the cell to check)
    ("x0", v["x0"], (2, 0)), ("a", v["a"], (2, 0)),
    ("Q", v["Q"], (2, 0)), ("K", v["K"], (2, 0)), ("V", v["V"], (2, 0)),
    ("S", v["S"], (2, 1)), ("A", v["A"], (2, 1)), ("C", v["C"], (2, 0)),
    ("o", v["o"], (2, 1)), ("x1", v["x1"], (2, 0)), ("u", v["u"], (2, 0)),
    ("f", v["f"], (2, 3)), ("x2", v["x2"], (2, 1)), ("h", v["h"], (2, 1)),
    ("z", v["z"], (2, 4)), ("p", v["p"], (2, 3)),
]
for name, a, idx in forward:
    print(f"{name:>3}  {shape(a):>5}   {name}{list(idx)} = {a[idx]:+.4f}")

print("\n== 2. loss")
print("per position", np.round(v["nll"], 4), f"  mean L = {v['loss']:.4f}")

print("\n== 3. backward, in reverse (each dX has the same shape as X)")
backward = [
    ("dz", d["z"], (2, 3)), ("dWout", g["Wout"], (1, 3)), ("dh", d["h"], (2, 0)),
    ("dx2", d["x2"], (2, 0)), ("df", d["f"], (2, 3)), ("dW2", g["W2"], (5, 3)),
    ("dgelu", d["gu"], (2, 0)), ("du", d["u"], (2, 0)), ("dW1", g["W1"], (0, 0)),
    ("dbn", d["bn"], (2, 2)), ("dx1", d["x1"], (2, 0)), ("do", d["o"], (2, 0)),
    ("dWo", g["Wo"], (0, 0)), ("dC", d["C"], (2, 1)), ("dA", d["A"], (2, 0)),
    ("dV", d["V"], (1, 1)), ("dS", d["S"], (2, 1)), ("dQ", d["Q"], (2, 3)),
    ("dK", d["K"], (0, 0)), ("dWq", g["Wq"], (0, 3)), ("da", d["a"], (1, 3)),
    ("dx0", d["x0"], (2, 0)), ("dE", g["E"], (2, 0)),
]
for name, a, idx in backward:
    print(f"{name:>6}  {shape(a):>5}   {name}{list(idx)} = {a[idx]:+.4f}")
print("dE rows 3-4 ('on', 'mat', never looked up):", g["E"][3:].ravel().tolist())

print("\n== 4. update")
W_sgd = {k: W[k] - 1.0 * g[k] for k in W}           # SGD: w <- w - lr * grad, lr = 1
W_adam = tiny.adam_step(W, g)                        # Adam's first step, lr = 0.1
for k, idx in [("Wout", (1, 3)), ("W1", (0, 0))]:
    print(f"{k}{list(idx)}: start {W[k][idx]:+.4f}  grad {g[k][idx]:+.4f}  "
          f"SGD {W_sgd[k][idx]:+.4f}  Adam {W_adam[k][idx]:+.4f}")
for name, Wn in [("SGD  lr 1.0", W_sgd), ("Adam lr 0.1", W_adam)]:
    vn = tiny.forward(Wn)
    print(f"{name}: loss {v['loss']:.4f} -> {vn['loss']:.4f}   per position {np.round(vn['nll'], 4)}")

print("\n== 5. the PyTorch check")
t_loss, t_grads = tiny.torch_check(W)
worst = max(np.abs(g[k] - t_grads[k]).max() for k in t_grads)
print(f"loss numpy {v['loss']:.6f}  torch {t_loss:.6f}   "
      f"largest gradient difference over {sum(a.size for a in W.values())} weights: {worst:.1e}")

print(f"\n== 6. train more: {STEPS} Adam step(s) at lr 0.1 (the same loop as labs/why.py)")
Wt = {k: a.copy() for k, a in W.items()}
m = {k: np.zeros_like(a) for k, a in Wt.items()}
s2 = {k: np.zeros_like(a) for k, a in Wt.items()}
for t in range(1, STEPS + 1):
    gt, _ = tiny.backward(Wt, tiny.forward(Wt))
    for k in Wt:
        m[k] = 0.9 * m[k] + 0.1 * gt[k]                   # running mean of the gradient
        s2[k] = 0.999 * s2[k] + 0.001 * gt[k] ** 2        # running mean of its square
        Wt[k] = Wt[k] - 0.1 * (m[k] / (1 - 0.9**t)) / (np.sqrt(s2[k] / (1 - 0.999**t)) + 1e-8)
vt = tiny.forward(Wt)
print(f"training text 'the cat sat' -> 'cat sat on': loss {vt['loss']:.4f}   "
      f"p[target] {np.round(vt['p'][np.arange(3), tiny.targets], 4)}")
print("E rows 3-4 still exactly their random start:", np.array_equal(Wt["E"][3:], W["E"][3:]))
# A text the model never trained on. tiny.forward reads tiny.inputs and tiny.targets, so swap them.
tiny.inputs, tiny.targets = np.array([1, 2, 3]), np.array([2, 3, 4])    # 'cat sat on' -> 'sat on mat'
print(f"unseen text   'cat sat on' -> 'sat on mat': loss at start {tiny.forward(W)['loss']:.4f}, "
      f"after training {tiny.forward(Wt)['loss']:.4f}")
