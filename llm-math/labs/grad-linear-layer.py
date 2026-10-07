"""Card 13, "Let the shapes write the gradient": the measurements behind it.

Uses the tiny model from tiny.py (it never changes it) and prints:
  1. one entry of dWout, as a sum over the 3 positions
  2. the whole rule, dW = x.T @ dy and dx = dy @ W.T, on the output head, with a nudge check
  3. why only x.T @ dy has the right shape
  4. the same rule for every linear layer in the block, checked against PyTorch autograd
  5. the bias rule, db = dy summed over rows, and PyTorch's transposed weight.grad

Run:  python labs/grad-linear-layer.py      (needs numpy and torch)"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tiny  # noqa: E402

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
W = tiny.W
v = tiny.forward(W)
g, d = tiny.backward(W, v)
h, dz, Wout = v["h"], d["z"], W["Wout"]
words = ["the", "the cat", "the cat sat"]

print("== 1. one entry: dWout[1,3] = sum over t of h[t,1] * dz[t,3]")
total = 0.0
for t in range(tiny.T):
    term = h[t, 1] * dz[t, 3]
    total += term
    print(f"position {t} ({words[t]:11s}): h[{t},1] = {h[t,1]:+.4f}   dz[{t},3] = {dz[t,3]:+.4f}   product {term:+.4f}")
print(f"sum = {total:+.4f}     (tiny.py's dWout[1,3] = {g['Wout'][1, 3]:+.4f})")

print("\n== 2. the whole matrix at once")
dWout = h.T @ dz                      # (4 x 3) @ (3 x 5) = 4 x 5, the shape of Wout
dh = dz @ Wout.T                      # (3 x 5) @ (5 x 4) = 3 x 4, the shape of h
print(f"dWout = h.T @ dz, shape {dWout.shape}, equals tiny.py: {np.allclose(dWout, g['Wout'])}")
print(f"dh    = dz @ Wout.T, shape {dh.shape}, equals tiny.py: {np.allclose(dh, d['h'])}")
print("dh row 2, entry 0 = sum over j of dz[2,j] * Wout[0,j]:")
print("  " + " + ".join(f"({dz[2,j]:.4f})({Wout[0,j]:.1f})" for j in range(tiny.V)) + f" = {dh[2,0]:.4f}")
for nudge in (0.01, 0.001):           # does the loss really move by gradient * nudge?
    W2 = {k: a.copy() for k, a in W.items()}
    W2["Wout"][1, 3] += nudge
    change = tiny.forward(W2)["loss"] - v["loss"]
    print(f"nudge Wout[1,3] by {nudge}: loss change {change:+.6f}, change/nudge {change/nudge:+.4f}")
print("per-position pieces of dWout (each one is h[t] outer dz[t]); entry [1,3]:",
      [f"{np.outer(h[t], dz[t])[1, 3]:+.4f}" for t in range(tiny.T)])

print("\n== 3. the shape puzzle: h is 3x4, dz is 3x5, dWout must be 4x5")
for name, f in [("h @ dz", lambda: h @ dz), ("dz @ h.T", lambda: dz @ h.T),
                ("dz.T @ h", lambda: dz.T @ h), ("h.T @ dz", lambda: h.T @ dz)]:
    try:
        print(f"{name:9s} -> shape {f().shape}")
    except ValueError:
        print(f"{name:9s} -> does not fit (inner sizes differ)")

print("\n== 4. one rule, every linear layer: dW = (its input).T @ (gradient at its output)")
_, tg = tiny.torch_check(W)
for wk, x_in, dy in [("Wq", v["a"], d["Q"]), ("Wk", v["a"], d["K"]), ("Wv", v["a"], d["V"]),
                     ("Wo", v["C"], d["o"]), ("W1", v["bn"], d["u"]), ("W2", v["gu"], d["f"]),
                     ("Wout", v["h"], d["z"])]:
    mine = x_in.T @ dy
    print(f"{wk:4s} {str(x_in.shape):7s}.T @ {str(dy.shape):7s} = {str(mine.shape):7s}"
          f"  largest |difference| from torch {np.abs(mine - tg[wk]).max():.1e}")

print("\n== 5. bias: c2 is added to every row, so dc2 = df summed over rows")
print("df =\n", d["f"])
print("dc2 = df.sum(0) =", d["f"].sum(0), "  tiny.py's dc2 =", g["c2"])

import torch  # noqa: E402
head = torch.nn.Linear(tiny.D, tiny.V, bias=False).double()
with torch.no_grad():
    head.weight.copy_(torch.tensor(Wout).T)     # nn.Linear stores W.T: shape (out, in) = (5, 4)
hh = torch.tensor(h, requires_grad=False)
(head(hh) * torch.tensor(dz)).sum().backward()  # pushes exactly dz back into the head
print(f"\nPyTorch head.weight.grad shape {tuple(head.weight.grad.shape)}; "
      f"entry [3,1] = {head.weight.grad[3, 1].item():+.4f}; equals dWout.T: "
      f"{np.allclose(head.weight.grad.numpy(), dWout.T)}")
