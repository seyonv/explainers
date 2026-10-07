"""The feed-forward block of the tiny model, worked number by number, plus two "why" checks.
Card: llm-math/ffn-and-gelu.html  ("The bend that makes depth worth having")

Run:  python labs/ffn-and-gelu.py            with GELU, as GPT-2 and Raschka use
      python labs/ffn-and-gelu.py relu       swap in ReLU and see which gradients die
Needs: numpy (tiny.py imports torch only for its own check, which this script doesn't call)."""
import sys

import numpy as np

import tiny

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
ACT = sys.argv[1] if len(sys.argv) > 1 else "gelu"

if ACT == "relu":                       # swap the bend for ReLU: max(u, 0), slope 0 or 1
    tiny.gelu = lambda u: np.maximum(u, 0.0)
    tiny.gelu_back = lambda u: (u > 0).astype(float)

W = tiny.W
v = tiny.forward(W)
g, d = tiny.backward(W, v)
bn, u, gu, f = v["bn"], v["u"], v["gu"], v["f"]

print(f"== activation: {ACT}")
# 1. one entry by hand: u[0,0] = row 0 of LN2(x1) dotted with column 0 of W1 (c1 is still 0)
terms = bn[0] * W["W1"][:, 0]
print("u[0,0] terms  bn[0,k] * W1[k,0] =", terms, " sum =", f"{terms.sum():.4f}")

# 2. the tanh form of GELU, one step at a time, on that entry (printed in both modes)
x = u[0, 0]
inner = x + 0.044715 * x**3
t = np.tanh(np.sqrt(2 / np.pi) * inner)
print(f"GELU({x:.4f}): u^3 = {x**3:.4f}, inner = {inner:.4f}, tanh(0.7979 * inner) = {t:.4f}, "
      f"0.5 * u * (1 + tanh) = {0.5 * x * (1 + t):.4f}")
if ACT == "gelu":                       # the dip: GELU goes a little below 0 before coming back up
    grid = np.linspace(-3, 0, 30001)
    low = grid[tiny.gelu(grid).argmin()]
    print(f"GELU's lowest point on [-3, 0]: {tiny.gelu(low):.4f} at u = {low:.4f}")

# 3. each token's row is transformed on its own: change token 2's input, rows 0 and 1 don't move
bn2 = bn.copy(); bn2[2] += 1.0
f2 = tiny.gelu(bn2 @ W["W1"] + W["c1"]) @ W["W2"] + W["c2"]
print("change row 2 of the input by +1: rows 0-1 of f move by", f"{np.abs(f2[:2] - f[:2]).max():.1f}",
      "| row 2 moves by up to", f"{np.abs(f2[2] - f[2]).max():.4f}")

# 4. the gradient that reaches each of the 24 hidden numbers
print("u =\n", u, sep="")
print("activation slope at u =\n", tiny.gelu_back(u), sep="")
print("du =\n", d["u"], sep="")
print(f"hidden numbers with exactly zero gradient: {(d['u'] == 0).sum()} of {u.size}"
      f"   (u < 0 in {(u < 0).sum()})")
print(f"loss = {v['loss']:.4f}")
