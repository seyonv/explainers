"""Card 9, "Normalise, then add, don't replace": the extra measurements behind it.

Uses the tiny model from tiny.py (row 0 of x0 is the token "the" plus position 0).
  1. LayerNorm on row 0, one step at a time, and what changes if you divide by D-1.
  2. RMSNorm (Qwen's version) on the same row.
  3. LayerNorm ignores the row's overall size: 10 x row 0 gives the same output.
  4. 24 random layers with weights N(0, 1), with and without a LayerNorm in front of each.
  5. The residual stream itself is never normalised: the std of each row of x0, x1, x2.

Run:  python labs/norm-and-residual.py      Needs: numpy."""
import numpy as np

import tiny

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
v = tiny.forward(tiny.W)
row = v["x0"][0]

print("== 1. LayerNorm, row 0 of x0, by hand")
mu = row.mean()
dev = row - mu
var = (dev ** 2).mean()                       # divide by D = 4
sigma = np.sqrt(var + tiny.EPS)
print("x0 row 0      ", row)
print(f"mean           {mu:.4f}")
print("x - mean      ", dev)
print("squared       ", dev ** 2, f" sum {(dev ** 2).sum():.4f}")
print(f"variance (/D)  {var:.4f}   sigma {sigma:.4f}")
print("a row 0       ", dev / sigma, " (tiny.py:", v["a"][0], ")")
var1 = (dev ** 2).sum() / (len(row) - 1)      # divide by D - 1 instead
print(f"divide by D-1: variance {var1:.4f}  sigma {np.sqrt(var1 + tiny.EPS):.4f}  row ", dev / np.sqrt(var1 + tiny.EPS))
print("each row of a has mean", v["a"].mean(-1).round(4) + 0.0, "and std", v["a"].std(-1).round(4))

print("\n== 2. RMSNorm, the same row (no mean subtraction, no shift)")
rms = np.sqrt((row ** 2).mean() + tiny.EPS)
print(f"mean of squares {(row ** 2).mean():.4f}   rms {rms:.4f}")
print("RMSNorm row 0 ", row / rms)

print("\n== 3. scale the row by 10: LayerNorm's output doesn't move")
print("LN(10 * row 0)", tiny.layernorm(10 * v["x0"], tiny.W["g1"], tiny.W["b1"])[0][0])

print("\n== 4. 24 random layers, d = 896, weights N(0, 1)")
rng = np.random.default_rng(0)
d = 896
for use_norm in (False, True):
    x = rng.standard_normal(d)
    seen = []
    for _ in range(24):
        inp = (x - x.mean()) / np.sqrt(x.var() + tiny.EPS) if use_norm else x
        seen.append(inp.std())
        x = inp @ rng.standard_normal((d, d))
    label = "LayerNorm first" if use_norm else "no norm        "
    print(f"{label}: std of the input to layers 1, 2, 3, 24 = "
          f"{seen[0]:.3g}, {seen[1]:.3g}, {seen[2]:.3g}, {seen[-1]:.3g}   output of layer 24 {x.std():.3g}")

print("\n== 5. the residual stream is not normalised: std of each row (token)")
for name in ("x0", "x1", "x2"):
    print(f"{name}: ", v[name].std(-1))
