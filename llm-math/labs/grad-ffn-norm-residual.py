"""Card 14, "The residual is a gradient highway": the backward pass through x2 = x1 + FFN(LN2(x1)).

Uses the tiny model from tiny.py and follows row 2 (the token "sat") backward:
  1. Through the +: dx2 is copied to both the stream (x1) and the branch (f).
  2. Through W2 and GELU: du = (df @ W2.T) * GELU'(u), one number at a time.
  3. dW1 = bn.T @ du, the linear-layer rule, with entry [0, 0] spelled out.
  4. Through LayerNorm 2: the two subtractions, and why each row of the result sums to 0.
  5. What goes wrong if LayerNorm's backward only divided by sigma.
  6. dx1 = dx2 + (LayerNorm 2's backward): how much of dx1 came down the highway.

Run:  python labs/grad-ffn-norm-residual.py      Needs: numpy."""
import numpy as np

import tiny

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
W = tiny.W
v = tiny.forward(W)
g, d = tiny.backward(W, v)
r = 2                                          # row 2 = "sat", the last position

print("== 1. through the +  (x2 = x1 + f)")
print("dx2 row 2           ", d["x2"][r])
print("df row 2 (a copy)   ", d["x2"][r], "  the branch gets the same numbers")

print("\n== 2. through W2, then GELU")
dgu = d["x2"] @ W["W2"].T                     # d(GELU(u)) = df @ W2.T, 3 x 8
slope = tiny.gelu_back(v["u"])                # GELU'(u), one slope per number
print("u row 2             ", v["u"][r])
print("dgu row 2           ", dgu[r])
print("GELU'(u) row 2      ", slope[r])
print("du row 2 = dgu*slope", dgu[r] * slope[r], "  (tiny.py:", d["u"][r], ")")
for j in (0, 1):
    print(f"  entry [2,{j}]: {dgu[r, j]:.4f} x {slope[r, j]:.4f} = {dgu[r, j] * slope[r, j]:.4f}")

print("\n== 3. dW1 = bn.T @ du  (4 x 3 times 3 x 8 = 4 x 8)")
col = v["bn"][:, 0] * d["u"][:, 0]
print("bn[:,0]             ", v["bn"][:, 0])
print("du[:,0]             ", d["u"][:, 0])
print("products            ", col, f" sum {col.sum():.4f}   (dW1[0,0] = {g['W1'][0, 0]:.4f})")

print("\n== 4. through LayerNorm 2  dx = (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)) / sigma")
xhat, sigma = v["ln2"]
dbn = d["u"] @ W["W1"].T                      # gradient arriving at LayerNorm 2's output
dxhat = dbn * W["g2"]                         # g2 = 1 at the start, so dxhat = dbn
m1 = dxhat.mean(-1, keepdims=True)
m2 = (dxhat * xhat).mean(-1, keepdims=True)
dln2 = (dxhat - m1 - xhat * m2) / sigma
print("xhat row 2 (= bn)   ", xhat[r])
print(f"sigma row 2          {sigma[r, 0]:.4f}")
print("dxhat row 2         ", dxhat[r])
print(f"mean(dxhat)          {m1[r, 0]:.4f}")
print(f"mean(dxhat * xhat)   {m2[r, 0]:.4f}")
print("dxhat - mean        ", (dxhat - m1)[r])
print("  - xhat*mean2      ", (dxhat - m1 - xhat * m2)[r])
print("  / sigma = LN back ", dln2[r])
print("row sums of LN2 back", dln2.sum(-1), "  (each is 0 up to rounding)")
print("dxhat row sums      ", dxhat.sum(-1), "  (before the subtractions: not 0)")
print("sum(LN back * xhat) per row", (dln2 * xhat).sum(-1), "  (also 0: no push on the spread)")
shift = tiny.layernorm(v["x1"] + 0.5, W["g2"], W["b2"])[0] - v["bn"]
print(f"add 0.5 to every number of x1: LN2 output changes by at most {np.abs(shift).max():.1e}")
grow = tiny.layernorm(v["x1"] * 1.5, W["g2"], W["b2"])[0] - v["bn"]
print(f"scale x1 by 1.5:               LN2 output changes by at most {np.abs(grow).max():.1e}")

print("\n== 5. a wrong LayerNorm backward that only divides by sigma")
naive = dxhat / sigma
print("naive row 2         ", naive[r], f"  sum {naive[r].sum():.4f}")
print("correct row 2       ", dln2[r], f"  sum {dln2[r].sum():.4f}")

print("\n== 6. dx1 = dx2 + LN2 back")
print("dx2 row 2           ", d["x2"][r])
print("LN2 back row 2      ", dln2[r])
print("dx1 row 2           ", d["x2"][r] + dln2[r], "  (tiny.py:", d["x1"][r], ")")
for t in range(3):
    print(f"row {t}: size of dx2 {np.linalg.norm(d['x2'][t]):.4f}   size of LN2 back {np.linalg.norm(dln2[t]):.4f}"
          f"   size of dx1 {np.linalg.norm(d['x1'][t]):.4f}")
