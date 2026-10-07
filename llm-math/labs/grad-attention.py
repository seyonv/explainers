"""Backprop through attention by hand, on the tiny model (card: grad-attention).

Walks the attention backward pass one line at a time, checks each step against tiny.backward,
and measures why Wq's gradients are small.

Run:  python labs/grad-attention.py      (needs numpy; imports tiny.py from this folder)"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tiny  # noqa: E402

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
W = tiny.W
v = tiny.forward(W)
g, d = tiny.backward(W, v)
A, V, Q, K, a = v["A"], v["V"], v["Q"], v["K"], v["a"]

# ---- step 1: o = C @ Wo + bo, so dC = do @ Wo.T   (do is dx1: the residual copies it)
do = d["x1"]
dC = do @ W["Wo"].T
print("== dC = do @ Wo.T")
print("do row 2 =", do[2])
print("dC row 2 =", dC[2], "  matches tiny:", np.allclose(dC, d["C"]))

# ---- step 2: C = A @ V, the linear rule twice
dA = dC @ V.T
dV = A.T @ dC
print("\n== dA = dC @ V.T   and   dV = A.T @ dC")
print("dA row 2 =", dA[2])
print("dA[2,1] = dC row 2 . V row 1 =", " + ".join(f"({x:.4f})({y:.4f})" for x, y in zip(dC[2], V[1])), f"= {dC[2] @ V[1]:.4f}")
print("dV row 0 = 1.0000*dC0 + 0.4979*dC1 + 0.1737*dC2 =", dV[0])
print("dV row 2 = only token 2 looks at it, with weight", f"{A[2, 2]:.4f}:", dV[2])
print("matches tiny:", np.allclose(dA, d["A"]), np.allclose(dV, d["V"]))

# ---- step 3: softmax backward, one row at a time: dS = A * (dA - sum(dA * A))
print("\n== softmax backward")
for i in range(3):
    s = (dA[i] * A[i]).sum()
    print(f"row {i}: A = {A[i]}  dA = {dA[i]}  sum(dA*A) = {s:.4f}  dA - sum = {dA[i] - s}  dS = {A[i] * (dA[i] - s)}")
dS = A * (dA - (dA * A).sum(-1, keepdims=True))
print("row sums of dS (always 0):", dS.sum(-1))
print("row 1, two keys: A10*A11*(dA10 - dA11) =", f"{A[1,0]:.4f}*{A[1,1]:.4f}*({dA[1,0]:.4f} - {dA[1,1]:.4f}) = {A[1,0]*A[1,1]*(dA[1,0]-dA[1,1]):.4f}")
print("matches tiny:", np.allclose(dS, d["S"]))

# ---- step 4: S = Q @ K.T / sqrt(d), the linear rule again
dSr = dS / np.sqrt(tiny.D)
dQ = dSr @ K
dK = dSr.T @ Q
print("\n== dQ = (dS/2) @ K   and   dK = (dS/2).T @ Q")
print("dS/2 row 2 =", dSr[2])
print("dQ row 2 =", dQ[2])
print("dQ[2,3] =", " + ".join(f"({x:.4f})({y:.4f})" for x, y in zip(dSr[2], K[:, 3])), f"= {dQ[2, 3]:.4f}")
print("dK rows sum to", dK.sum(0), " (each dS row sums to 0)")
print("matches tiny:", np.allclose(dQ, d["Q"]), np.allclose(dK, d["K"]))

# ---- step 5: Q, K, V = a @ Wq, a @ Wk, a @ Wv
dWq, dWk, dWv = a.T @ dQ, a.T @ dK, a.T @ dV
print("\n== dWq = a.T @ dQ, dWk = a.T @ dK, dWv = a.T @ dV")
print("dWq[0,3] = sum_i a[i,0]*dQ[i,3] =", " + ".join(f"({a[i,0]:.4f})({dQ[i,3]:.4f})" for i in range(3)), f"= {dWq[0, 3]:.4f}")
for name, m in [("Wq", dWq), ("Wk", dWk), ("Wv", dWv)]:
    print(f"largest |d{name}| = {np.abs(m).max():.4f}")
print("matches tiny:", np.allclose(dWq, g["Wq"]), np.allclose(dWk, g["Wk"]), np.allclose(dWv, g["Wv"]))

# ---- step 6: a fed three branches, so its gradient is the sum of three
pq, pk, pv = dQ @ W["Wq"].T, dK @ W["Wk"].T, dV @ W["Wv"].T
print("\n== da = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T")
print("row 2:  via Q", pq[2], " via K", pk[2], " via V", pv[2])
print("        total", (pq + pk + pv)[2], "  matches tiny:", np.allclose(pq + pk + pv, d["a"]))
print("row 0:  via Q", pq[0], " via K", pk[0], " via V", pv[0])

# ---- why Wq's gradients are small: how much shrinks at each stage
print("\n== why Wq is small: the size of the gradient at each stage (largest |entry|)")
for name, m in [("dC", dC), ("dA", dA), ("dS", dS), ("dS/2", dSr), ("dQ", dQ), ("dV", dV)]:
    print(f"{name:5s} {np.abs(m).max():.4f}")

# ---- the nudge test: does the hand-derived gradient predict the loss change?
print("\n== nudge test (central difference, h = 0.001)")
for key, idx in [("Wq", (0, 3)), ("Wk", (3, 1)), ("Wv", (0, 2))]:
    h = 1e-3
    Wp = {k: x.copy() for k, x in W.items()}; Wp[key][idx] += h
    Wm = {k: x.copy() for k, x in W.items()}; Wm[key][idx] -= h
    est = (tiny.forward(Wp)["loss"] - tiny.forward(Wm)["loss"]) / (2 * h)
    print(f"{key}{list(idx)}: loss change / nudge = {est:.4f}   gradient = {g[key][idx]:.4f}")
