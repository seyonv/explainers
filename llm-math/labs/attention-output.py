"""Why attention is a weighted average, measured on the tiny model (card: attention-output).

Run:  python labs/attention-output.py      (needs numpy; imports tiny.py from this folder)"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tiny  # noqa: E402

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
v = tiny.forward(tiny.W)
A, V, S = v["A"], v["V"], v["S_masked"]

print("== C = A @ V, row 2 term by term")
for j in range(3):
    print(f"A[2,{j}] = {A[2, j]:.4f}  x  V{j} = {V[j]}  ->  {A[2, j] * V[j]}")
print("C row 2 =", v["C"][2], "  weights sum to", f"{A[2].sum():.4f}")

print("\n== a weighted average stays inside the values' range (each column)")
print("V column min =", V.min(0))
print("V column max =", V.max(0))
print("C row 2      =", v["C"][2])

print("\n== skip the divide-by-sum: use exp(S) as the weights")
e = np.exp(S[2])
print("exp(S row 2) =", e, "  sum =", f"{e.sum():.4f}")
print("C row 2 without dividing =", e @ V, f"  ({e.sum():.4f} x too big)")

print("\n== sum vs average as the context grows (random unit-variance values, d = 64)")
rng = np.random.default_rng(0)
for n in [3, 64, 1024]:
    vals = rng.standard_normal((n, 64))
    s = rng.standard_normal(n)
    w = np.exp(s - s.max()); w = w / w.sum()
    raw = np.exp(s)
    print(f"n = {n:4d}: size of output, weighted average {np.linalg.norm(w @ vals):7.3f}"
          f"   unnormalised exp weights {np.linalg.norm(raw @ vals):8.2f}   size of one value {np.sqrt(64):.3f}")

print("\n== why V is separate from K: if the values were the keys")
print("K =\n", v["K"])
print("V =\n", V)
print("C row 2 with V = K would be", A[2] @ v["K"])

print("\n== two heads instead of one: split the 4 columns of Q, K, V into 2 heads of 2")
Q, K = v["Q"], v["K"]
mask = np.triu(np.ones((3, 3), bool), k=1)
Cs = []
for h in range(2):
    cols = slice(2 * h, 2 * h + 2)
    Sh = np.where(mask, -np.inf, Q[:, cols] @ K[:, cols].T / np.sqrt(2))
    Ah = tiny.softmax(Sh)
    Cs.append(Ah @ V[:, cols])
    print(f"head {h}: A =\n{Ah}")
print("concatenated C (3 x 4) =\n", np.concatenate(Cs, axis=1))
