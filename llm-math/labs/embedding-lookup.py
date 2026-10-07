"""Card 4, "A token becomes a row of a table": three measurements on the tiny model.

1. Looking up rows E[ids] gives exactly the same numbers as one-hot(ids) @ E.
2. Why a table, not the ID itself: if each token were just its ID times one learned vector w,
   every token would sit on one line, `the` (ID 0) would always be all zeros,
   and `sat` (ID 2) would always be exactly halfway between `cat` (1) and `on` (3).
3. What the lookup saves: the one-hot product does vocab x d_model multiply-adds per token,
   almost all of them by zero. The lookup just copies d_model numbers.

Run:  python labs/embedding-lookup.py      (needs numpy; imports tiny.py from this folder)"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tiny  # noqa: E402  (the shared tiny model: same weights as every card)

E, P, ids = tiny.W["E"], tiny.W["P"], tiny.inputs
V, D = E.shape

# ---------------------------------------------------------------- 1. lookup == one-hot @ E
onehot = np.eye(V)[ids]                  # 3 x 5: row t has a single 1 in column ids[t]
print("one-hot(ids) =\n", onehot)
print("one-hot(ids) @ E =\n", onehot @ E)
print("E[ids] =\n", E[ids])
print("largest difference:", np.abs(onehot @ E - E[ids]).max())
zeros = (onehot[0] == 0).sum() * D       # multiplications by zero for token 0
print(f"multiply-adds for one token: {V * D}, of which {zeros} are by zero")
print("x0 = E[ids] + P =\n", E[ids] + P)

# ---------------------------------------------------------------- 2. the ID as a number
w = E[1]                                 # any one learned vector; the results hold for every w
by_id = np.arange(V)[:, None] * w + 0.0      # token k gets k * w
print("\nID-times-w table (one row per token) =\n", by_id)
print("rank of the ID-times-w table:", np.linalg.matrix_rank(by_id), "  rank of E:", np.linalg.matrix_rank(E))
print("`the` (ID 0) is always:", by_id[0])
print("sat - (cat + on)/2 =", np.abs(by_id[2] - (by_id[1] + by_id[3]) / 2))
# the first layer is linear, so the "halfway" survives it: check through Wq
q = by_id @ tiny.W["Wq"]
print("through Wq, sat - (cat + on)/2 =", np.abs(q[2] - (q[1] + q[3]) / 2))
print("with the table E, sat - (cat + on)/2 =", E[2] - (E[1] + E[3]) / 2)
cos = by_id[1] @ by_id[4] / np.linalg.norm(by_id[1]) / np.linalg.norm(by_id[4])
print(f"cosine(cat, mat) = {cos:.4f}   (with the table E: "
      f"{E[1] @ E[4] / np.linalg.norm(E[1]) / np.linalg.norm(E[4]):.4f})")

# ---------------------------------------------------------------- 3. training moves rows, but only looked-up rows
v = tiny.forward(tiny.W)
g, _ = tiny.backward(tiny.W, v)
print("\ndE =\n", g["E"])
print("E after one SGD step at lr 1.0 (E - dE) =\n", E - g["E"])

# ---------------------------------------------------------------- 4. the cost at real scale
for name, vocab, d in [("tiny", V, D), ("GPT-2 124M", 50_257, 768), ("Qwen2.5-0.5B", 151_936, 896)]:
    print(f"{name:13s} E is {vocab:,} x {d} = {vocab * d:,} weights; "
          f"one-hot product: {vocab * d:,} multiply-adds per token; lookup: copy {d} numbers")
print(f"Qwen's E as a share of its 494,032,768 parameters: {151_936 * 896 / 494_032_768:.1%}")
