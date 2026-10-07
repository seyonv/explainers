"""Vectors and matrix multiply on the tiny model (card: vectors-and-matmul).

1. One dot product: sat's query against cat's key.
2. One entry of a matrix product, Q[0,0] = a row 0 . Wq column 0, and the shape rule.
3. Transpose: K is 3 x 4, K.T is 4 x 3.
4. Where the work goes: count the operations in one forward pass, layer by layer,
   for the tiny model and for Qwen2.5-0.5B on a 7-token prompt.
Run:  python labs/vectors-and-matmul.py      (needs numpy; imports tiny.py from this folder)"""
import numpy as np

import tiny

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
v = tiny.forward(tiny.W)

# ---------------------------------------------------------------- 1. a dot product
q, k = v["Q"][2], v["K"][1]                      # row 2 = "sat" asking, row 1 = "cat"
print("== dot product")
print("Q row 2       =", q)
print("K row 1       =", k)
print("pairwise      =", q * k)
print(f"sum           = {(q * k).sum():.4f}   (S_raw[2,1] in tiny.out)")
# How aligned are they? Divide by both lengths to get the cosine (1 = same direction, -1 = opposite).
print(f"|q|, |k|      = {np.linalg.norm(q):.4f}, {np.linalg.norm(k):.4f}")
print(f"cosine        = {(q @ k) / np.linalg.norm(q) / np.linalg.norm(k):.4f}")
print(f"S_raw row 2   = {v['S_raw'][2]}   (cat is the most aligned key)")

# ---------------------------------------------------------------- 2. one entry of a matmul
a, Wq = v["a"], tiny.W["Wq"]
print("\n== matmul: Q = a @ Wq")
print("shapes        :", a.shape, "@", Wq.shape, "->", (a @ Wq).shape)
print("a row 0       =", a[0])
print("Wq column 0   =", Wq[:, 0])
print("pairwise      =", a[0] * Wq[:, 0])
print(f"Q[0,0]        = {a[0] @ Wq[:, 0]:.4f}   (tiny.out Q[0,0] = {v['Q'][0, 0]:.4f})")
print(f"dot products  = {a.shape[0] * Wq.shape[1]} (one per output cell), each of length {a.shape[1]}")

# ---------------------------------------------------------------- 3. transpose
print("\n== transpose")
print("K   shape", v["K"].shape, "\n", v["K"])
print("K.T shape", v["K"].T.shape, "\n", v["K"].T)
print("Q @ K.T shape:", (v["Q"] @ v["K"].T).shape)


# ---------------------------------------------------------------- 4. where the work goes
# A matmul (m x n) @ (n x p) does m*n*p multiply-adds.
# Everything else is element-wise. We count it generously, as a fixed number of operations
# per number produced (a rough count, rounded up):
#   LayerNorm / RMSNorm 7 / 5 per number, softmax 5, GELU (tanh form) 9, SiLU-gate 5, RoPE 3,
#   add / scale / mask 1.
def mm(m, n, p):
    return m * n * p


def tiny_counts(T=3, D=4, H=8, V=5):
    return [
        ("embedding lookup + positions", 0, T * D),          # a lookup copies rows; then one add per number
        ("LayerNorm 1", 0, 7 * T * D),
        ("Q, K, V = a @ Wq, Wk, Wv", 3 * mm(T, D, D), 0),
        ("scores Q @ K.T", mm(T, D, T), 0),
        ("scale, mask, softmax", 0, (1 + 1 + 5) * T * T),
        ("mix A @ V", mm(T, T, D), 0),
        ("output projection @ Wo", mm(T, D, D), 0),
        ("residual add", 0, T * D),
        ("LayerNorm 2", 0, 7 * T * D),
        ("FFN up @ W1", mm(T, D, H), 0),
        ("GELU", 0, 9 * T * H),
        ("FFN down @ W2", mm(T, H, D), 0),
        ("residual add", 0, T * D),
        ("final LayerNorm", 0, 7 * T * D),
        ("output head @ Wout", mm(T, D, V), 0),
        ("softmax over vocab", 0, 5 * T * V),
    ]


def qwen_counts(T=7, D=896, L=24, nq=14, nkv=2, hd=64, F=4864, V=151936):
    per_layer = [
        ("RMSNorm", 0, 5 * T * D),
        ("Q, K, V projections", mm(T, D, nq * hd) + 2 * mm(T, D, nkv * hd), 0),
        ("RoPE on Q and K", 0, 3 * T * (nq + nkv) * hd),
        ("scores Q @ K.T (14 heads)", nq * mm(T, hd, T), 0),
        ("scale, mask, softmax", 0, 7 * nq * T * T),
        ("mix A @ V (14 heads)", nq * mm(T, T, hd), 0),
        ("output projection", mm(T, nq * hd, D), 0),
        ("residual adds", 0, 2 * T * D),
        ("RMSNorm", 0, 5 * T * D),
        ("FFN gate + up", 2 * mm(T, D, F), 0),
        ("SiLU gate", 0, 5 * T * F),
        ("FFN down", mm(T, F, D), 0),
    ]
    total = [(n, L * x, L * y) for n, x, y in per_layer]
    return [("embedding lookup", 0, 0)] + total + [("final RMSNorm", 0, 5 * T * D),
                                                   ("output head (896 x 151,936)", mm(T, D, V), 0),
                                                   ("softmax over vocab", 0, 5 * T * V)]


def report(title, rows):
    print(f"\n== where the work goes: {title}")
    print(f"{'step':34s} {'matmul mult-adds':>18s} {'other ops':>12s}")
    for name, m_, o in rows:
        print(f"{name:34s} {m_:18,d} {o:12,d}")
    M, O = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"{'total':34s} {M:18,d} {O:12,d}")
    print(f"matmul share = {M / (M + O):.1%}")
    return M, O


report("tiny model (3 tokens, d 4)", tiny_counts())
report("Qwen2.5-0.5B, 7-token prompt, all 24 layers", qwen_counts())
print(f"\none Qwen matmul: (7 x 896) @ (896 x 896) = {mm(7, 896, 896):,} multiply-adds")

# The same count per token, next to the parameter count: about one multiply-add per weight.
M = sum(r[1] for r in qwen_counts())
print(f"Qwen matmul multiply-adds per token = {M / 7:,.0f}   (parameters: 494,032,768)")

# Why the share grows: widen the tiny model's d (keeping 3 tokens, FFN hidden = 2d, vocab 5).
# Matmuls grow like d * d; the element-wise steps only like d.
print("\n== matmul share as the tiny model gets wider")
for d in [4, 16, 64, 256, 896]:
    rows = tiny_counts(D=d, H=2 * d)
    M, O = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"d = {d:4d}: matmul share {M / (M + O):.1%}")
