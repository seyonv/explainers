"""Megatron tensor parallelism: split the MLP and attention, check it equals the unsplit layer.

Run: python3 labs/tp-split.py   (numpy only, CPU, about 1 s)

Part 1 checks numerically, on small shapes with the same structure as Llama-3.1-70B at TP8
(64 query heads, 8 KV heads, 8 GPUs, F = 3.5 x D), that:
  - Megatron's GeLU MLP split column-then-row equals the unsplit MLP (Eq 1 and 3),
    while splitting the first GEMM by rows first does NOT work without a sync (Eq 2);
  - the backward pass through the split MLP needs the f all-reduce to get dX right;
  - Llama's SwiGLU MLP (gate and up column-parallel, down row-parallel) equals the unsplit one;
  - GQA attention split by heads (q, k, v column-parallel, o row-parallel) equals the unsplit one.
A fake communicator counts every all-reduce and its bytes.
Part 2 applies the split to real Llama-3.1-70B shapes: per-GPU slices, weight bytes, and the
all-reduce messages per decode step. "2 all-reduces per layer, forward only" is our inference
from Megatron's count of 2 forward + 2 backward per layer (the paper is about training).
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------- a fake communicator that counts all-reduces ----------------
class Comm:
    def __init__(self):
        self.calls, self.bytes = 0, 0
    def all_reduce(self, parts, bytes_per_elem=2):   # parts: one partial array per GPU
        self.calls += 1
        self.bytes += parts[0].size * bytes_per_elem  # message size S (BF16 on a real GPU)
        total = sum(parts)
        return [total.copy() for _ in parts]          # every GPU gets the full sum

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_grad(x):
    t = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)

def silu(x):
    return x / (1 + np.exp(-x))

def maxerr(a, b):
    return float(np.max(np.abs(a - b)))

# ---------------- Part 1: small shapes, same structure as 70B at TP8 ----------------
TP = 8
N_Q, N_KV, HD = 64, 8, 8          # heads like 70B (64 q, 8 kv); head dim shrunk from 128 to 8
D = N_Q * HD                      # 512   (70B: 8192)
F = int(3.5 * D)                  # 1792  (70B: 28672 = 3.5 x 8192)
S = 16                            # tokens in the batch (one sequence, causal)
X = rng.standard_normal((S, D))

print("PART 1: numeric check on small shapes (float64)")
print(f"  TP={TP}, D={D}, F={F}, {N_Q} query heads, {N_KV} KV heads, head dim {HD}, {S} tokens")

# (a) GeLU non-linearity: why the row split of A needs a sync
x1, x2 = 1.0, -0.5
print(f"\n(a) GeLU is non-linear: GeLU({x1}+{x2}) = {gelu(x1 + x2):.4f}  vs  "
      f"GeLU({x1})+GeLU({x2}) = {gelu(x1):.4f}+({gelu(x2):.4f}) = {gelu(x1) + gelu(x2):.4f}")

# (b) Megatron GeLU MLP: Z = GeLU(X A) B
A = rng.standard_normal((D, F)) / np.sqrt(D)
B = rng.standard_normal((F, D)) / np.sqrt(F)
Z_ref = gelu(X @ A) @ B

comm = Comm()
A_cols = np.split(A, TP, axis=1)                 # column-parallel: A = [A1 ... A8]
B_rows = np.split(B, TP, axis=0)                 # row-parallel:    B = [B1; ...; B8]
Y_i = [gelu(X @ Ai) for Ai in A_cols]            # f: X replicated, identity forward; no comm
Z_parts = [Yi @ Bi for Yi, Bi in zip(Y_i, B_rows)]   # partial sums {U}
Z_split = comm.all_reduce(Z_parts)[0]            # g: the one forward all-reduce
print(f"\n(b) GeLU MLP, column-then-row split: max |split - unsplit| = {maxerr(Z_split, Z_ref):.1e}"
      f"   all-reduces: {comm.calls}, message {S}x{D} = {S*D:,} elements")

# row-first (Eq 2): X split by columns, A by rows. Applying GeLU before summing is wrong.
X_cols = np.split(X, TP, axis=1)
A_rows = np.split(A, TP, axis=0)
wrong = sum(gelu(Xi @ Ai) for Xi, Ai in zip(X_cols, A_rows)) @ B
comm_r = Comm()
right = gelu(comm_r.all_reduce([Xi @ Ai for Xi, Ai in zip(X_cols, A_rows)])[0]) @ B
print(f"    row-first, GeLU before the sum : max error = {maxerr(wrong, Z_ref):.2f}  (wrong)")
print(f"    row-first, all-reduce then GeLU: max error = {maxerr(right, Z_ref):.1e}, but that sync"
      f" moves {S}x{F} = {S*F:,} elements ({F/D:.1f}x the g message), and every GPU then holds all of B")

# (c) backward: why f is an all-reduce in the backward pass
dZ = rng.standard_normal((S, D))                 # upstream gradient, replicated (g backward = identity)
H = X @ A
dX_ref = ((dZ @ B.T) * gelu_grad(H)) @ A.T
dX_parts = [((dZ @ Bi.T) * gelu_grad(X @ Ai)) @ Ai.T for Ai, Bi in zip(A_cols, B_rows)]
comm_b = Comm()
dX_split = comm_b.all_reduce(dX_parts)[0]        # f backward
print(f"\n(c) backward: one GPU's dX alone has max error {maxerr(dX_parts[0], dX_ref):.2f};"
      f" after f's all-reduce: {maxerr(dX_split, dX_ref):.1e}")

# (d) Llama SwiGLU MLP: Z = (silu(X Wg) * (X Wu)) Wd
Wg = rng.standard_normal((D, F)) / np.sqrt(D)
Wu = rng.standard_normal((D, F)) / np.sqrt(D)
Wd = rng.standard_normal((F, D)) / np.sqrt(F)
Zs_ref = (silu(X @ Wg) * (X @ Wu)) @ Wd
comm_s = Comm()
parts = [(silu(X @ g) * (X @ u)) @ d for g, u, d in
         zip(np.split(Wg, TP, 1), np.split(Wu, TP, 1), np.split(Wd, TP, 0))]
Zs_split = comm_s.all_reduce(parts)[0]
print(f"\n(d) SwiGLU MLP, gate+up column / down row: max error = {maxerr(Zs_split, Zs_ref):.1e}"
      f"   all-reduces: {comm_s.calls}")

# (e) GQA attention with RoPE and a causal mask, split by heads
def rope(x):                                     # x: (S, heads, HD); rotates pairs within one head
    half = HD // 2
    freqs = 1.0 / (10000 ** (np.arange(half) / half))
    ang = np.arange(S)[:, None] * freqs[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)

def attention(X, Wq, Wk, Wv, nq, nkv):
    q = rope((X @ Wq).reshape(S, nq, HD))
    k = rope((X @ Wk).reshape(S, nkv, HD))
    v = (X @ Wv).reshape(S, nkv, HD)
    rep = nq // nkv                              # query heads per KV head
    out = np.empty((S, nq, HD))
    mask = np.triu(np.full((S, S), -np.inf), 1)
    for h in range(nq):
        kv = h // rep
        sc = q[:, h] @ k[:, kv].T / np.sqrt(HD) + mask
        p = np.exp(sc - sc.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        out[:, h] = p @ v[:, kv]
    return out.reshape(S, nq * HD)

Wq = rng.standard_normal((D, N_Q * HD)) / np.sqrt(D)
Wk = rng.standard_normal((D, N_KV * HD)) / np.sqrt(D)
Wv = rng.standard_normal((D, N_KV * HD)) / np.sqrt(D)
Wo = rng.standard_normal((N_Q * HD, D)) / np.sqrt(D)
O_ref = attention(X, Wq, Wk, Wv, N_Q, N_KV) @ Wo

comm_a = Comm()
q_per, kv_per = N_Q // TP, N_KV // TP
parts = []
for i in range(TP):                              # GPU i owns query heads 8i..8i+7 and KV head i
    Wq_i = Wq[:, i * q_per * HD:(i + 1) * q_per * HD]
    Wk_i = Wk[:, i * kv_per * HD:(i + 1) * kv_per * HD]
    Wv_i = Wv[:, i * kv_per * HD:(i + 1) * kv_per * HD]
    Wo_i = Wo[i * q_per * HD:(i + 1) * q_per * HD, :]
    parts.append(attention(X, Wq_i, Wk_i, Wv_i, q_per, kv_per) @ Wo_i)
O_split = comm_a.all_reduce(parts)[0]
print(f"\n(e) GQA attention by heads ({q_per} query heads + {kv_per} KV head per GPU): "
      f"max error = {maxerr(O_split, O_ref):.1e}   all-reduces: {comm_a.calls}")
print(f"    whole layer (attention + MLP), forward: {comm_a.calls + comm_s.calls} all-reduces")

# ---------------- Part 2: Llama-3.1-70B at TP8 ----------------
L, D, F, NQ, NKV, HD, V = 80, 8192, 28672, 64, 8, 128, 128256
TP, BYTES = 8, 2
print("\nPART 2: Llama-3.1-70B, BF16, TP8")
mats = [  # name, full shape, split axis ("col" splits outputs, "row" splits inputs)
    ("q_proj", (D, NQ * HD), "col"), ("k_proj", (D, NKV * HD), "col"), ("v_proj", (D, NKV * HD), "col"),
    ("o_proj", (NQ * HD, D), "row"),
    ("gate_proj", (D, F), "col"), ("up_proj", (D, F), "col"), ("down_proj", (F, D), "row"),
]
tot_full = tot_gpu = 0
print(f"  {'matrix':10s} {'full shape':>14s} {'split':>6s} {'per-GPU shape':>14s} {'params/GPU':>12s} {'MB/GPU':>8s}")
for name, (r, c), ax in mats:
    pr, pc = (r, c // TP) if ax == "col" else (r // TP, c)
    n = pr * pc
    tot_full += r * c; tot_gpu += n
    print(f"  {name:10s} {f'{r}x{c}':>14s} {ax:>6s} {f'{pr}x{pc}':>14s} {n/1e6:10.2f} M {n*BYTES/1e6:8.1f}")
attn_full = D * NQ * HD * 2 + D * NKV * HD * 2
mlp_full = 3 * D * F
print(f"  per layer: attention {attn_full/1e6:.1f}M, MLP {mlp_full/1e6:.1f}M, total {tot_full/1e6:.1f}M params"
      f" -> per GPU {tot_gpu/1e6:.2f}M = {tot_gpu*BYTES/1e6:.1f} MB")
norms = 2 * D                                    # two RMSNorm weight vectors per layer, replicated
emb = 2 * V * D                                  # input embedding + LM head (untied), vocab-split
per_gpu = L * (tot_gpu + norms) + emb / TP + D   # + final norm
total = L * (tot_full + norms) + emb + D
print(f"  x{L} layers = {L*tot_gpu*BYTES/1e9:.2f} GB; + embeddings/LM head split 8 ways "
      f"{emb/TP*BYTES/1e9:.2f} GB; + replicated norms {(L*norms+D)*BYTES/1e6:.1f} MB")
print(f"  weights per GPU = {per_gpu*BYTES/1e9:.2f} GB  (whole model {total/1e9:.2f}B params = {total*BYTES/1e9:.1f} GB)")
print(f"  query heads per GPU {NQ//TP}, KV heads per GPU {NKV//TP}; KV cache per token per GPU "
      f"{2*L*(NKV//TP)*HD*BYTES:,} B (of {2*L*NKV*HD*BYTES:,} B)")
for tp in (16, 32):
    rep = tp // NKV
    print(f"  TP{tp}: {NQ//tp} query heads per GPU, but {NKV} KV heads / {tp} GPUs -> each KV head replicated on "
          f"{rep} GPUs (KV cache stored {rep}x across the group)")

print("\n  all-reduces per forward pass: 2 per layer x 80 layers = 160 (training: 4 per layer = 320)")
print(f"  {'tokens in step':>16s} {'message B*D*2':>14s} {'per-GPU ring traffic 1.75x':>28s} {'x160 per step':>14s} {'bw time @450 GB/s':>18s}")
for toks, label in [(1, "decode B=1"), (64, "decode B=64"), (256, "decode B=256"), (2048, "prefill 2,048")]:
    msg = toks * D * BYTES
    ring = 2 * (TP - 1) / TP * msg
    t = ring / 450e9
    print(f"  {label:>16s} {msg:>12,} B {ring:>26,.0f} B {160*ring/1e6:>11.2f} MB {t*1e6:8.2f} us each, {160*t*1e3:6.2f} ms/step")
w_layer = tot_gpu * BYTES
print(f"  compare: each GPU reads {w_layer/1e6:.1f} MB of weights per layer; the B=1 message is "
      f"{D*BYTES/1024:.0f} KiB = 1/{w_layer/(D*BYTES):,.0f} of that")
print(f"  row-first MLP alternative would all-reduce B*F*2 = {F*BYTES:,} B per token ({F/D:.1f}x the g message)")

# Try this:
# 1. Set TP = 16 in Part 1 (N_KV stays 8): kv_per becomes 0 and the split breaks. Replicate each
#    KV head on 2 GPUs (kv index i // 2) to fix it, as serving engines do.
# 2. In (d), split Wd by columns instead of rows and see the error: the down projection must be row-parallel.
# 3. Change BYTES to 1 (FP8 weights) in Part 2: weights per GPU halve, but activations (and so the
#    all-reduce messages) are usually still BF16.
