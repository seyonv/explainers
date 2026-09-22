"""FLOPs per token: 2N to serve, 6N to train.

(a) Counts forward-pass FLOPs for ONE token of Llama-3.1-8B, matrix by matrix,
    using the rule: (m x k) @ (k x n) costs 2*m*k*n FLOPs (one multiply + one add).
(b) Adds the attention dot-products (q.K and weights.V), which depend on how
    many tokens are already in context, and compares with the 2N rule and the
    Scaling Book's T/(8D) shortcut.
(c) Training total with 6*N*tokens, turned into H100-hours at an ASSUMED MFU,
    and compared with the GPU-hours Meta reports in the Llama 3.1 model card.
(d) A tiny numpy check that the 2mkn rule and the "backward = 2x forward" rule
    are just counting.

Run: python3 labs/flops-per-token.py   (numpy only, CPU, < 1 s)
"""
import numpy as np

# Llama-3.1-8B shapes (config.json)
L, D, F, N_HEADS, K_HEADS, H, V = 32, 4096, 14336, 32, 8, 128, 128256
CONTEXT = 8192               # tokens already in the KV cache when this token is processed
H100_BF16 = 989e12           # dense BF16 FLOP/s, H100 SXM
MFU = 0.40                   # ASSUMED; Llama 3 paper reports 38-43% for 405B pretraining
TRAIN_TOKENS = 15e12         # "~15T tokens" (Llama 3.1 model card)
META_GPU_HOURS = 1.46e6      # Llama 3.1 8B "Training Time (GPU hours)" (model card)


def mm(m, k, n):
    """FLOPs of an (m x k) @ (k x n) matmul."""
    return 2 * m * k * n


# ---------------------------------------------------------------- (a) matrices
print("== (a) forward FLOPs for one token (m = 1 row), matrix by matrix ==")
rows = [  # name, k (in), n (out)
    ("W_q    D -> N*H", D, N_HEADS * H),
    ("W_k    D -> K*H", D, K_HEADS * H),
    ("W_v    D -> K*H", D, K_HEADS * H),
    ("W_o    N*H -> D", N_HEADS * H, D),
    ("W_gate D -> F", D, F),
    ("W_up   D -> F", D, F),
    ("W_down F -> D", F, D),
]
per_layer = 0
for name, k, n in rows:
    f = mm(1, k, n)
    per_layer += f
    print(f"{name:<16} 2*1*{k}*{n:<6} = {f/1e6:8.2f} MFLOP   (params {k*n/1e6:6.2f}M)")
layers = L * per_layer
head = mm(1, D, V)
matmul_total = layers + head
params_layers = L * sum(k * n for _, k, n in rows) + L * 2 * D
params_total = params_layers + V * D + D * V + D
print(f"one layer                       = {per_layer/1e6:8.2f} MFLOP  (= 2 x {per_layer/2/1e6:.2f}M params)")
print(f"x {L} layers                     = {layers/1e9:8.2f} GFLOP")
print(f"LM head  2*1*{D}*{V}      = {head/1e9:8.2f} GFLOP")
print(f"embedding lookup                =     0.00 GFLOP  (a table read, not a matmul)")
print(f"all matmuls, one token          = {matmul_total/1e9:8.2f} GFLOP")
print(f"2N rule: 2 x {params_total/1e9:.2f}B params     = {2*params_total/1e9:8.2f} GFLOP")
print(f"counted / 2N                    = {matmul_total/(2*params_total):8.1%}  "
      f"(gap = the {V*D/1e6:.0f}M-param embedding table)")

# ---------------------------------------------------------------- (b) attention
print(f"\n== (b) attention dot-products for one token at context S (forward) ==")
print("q.K^T : 2*S*H per query head;  weights.V : 2*S*H per query head")
print("per layer = 4*S*N*H = 4*S*D  (GQA shares K/V between heads but NOT the FLOPs)")
print(f"{'context S':>10} {'attn GFLOP':>11} {'vs 2N':>8} {'book T/(8D)':>12} {'total GFLOP':>12}")
for S in sorted({1024, 8192, 32768, 131072, CONTEXT}):
    att = L * 4 * S * N_HEADS * H
    print(f"{S:>10,} {att/1e9:11.2f} {att/(2*params_total):8.1%} {S/(8*D):12.1%} "
          f"{(2*params_total+att)/1e9:12.2f}")
print("book T/(8D) assumes F = 4D, K = N (plain MHA), D = N*H, full (non-causal) T x T,")
print("and divides by per-layer matmul FLOPs only. Llama has F = 3.5D and GQA, hence the gap.")
print("In prefill with a causal mask, token t only sees t tokens: average context = T/2.")

# ---------------------------------------------------------------- (c) training
print("\n== (c) training: C = 6 * N * tokens ==")
c_train = 6 * params_total * TRAIN_TOKENS
per_gpu = H100_BF16 * MFU
hours = c_train / per_gpu / 3600
implied_mfu = c_train / (META_GPU_HOURS * 3600 * H100_BF16)
print(f"6 x {params_total/1e9:.2f}e9 x {TRAIN_TOKENS/1e12:.0f}e12  = {c_train:.3e} FLOPs")
print(f"one H100 at {MFU:.0%} MFU (assumed) = {per_gpu/1e12:.1f} TFLOP/s")
print(f"H100-hours                     = {c_train:.3e} / {per_gpu:.3e} / 3600 = {hours/1e6:.3f}M")
print(f"Meta reports (model card)      = {META_GPU_HOURS/1e6:.2f}M GPU-hours")
print(f"MFU implied by Meta's hours    = {implied_mfu:.1%}  (if all those hours were this one 6ND run)")
att_train = 3 * L * 4 * (8192 / 2) * N_HEADS * H   # 8k sequences, causal: average context 4k
print(f"attention adds, 8k-token sequences, causal: 3 x {L} x 4 x 4096 x {D} = {att_train/1e9:.2f} GFLOP/token "
      f"= +{att_train/(6*params_total):.1%} on 6N")
print(f"check on 405B: 6 x 405e9 x 15.6e12 = {6*405e9*15.6e12:.2e}  (paper: 3.8e25)")

# ---------------------------------------------------------------- (d) numpy check
print("\n== (d) counting check on a small matmul ==")
m, k, n = 64, 96, 80
A, W = np.random.rand(m, k), np.random.rand(k, n)
C = A @ W
mults = m * n * k                 # each output: k multiplies ...
adds = m * n * (k - 1)            # ... and k-1 adds (~k)
print(f"({m}x{k})@({k}x{n}): multiplies {mults:,} + adds {adds:,} = {mults+adds:,}  ~ 2mkn = {mm(m,k,n):,}")
dC = np.ones_like(C)
dW = A.T @ dC                     # gradient for the weights: (k x m)@(m x n) = 2mkn
dA = dC @ W.T                     # gradient for the input:   (m x n)@(n x k) = 2mkn
print(f"forward 2mkn = {mm(m,k,n):,}; backward dW + dA = {mm(k,m,n)+mm(m,n,k):,}  -> total 3x forward = 6mkn")

# Try this:
# 1. Set CONTEXT = 131072 and look at the last row of (b): attention alone is ~4.3x the
#    2N budget, so one token costs ~85 GFLOP, not 16. The 2N rule is off by ~5x.
# 2. DeepSeek-V3 (MoE): only 37B of 671B params are used per token, so 2N uses N = 37e9:
#    print(2*37e9/1e9, "GFLOP/token") -> 74, vs 1342 if you wrongly used 671B.
# 3. Change MFU to 0.14 and watch the H100-hours land near Meta's 1.46M.
