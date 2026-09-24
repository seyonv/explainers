"""DeepSeek-V3 from its config: parameters, KV cache and bytes read per decode token.

Card: perf-6-scaling-out/deepseek-v3-architecture.html
Run:  python3 labs/ds-v3-params.py      (stdlib + numpy, CPU, a few seconds)

Shapes: DeepSeek-V3 report §4.2 (arXiv 2412.19437) plus the Hugging Face config.json
(dense FFN width 18432 and vocab 129280 are only in the config).
Part 4 (gate) and part 5 (bias balancing) use synthetic affinities: illustrative only.
"""
import numpy as np

# ---------------------------------------------------------------- config
L = 61            # layers
DENSE = 3         # first 3 layers keep a dense FFN
D = 7168          # hidden size
V = 129280        # vocab (config.json; the report says "128K")
NH, DH, DR = 128, 128, 64   # heads, per-head dim, decoupled RoPE dim
DC, DCQ = 512, 1536         # KV latent, query latent
F_DENSE = 18432   # dense FFN intermediate (config.json)
F_EXP = 2048      # each expert's intermediate width
N_SHARED, N_ROUTED, K_ACTIVE = 1, 256, 8
NODES_TRAIN, M_NODES = 8, 4  # training layout: each layer's routed experts on 8 nodes; token goes to <= 4

B = 1e9
def swiglu(d, f):            # gate, up, down projections
    return 3 * d * f

# ---------------------------------------------------------------- 1. parameters
attn = (D * DCQ + DCQ                       # W_DQ + its RMSNorm
        + DCQ * NH * (DH + DR)              # W_UQ and W_QR (content + RoPE queries)
        + D * (DC + DR) + DC                # W_DKV and W_KR + latent RMSNorm
        + DC * NH * (DH + DH)               # W_UK and W_UV
        + NH * DH * D)                      # W_O
norms = 2 * D                               # pre-attention and pre-FFN RMSNorm
expert = swiglu(D, F_EXP)
dense_ffn = swiglu(D, F_DENSE)
router = N_ROUTED * D + N_ROUTED            # centroids e_i + balancing bias b_i
moe_total = (N_SHARED + N_ROUTED) * expert + router
moe_active = (N_SHARED + K_ACTIVE) * expert + router
emb = V * D                                 # embedding; the output head is the same size (untied)

total = (L * (attn + norms) + DENSE * dense_ffn + (L - DENSE) * moe_total
         + 2 * emb + D)
active = (L * (attn + norms) + DENSE * dense_ffn + (L - DENSE) * moe_active
          + 2 * emb + D)

print("1. PARAMETERS FROM THE CONFIG")
print(f"  one expert  3 x {D} x {F_EXP}           = {expert:,}  ({expert/1e6:.1f}M)")
print(f"  MLA attention per layer                 = {attn/1e6:,.1f}M   x {L} = {L*attn/B:.2f}B")
print(f"  dense FFN (layers 1-3)  3 x {D} x {F_DENSE} = {dense_ffn/1e6:,.1f}M  x {DENSE} = {DENSE*dense_ffn/B:.2f}B")
print(f"  MoE layer, all 257 experts + router     = {moe_total/B:.3f}B x {L-DENSE} = {(L-DENSE)*moe_total/B:.1f}B")
print(f"     routed experts only 256 x {L-DENSE} x 44.04M = {N_ROUTED*(L-DENSE)*expert/B:.1f}B")
print(f"     shared experts  {L-DENSE} x 44.04M            = {(L-DENSE)*expert/B:.2f}B")
print(f"  embedding + output head 2 x {V} x {D} = {2*emb/B:.2f}B")
print(f"  TOTAL  = {total/B:.1f}B   (report: 671B)")
print(f"  ACTIVE = {active/B:.1f}B   (report: 37B)  = attn {L*attn/B:.1f} + dense {DENSE*dense_ffn/B:.1f}"
      f" + 9 experts x 58 {(L-DENSE)*9*expert/B:.1f} + emb/head {2*emb/B:.1f} + router/norms"
      f" {((L-DENSE)*router + L*norms + D)/B:.2f}")
print(f"  active share = {active/total*100:.1f}%   routed-expert share of total = "
      f"{N_ROUTED*(L-DENSE)*expert/total*100:.1f}%")
mtp = attn + norms + moe_total + D * 2 * D + 2 * D + D   # one extra block + M_k (d x 2d) + 2 norms + head norm
print(f"  MTP module (1 block + d x 2d projection + norms) = {mtp/B:.1f}B; + its stored copies of"
      f" embedding and head = {(mtp+2*emb)/B:.1f}B   (HF README: 14B, 671 + 14 = 685B)")

# ---------------------------------------------------------------- 2. KV cache
print("\n2. KV CACHE PER TOKEN (BF16 = 2 bytes)")
kv_mla = (DC + DR) * L * 2
kv_gqa8 = 2 * 8 * DH * L * 2          # hypothetical: same heads dims, 8 KV heads like Llama
kv_mha = 2 * NH * DH * L * 2
for name, b in [("MLA  (512+64) x 61 x 2 B", kv_mla),
                ("GQA-8 hypothetical 2 x 8 x 128 x 61 x 2 B", kv_gqa8),
                ("MHA  hypothetical 2 x 128 x 128 x 61 x 2 B", kv_mha)]:
    print(f"  {name:44s} = {b:>9,} B = {b/1024:8.1f} KiB   32k-token seq = {b*32768/1e9:6.2f} GB")
print(f"  GQA-8 / MLA = {kv_gqa8/kv_mla:.2f}x   MHA / MLA = {kv_mha/kv_mla:.1f}x")

# ---------------------------------------------------------------- 3. bytes read per decode step
print("\n3. WEIGHT BYTES READ PER DECODE STEP (FP8 weights = 1 byte/param)")
llama70 = 70.55e9
print(f"  batch 1, DeepSeek-V3: {active/B:.1f} GB   (all 671B would be {total/B:.0f} GB -> {total/active:.1f}x more)")
print(f"  batch 1, Llama-3.1-70B FP8: {llama70/B:.1f} GB   -> V3 reads {active/llama70:.2f}x the bytes"
      f" of a model {total/llama70:.1f}x its size")
non_expert = active - (L - DENSE) * K_ACTIVE * expert      # read every step regardless of batch
print("  batch B, uniform routing (illustrative): distinct routed experts touched per layer = 256(1-(1-8/256)^B)")
for bs in (1, 8, 32, 64, 256):
    touched = N_ROUTED * (1 - (1 - K_ACTIVE / N_ROUTED) ** bs)
    step = non_expert + (L - DENSE) * touched * expert
    print(f"    B={bs:4d}: experts/layer {touched:6.1f}   bytes/step {step/B:6.1f} GB   per token {step/bs/B:6.2f} GB")

# ---------------------------------------------------------------- 4. the gate, on one token
print("\n4. GATE FOR ONE TOKEN (synthetic affinities, illustrative)")
rng = np.random.default_rng(0)
logits = rng.normal(0, 1, N_ROUTED)
s = 1 / (1 + np.exp(-logits))                         # s_i = sigmoid(u . e_i)
per_node = N_ROUTED // NODES_TRAIN                    # 32 experts per node
node_score = np.sort(s.reshape(NODES_TRAIN, per_node), axis=1)[:, -K_ACTIVE // M_NODES:].sum(1)
nodes = np.sort(np.argsort(-node_score)[:M_NODES])    # top M=4 nodes by sum of their top K/M=2 scores
mask = np.full(N_ROUTED, -np.inf)
for n in nodes:
    mask[n * per_node:(n + 1) * per_node] = 0
chosen = np.argsort(-(s + mask))[:K_ACTIVE]
g = s[chosen] / s[chosen].sum()                       # normalize over the selected 8
free = np.argsort(-s)[:K_ACTIVE]
print(f"  node scores (sum of top-2 per node): {np.round(node_score, 3).tolist()}")
print(f"  nodes kept (M=4): {nodes.tolist()}")
print(f"  chosen experts: {sorted(chosen.tolist())}   (unrestricted top-8 would be {sorted(free.tolist())},"
      f" spanning {len(set((free // per_node).tolist()))} nodes)")
print(f"  s of chosen: {np.round(s[chosen], 3).tolist()}")
print(f"  gates g = s / sum(s): {np.round(g, 3).tolist()}  sum = {g.sum():.3f}")
from math import comb
exp_nodes = NODES_TRAIN * (1 - comb(N_ROUTED - per_node, K_ACTIVE) / comb(N_ROUTED, K_ACTIVE))
print(f"  uniform random top-8 with no node limit touches {exp_nodes:.2f} of 8 nodes on average (max 8);"
      f" with M=4 at most 4")
print(f"  cross-node copies of one token's hidden state, FP8: {exp_nodes:.2f} x {D} B = {exp_nodes*D/1024:.1f} KiB"
      f"  vs  <= 4 x {D} B = {4*D/1024:.1f} KiB")

# ---------------------------------------------------------------- 5. aux-loss-free balancing
print("\n5. BIAS BALANCING (synthetic skewed router, 1024 tokens/step, illustrative)")
T, steps, gamma = 1024, 200, 0.001
skew = rng.normal(0, 0.5, N_ROUTED)                   # some experts are simply more attractive
b = np.zeros(N_ROUTED)
def step_load(b):
    s = 1 / (1 + np.exp(-(skew + rng.standard_normal((T, N_ROUTED), dtype=np.float32))))
    top = np.argpartition(-(s + b), K_ACTIVE, axis=1)[:, :K_ACTIVE]   # bias used ONLY to choose
    return np.bincount(top.ravel(), minlength=N_ROUTED), s, top
for t in range(steps + 1):
    load, s_all, top = step_load(b)
    if t in (0, 25, 50, 100, 200):
        print(f"  step {t:3d}: max/mean expert load = {load.max()/load.mean():.2f}   idle experts = {(load == 0).sum()}")
    b -= gamma * np.sign(load - load.mean())          # overloaded: -gamma, underloaded: +gamma
sel = np.take_along_axis(s_all, top, 1)
g_all = sel / sel.sum(1, keepdims=True)
print(f"  bias range after {steps} steps: {b.min():+.3f} .. {b.max():+.3f}; gates still come from raw s"
      f" (first token's gates sum to {g_all[0].sum():.3f})")

# ---------------------------------------------------------------- 6. MTP as a draft
print("\n6. MTP AS A SPECULATIVE DRAFT (depth 1)")
for acc in (0.85, 0.90):
    print(f"  acceptance {acc:.2f}: tokens per step = 1 + {acc:.2f} = {1+acc:.2f};"
          f" reported 1.8x TPS -> implied step overhead {(1+acc)/1.8:.2f}x")

# Try this:
# 1. Set K_ACTIVE = 16 (or F_EXP = 4096 with K_ACTIVE = 4): active params and bytes/step change, total doesn't.
# 2. Set M_NODES = 8 in part 4: the node limit disappears and the top-8 can land on any node.
# 3. Set gamma = 0.0 in part 5 from step 0 (V3 only did this for its last 500B tokens): the skew never gets corrected.
