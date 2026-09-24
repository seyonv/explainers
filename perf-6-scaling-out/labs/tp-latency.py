"""tp-latency.py: Megatron II in numbers, then what tensor parallelism costs a 70B decode step.

Part A checks Megatron's two vocab-parallel tricks in numpy (split result == unsplit result):
  1. the vocab-parallel input embedding (masked local lookup + one all-reduce)
  2. the fused vocab-parallel cross-entropy (only b*s-sized tensors cross GPUs, never b*s*v logits)
Part B redoes the paper's arithmetic (vocab padding, 76% vs 74%, strong scaling, bandwidths).
Part C models one Llama-3.1-70B BF16 decode step on H100s:
    step = weight read / TP  +  160 all-reduces x (alpha_AR + 2(n-1)/n * B*D*2 bytes / W_link)
alpha_AR is the fixed cost of one all-reduce. The course's shared ILLUSTRATIVE value (see collectives.html):
    alpha_hop = 1 us per ring hop (Scaling Book's TPU ICI figure), alpha_AR = 2(n-1) * alpha_hop inside a node
    (2 / 6 / 14 us at TP2 / TP4 / TP8), and 2 x the TP8 value (28 us) for an all-reduce that crosses nodes.
NVIDIA publishes no NVLink/IB collective latency. Change them below or on the command line:
    python3 labs/tp-latency.py [alpha_hop_us] [cross_node_factor]
KV-cache reads and FLOPs are ignored (weights-only step, as in the course's shared table).
Stdlib + numpy, CPU, well under a second.
"""
import sys
import numpy as np

ALPHA_HOP = 1e-6       # s per ring hop inside one NVLink node (ILLUSTRATIVE, course shared value)
CROSS_NODE = 2.0       # cross-node all-reduce latency = CROSS_NODE x the in-node TP8 value (ILLUSTRATIVE)
if len(sys.argv) > 1: ALPHA_HOP = float(sys.argv[1]) * 1e-6
if len(sys.argv) > 2: CROSS_NODE = float(sys.argv[2])


def alpha_in_node(n):
    return 2 * (n - 1) * ALPHA_HOP


ALPHA_IB = CROSS_NODE * alpha_in_node(8)   # 28 us with the defaults

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Part A: numpy checks
print("== A. Megatron's vocab-parallel tricks, checked in numpy ==")
n, v, h, b, s = 4, 1000, 64, 2, 8                 # 4 'GPUs', toy vocab and hidden size
E = rng.standard_normal((v, h)).astype(np.float64)  # embedding table, rows = tokens
tok = rng.integers(0, v, size=(b, s))
shard = v // n
# input embedding: GPU r holds rows [r*shard, (r+1)*shard); looks up only its tokens, zeros elsewhere
partial = []
for r in range(n):
    lo, hi = r * shard, (r + 1) * shard
    mask = (tok >= lo) & (tok < hi)
    local = np.zeros((b, s, h))
    local[mask] = E[lo:hi][tok[mask] - lo]
    partial.append(local)
emb_tp = sum(partial)                              # the all-reduce (g operator)
print(f"  embedding: split lookup + all-reduce == full lookup: {np.allclose(emb_tp, E[tok])}")

# output layer (weights tied to E): logits = X @ E.T, sharded by vocab
X = rng.standard_normal((b, s, h))
target = rng.integers(0, v, size=(b, s))
logits = X @ E.T
full_ce = (np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1)) + logits.max(-1)
           - np.take_along_axis(logits, target[..., None], -1)[..., 0])
Y = [X @ E[r * shard:(r + 1) * shard].T for r in range(n)]              # [Y1..Yn], never gathered
gmax = np.max([y.max(-1) for y in Y], axis=0)                            # all-reduce(max), b*s values
sumexp = sum(np.exp(y - gmax[..., None]).sum(-1) for y in Y)             # all-reduce(sum), b*s values
tlogit = sum(np.where((target >= r * shard) & (target < (r + 1) * shard),
                      np.take_along_axis(Y[r], np.clip(target - r * shard, 0, shard - 1)[..., None], -1)[..., 0], 0.0)
             for r in range(n))                                          # all-reduce(sum), b*s values
fused_ce = np.log(sumexp) + gmax - tlogit
print(f"  fused cross-entropy == full cross-entropy: {np.allclose(fused_ce, full_ce)}"
      f"  (max abs diff {np.abs(fused_ce - full_ce).max():.1e})")
print(f"  toy sizes: all-gather of logits moves b*s*v = {b*s*v:,} values; fused version 3 x b*s = {3*b*s} values")

# ---------------------------------------------------------------- Part B: the paper's numbers
print("\n== B. Megatron-LM arithmetic (V100 DGX-2H, as the paper gives it) ==")
V_GPT2, MULT = 50257, 128 * 8
pad = -(-V_GPT2 // MULT) * MULT
print(f"  vocab padding: 50,257 -> {pad:,} (multiple of 128 x 8 = {MULT}); +{pad - V_GPT2} rows "
      f"= {100 * (pad - V_GPT2) / V_GPT2:.1f}%; per GPU at 8-way: {pad // 8:,} = {pad // 8 // 128} x 128")
V_LLAMA = 128256
pad_l = -(-V_LLAMA // MULT) * MULT
print(f"  same rule on Llama-3.1's 128,256: /8 = {V_LLAMA / 8:,.0f} per GPU = {V_LLAMA / 8 / 128:.2f} x 128 "
      f"-> would pad to {pad_l:,} (+{pad_l - V_LLAMA})")
bb, ss = 8, 1024
print(f"  logits all-gather, b={bb}, s={ss}, v={pad:,}: b*s*v = {bb*ss*pad:,} values "
      f"= {bb*ss*pad*2/1e6:,.0f} MB in fp16; fused: b*s = {bb*ss:,} values; ratio = v = {pad:,}x")
print(f"  abstract: 15.1 PF / (512 x 39 TF = {512*39/1000:.2f} PF) = {15.1e3/(512*39):.1%}  (the '76%')")
print(f"  Fig 5 bar: 74% x 512 x 39 TF = {0.74*512*39/1000:.2f} PF  (the '74%', a different number)")
print(f"  39 TF is '30% of peak' -> implied peak {39/0.30:.0f} TF per V100 (our arithmetic)")
print(f"  8-way MP group at 77%: 8 x 39 x 0.77 = {8*39*0.77:.0f} TF")
for g, sp in [(2, 1.64), (4, 2.34), (8, 2.98)]:
    print(f"  strong scaling (Table 8) {g} GPUs: {sp}x -> {sp/g:.0%} per-GPU efficiency")
print(f"  bandwidth: NVSwitch 300 GB/s per GPU vs 100 GB/s per 16-GPU server = {100/16:.2f} GB/s per GPU "
      f"-> {300/(100/16):.0f}x (our arithmetic)")
print(f"  B.1 groups: 8-way model parallel x 64-way data parallel = {8*64} GPUs")

# ---------------------------------------------------------------- Part C: 70B decode step vs TP
print("\n== C. Llama-3.1-70B BF16 decode step on H100 (our model) ==")
PARAMS = 70_553_706_496   # Llama-3.1-70B, from the Meta config (course shared facts: 70.55B)
WBYTES = PARAMS * 2            # 141.1 GB
HBM = 3.35e12                  # B/s
D, L = 8192, 80
N_AR = 2 * L                   # 160 all-reduces per decode step (our inference from Megatron)
W_NV, W_IB = 450e9, 50e9       # per direction, per GPU
print(f"  alpha per all-reduce (ILLUSTRATIVE): {ALPHA_HOP*1e6:g} us/hop -> in node TP2 {alpha_in_node(2)*1e6:g}, "
      f"TP4 {alpha_in_node(4)*1e6:g}, TP8 {alpha_in_node(8)*1e6:g} us; across nodes {ALPHA_IB*1e6:g} us")


def weight_ms(tp):
    return WBYTES / tp / HBM * 1e3


def ar_bw_s(tp, B, W):
    """ring all-reduce bandwidth term for a B x D bf16 message"""
    msg = B * D * 2
    return 2 * (tp - 1) / tp * msg / W


def ar_16_two_nodes_s(B):
    """TP16 = 2 nodes x 8: reduce-scatter in node, all-reduce of S/8 over each GPU's own NIC, all-gather in node"""
    msg = B * D * 2
    return 2 * 7 / 8 * msg / W_NV + 2 * (1 / 2) * (msg / 8) / W_IB


def step(tp, B, how):
    if tp == 1:
        return weight_ms(1), 0.0, 0.0
    if how == "nv":
        lat, bw = alpha_in_node(tp), ar_bw_s(tp, B, W_NV)
    elif how == "ib-flat":
        lat, bw = ALPHA_IB, ar_bw_s(tp, B, W_IB)
    else:  # "2node"
        lat, bw = ALPHA_IB, ar_16_two_nodes_s(B)
    return weight_ms(tp), N_AR * lat * 1e3, N_AR * bw * 1e3


BATCHES = (1, 64, 256)
print("  message per all-reduce: " + ", ".join(f"B={B}: {B*D*2/1e3:,.0f} kB" if B == 1 else f"B={B}: {B*D*2/1e6:.2f} MB" for B in BATCHES))
for B in BATCHES:
    print(f"  bandwidth term per all-reduce, TP8 NVLink, B={B}: {ar_bw_s(8, B, W_NV)*1e6:.2f} us "
          f"-> x160 = {N_AR*ar_bw_s(8, B, W_NV)*1e3:.2f} ms")
print(f"  compute check, B=256: FLOPs time {2*PARAMS*256/989e12*1e3:.1f} ms/TP vs weight read {weight_ms(1):.2f} ms/TP "
      f"(below the weight read, so ignoring FLOPs is fair up to B=256)")
print()
print("  layout                 B   weights  latency  bandwidth   step   comm%   tok/s/user  tok/s/GPU")
rows = [(1, "nv", "TP1 (won't fit)"), (2, "nv", "TP2 NVLink"), (4, "nv", "TP4 NVLink"),
        (8, "nv", "TP8 NVLink"), (8, "ib-flat", "TP8 over IB (hypoth.)"), (16, "2node", "TP16 = 2 nodes")]
table = {}
for tp, how, name in rows:
    for B in BATCHES:
        w, lat, bw = step(tp, B, how)
        tot = w + lat + bw
        table[(tp, how, B)] = tot
        print(f"  {name:22s} {B:3d}  {w:6.2f}   {lat:6.2f}   {bw:7.2f}   {tot:6.2f}  {100*(lat+bw)/tot:4.0f}%   "
              f"{1e3/tot:8.1f}   {B*1e3/tot/tp:8.1f}")
print()
for B in BATCHES:
    t8 = table[(8, "nv", B)]
    w16 = weight_ms(16)
    bw16 = N_AR * ar_16_two_nodes_s(B) * 1e3
    be = (t8 - w16 - bw16) / N_AR * 1e3
    print(f"  B={B:3d}: TP8 NVLink {t8:.2f} ms vs TP16 two nodes {table[(16, '2node', B)]:.2f} ms; "
          f"TP16 wins only if alpha_IB < {be:.1f} us")
print(f"  going TP1->2 saves {weight_ms(1)-weight_ms(2):.2f} ms of weight read; TP4->8 saves {weight_ms(4)-weight_ms(8):.2f}; "
      f"TP8->16 saves {weight_ms(8)-weight_ms(16):.2f}; the 160 all-reduces never shrink with TP")

print("\n  other TP traffic per step (once, not x160):")
V = 128256
print(f"    input embedding (vocab-parallel): +1 all-reduce of B*D; gathering full logits at B=256: "
      f"{256*V*2/1e6:.1f} MB -> all-gather {7/8*256*V*2/W_NV*1e6:.0f} us over NVLink")

# ---------------------------------------------------------------- Scaling Book limits
print("\n== D. Scaling Book limits, H100 numbers ==")
C = 990e12
for F, lab in [(28000, "book's F=28,000"), (28672, "Llama-3.1-70B F=28,672")]:
    print(f"  compute-bound TP limit, {lab}: F/2200 = {F/2200:.1f}-way in node, F/2475 = {F/2475:.1f}-way beyond")
print(f"  C/W = 990e12/450e9 = {C/W_NV:.0f};  990e12/400e9 = {C/400e9:.0f}")
for W, lab in [(W_NV, "NVLink 450 GB/s"), (W_IB, "IB 50 GB/s/GPU")]:
    beta = HBM / W
    lims = ", ".join(f"B={B}: {28672/(B*beta):,.0f}" for B in BATCHES)
    print(f"  generation limit Y < F/(B*beta), beta = 3.35 TB/s / {lab} = {beta:.1f}: {lims}")

# Try this:
# 1. python3 labs/tp-latency.py 1 1    -> cross-node all-reduces no slower than in-node ones: does TP16 over two nodes win?
# 2. python3 labs/tp-latency.py 3 2    -> a slower collective stack (3 us/hop): at B=1 the latency term dominates TP8.
# 3. Set W_NV = 370e9 (the Scaling Book's best measured 8xH100 busbw) and see how little the B=256 row moves.
