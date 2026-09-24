"""afd-util.py - why MoE experts starve at decode, and how many attention replicas feed them.

Course 6 card: "MegaScale-Infer I: splitting attention from the experts" (arXiv 2504.02263, §2.3, §3, §4.2-4.3).
A calculator: stdlib + numpy, CPU, well under a second. Nothing here is measured.

  1. §2.3 roofline, as the paper writes it: b >= F/B (F = FLOPS, B = memory BANDWIDTH, not batch).
     A100 312 TFLOPS / 2 TB/s = 156; Mixtral 8x22B at b = 156 -> 39 tokens per expert, 25% MFU.
  2. The same on the H100 (our recomputation): 295, Mixtral 1,180, DeepSeek-V3 9,440.
  3. FFN utilization vs batch: dense min((B/F)*b, 1) vs MoE min((topk/#expert)*(B/F)*b, 1).
  4. Disaggregated sizing: b_a * m * n_a = b_e * m * E / K = B (now B = global batch!), so the
     attention nodes needed for each expert to reach b_e >= F/B, and the balance rule n_a = k1*E/(k3*K).
  5. §4.3 Table 3 per-cost numbers recomputed, plus each GPU's ops:byte (our recomputation).
  6. What "one expert per expert node" holds, for coarse vs fine-grained MoE (our recomputation).

Run:  python3 labs/afd-util.py
"""
import math
import numpy as np

# ---------------------------------------------------------------------------
# Hardware (paper §2.3 for the A100; course facts for the H100)
# ---------------------------------------------------------------------------
A100 = dict(name="A100-SXM-80GB", F=312e12, B=2e12)       # paper §2.3
H100 = dict(name="H100 SXM", F=989e12, B=3.35e12)          # course facts (dense BF16, HBM3)

# MoE shapes: (topk, #experts). Mixtral 8x22B: paper §2.2 / Table 4. DeepSeek-V3: course facts.
MIXTRAL = dict(name="Mixtral 8x22B", K=2, E=8)
DSV3 = dict(name="DeepSeek-V3", K=8, E=256)


def ridge(hw):
    """b >= F/B: the batch (tokens) at which a bf16 GEMM stops being memory-bound.
    2*b*h*n FLOPs / F  >=  2*h*n bytes / B   ->   b >= F / B."""
    return hw["F"] / hw["B"]


def util_dense(b, hw):
    return np.minimum(b * hw["B"] / hw["F"], 1.0)


def util_moe(b, hw, K, E):
    return np.minimum((K / E) * b * hw["B"] / hw["F"], 1.0)


def section_roofline():
    print("== 1. §2.3 roofline, paper's numbers (A100, bf16) ==")
    r = ridge(A100)
    print(f"F/B = 312 TFLOPS / 2 TB/s                       = {r:.0f} tokens")
    b = 156
    per_expert = b * MIXTRAL["K"] / MIXTRAL["E"]
    print(f"Mixtral 8x22B at b = {b}: tokens per expert = {b} x 2/8 = {per_expert:.0f}")
    print(f"theoretical FFN MFU = min((2/8) x (B/F) x {b}, 1)  = {util_moe(b, A100, 2, 8):.0%}   (paper: 2/8 = 25%)")
    print()
    print("== 2. The same on an H100 (our recomputation) ==")
    rh = ridge(H100)
    print(f"F/B = 989 TFLOPS / 3.35 TB/s                    = {rh:.1f} -> {round(rh)} tokens")
    for m in (MIXTRAL, DSV3):
        need = round(rh) * m["E"] / m["K"]
        print(f"{m['name']:<14} top-{m['K']} of {m['E']:>3}: batch for experts to be compute-bound "
              f"= 295 x {m['E']}/{m['K']} = {need:,.0f} tokens;  util at b = 295: {util_moe(295, H100, m['K'], m['E']):.1%}")
    print()


def section_util_table():
    print("== 3. FFN utilization vs decode batch b (theoretical, formula only) ==")
    bs = [32, 64, 128, 156, 256, 295, 512, 1024, 1180, 4096, 9440]
    print(f"{'b':>6} | {'A100 dense':>10} {'A100 Mixtral':>12} | {'H100 dense':>10} {'H100 Mixtral':>12} {'H100 DS-V3':>10}")
    for b in bs:
        print(f"{b:>6} | {util_dense(b, A100):>10.1%} {util_moe(b, A100, 2, 8):>12.1%} | "
              f"{util_dense(b, H100):>10.1%} {util_moe(b, H100, 2, 8):>12.1%} {util_moe(b, H100, 8, 256):>10.1%}")
    print()


def attention_nodes_needed(target_be, b_a, K, E):
    """b_a * n_a = b_e * E / K  ->  n_a = target_be * E / (K * b_a)   (per micro-batch; m cancels)."""
    return math.ceil(target_be * E / (K * b_a))


def section_sizing():
    print("== 4. Disaggregated sizing: b_a * m * n_a = b_e * m * E / K = B  (§4.2; B = GLOBAL BATCH here) ==")
    m = 3          # Alg 1 starts its search at m = 3 micro-batches
    b_a = 128      # micro-batch per attention node: the paper's §7.3 example payload uses 128 for Mixtral
    for label, hw, model in (("Mixtral 8x22B on A100 (paper's roofline)", A100, MIXTRAL),
                             ("DeepSeek-V3 on H100 (our recomputation)", H100, DSV3)):
        target = round(ridge(hw))
        K, E = model["K"], model["E"]
        n_a = attention_nodes_needed(target, b_a, K, E)
        b_e = b_a * n_a * K / E
        B = b_a * m * n_a
        print(f"{label}: target b_e >= {target}, K = {K}, E = {E}, b_a = {b_a}, m = {m}")
        print(f"   n_a = ceil({target} x {E} / ({K} x {b_a})) = ceil({target * E / (K * b_a):.2f}) = {n_a} attention nodes")
        print(f"   b_e = {b_a} x {n_a} x {K}/{E} = {b_e:.0f} tokens per expert per micro-batch -> util {util_dense(b_e, hw):.0%}")
        print(f"   B   = b_a x m x n_a = {b_a} x {m} x {n_a} = {B:,} tokens in flight; check b_e x m x E/K = {b_e * m * E / K:,.0f}")
        one_replica = util_moe(b_a * m, hw, K, E)
        print(f"   same attention node alone (colocated, batch {b_a * m}): expert util {one_replica:.1%}")
    print()
    print("Balance rule (§4.2): T_a = k1*b_a + k2, T_e = k3*b_e + k4; set n_a = k1*E/(k3*K) so T_a ~ T_e.")
    for ratio in (0.5, 1.0, 2.0):
        n_a = ratio * MIXTRAL["E"] / MIXTRAL["K"]
        print(f"   illustrative k1/k3 = {ratio}: Mixtral n_a = {ratio} x 8/2 = {n_a:g} attention nodes per 8 expert nodes")
    print()


# §4.3 Table 3, prices normalized to L20 = 1.00: (price, capacity GB, bandwidth GB/s, TFLOPS)
TABLE3 = {
    "L20":  (1.00, 48, 864, 119.5),
    "H800": (5.28, 80, 3430.4, 989),
    "A800": (2.26, 80, 2039, 312),
    "H20":  (1.85, 96, 4096, 148),
    "L40S": (1.08, 48, 864, 362),
}


def section_table3():
    print("== 5. §4.3 Table 3 per-cost columns, recomputed; plus ops:byte (our recomputation) ==")
    print(f"{'GPU':<5} {'price':>5} | {'GB/cost':>7} {'GB/s/cost':>9} {'TFLOPS/cost':>11} | {'ops:byte = F/B':>14}")
    for g, (p, cap, bw, tf) in TABLE3.items():
        print(f"{g:<5} {p:>5.2f} | {cap / p:>7.1f} {bw / p:>9.1f} {tf / p:>11.1f} | {tf * 1e12 / (bw * 1e9):>14.0f}")
    best_gb = max(TABLE3, key=lambda g: TABLE3[g][1] / TABLE3[g][0])
    best_bw = max(TABLE3, key=lambda g: TABLE3[g][2] / TABLE3[g][0])
    best_tf = max(TABLE3, key=lambda g: TABLE3[g][3] / TABLE3[g][0])
    print(f"best capacity per cost: {best_gb}; best bandwidth per cost: {best_bw}; best TFLOPS per cost: {best_tf}")
    l40s_ridge = 362e12 / 864e9
    n_a = attention_nodes_needed(round(l40s_ridge), 128, 2, 8)
    print(f"L40S as an expert node: needs b_e >= 362/0.864 = {l40s_ridge:.0f} tokens per expert "
          f"-> Mixtral n_a = ceil({round(l40s_ridge)} x 8 / (2 x 128)) = {n_a} attention nodes at b_a = 128")
    print()


def section_expert_node():
    print("== 6. What one expert node stores: expert i of every MoE layer (our recomputation, SwiGLU = 3 matrices) ==")
    mix = 3 * 6144 * 16384 * 56           # Table 4: hidden 6144, intermediate 16384, 56 layers
    ds_one = 3 * 7168 * 2048              # course facts: hidden 7168, expert width 2048
    ds = ds_one * 58                      # 58 MoE layers
    print(f"Mixtral 8x22B: 3 x 6144 x 16384 x 56 = {mix / 1e9:.2f}B params = {mix * 2 / 1e9:.1f} GB bf16 per expert node")
    print(f"DeepSeek-V3  : 3 x 7168 x 2048 = {ds_one / 1e6:.1f}M per layer x 58 = {ds / 1e9:.2f}B params "
          f"= {ds / 1e9:.2f} GB FP8 per expert node (x256 nodes)")
    print()


if __name__ == "__main__":
    section_roofline()
    section_util_table()
    section_sizing()
    section_table3()
    section_expert_node()

# Try this:
#  1. Set b_a = 64 in section_sizing: DeepSeek-V3 on H100 then needs 148 attention nodes to feed 256 experts.
#  2. Use FP8 weights with BF16 math on the H100 (bytes per weight halve): replace H100["B"] with 2 * 3.35e12;
#     the ridge drops to ~148 and DeepSeek-V3 needs about 4,700 tokens instead of 9,440.
#  3. Swap DSV3 for dict(name="DBRX", K=4, E=16) (paper Table 4): tokens per expert are b/4, like Mixtral.
