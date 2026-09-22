"""Sharding prefill vs generation lab: Scaling Book Part 7, "Distributing Inference
Over Multiple Accelerators" -> Prefill, Generation.

Run: python3 labs/sharding-prefill-generation.py   (numpy, CPU, well under a second)

(a) Megatron-style model (tensor) parallelism for an MLP: split W_in / W_gate by
    columns and W_out by rows over 4 simulated devices; each device computes a
    partial output and a sum (the all-reduce) gives back the unsharded result.
(b) When does sharding the weights stop helping in decode? The book compares
        T_HBM = 2DF / (Y * W_hbm)   (each chip loads 1/Y of the weights)
        T_ICI = 2BD / W_ici         (activations moved between chips)
    and gets: comms dominate when Y > F / (B * beta), beta = W_hbm / W_ici.
"""
import numpy as np

# ---------------------------------------------------------------- (a) numerics
print("== (a) Column-split then row-split MLP on 4 simulated devices ==")
rng = np.random.default_rng(0)
B, D, F, Y = 8, 64, 256, 4          # small shapes, same structure as Llama's MLP
X = rng.standard_normal((B, D))
W_gate = rng.standard_normal((D, F)) / np.sqrt(D)
W_up = rng.standard_normal((D, F)) / np.sqrt(D)
W_out = rng.standard_normal((F, D)) / np.sqrt(F)


def silu(z):
    return z / (1 + np.exp(-z))


def mlp(x, wg, wu, wo):              # SwiGLU, as in Llama
    return (silu(x @ wg) * (x @ wu)) @ wo


full = mlp(X, W_gate, W_up, W_out)
cols = np.split(np.arange(F), Y)     # device i owns hidden units cols[i]
partials = [mlp(X, W_gate[:, c], W_up[:, c], W_out[c, :]) for c in cols]
sharded = sum(partials)              # the all-reduce: sum the [B, D] partials
print(f"B={B} D={D} F={F}, {Y} devices; each owns F/Y = {F // Y} hidden units")
print(f"each device holds W_gate[:, {F//Y}], W_up[:, {F//Y}], W_out[{F//Y}, :] = 1/{Y} of the weights")
print(f"max |sharded - unsharded| = {np.abs(sharded - full).max():.2e}  -> equal: {np.allclose(sharded, full)}")
print(f"only thing crossing chips: {Y} partial outputs of shape [{B}, {D}] (activations), never weights")
wrong = mlp(X, W_gate[:, cols[0]], W_up[:, cols[0]], W_out[cols[0], :])
print(f"(skip the all-reduce and device 0 is off by {np.abs(wrong - full).max():.2f})")

# ------------------------------------------------ (b1) the book's v5e example
print("\n== (b1) Book example, TPU v5e: F = 16384, B = 32, beta ~ 8 ==")
F_b, B_b, beta_b = 16384, 32, 8
print(f"Prefill / training rule (one axis): Y < F / 2200 = {F_b / 2200:.1f}  (book: 'usually 4-8 way')")
print(f"Generation rule: Y* = F / (B * beta) = {F_b} / ({B_b} * {beta_b}) = {F_b / (B_b * beta_b):.0f}")
print("T_ICI / T_HBM = Y * B * beta / F  (D cancels)")
print(f"{'Y':>5} {'T_ICI/T_HBM':>12}  bound")
for y in (8, 16, 32, 64, 128):
    r = y * B_b * beta_b / F_b
    print(f"{y:5d} {r:12.3f}  {'ICI (comms)' if r > 1 else ('tie' if r == 1 else 'HBM (weights)')}")
print(f"our check of beta: v5e W_hbm 8.2e11 / (2 x 4.5e10 bidirectional ICI) = {8.2e11 / 9e10:.1f}")
print(f"our check of 2200: v5e C 1.97e14 / 9e10 = {1.97e14 / 9e10:.0f}")

# --------------------------------------- (b2) our recomputation: 8x H100 node
print("\n== (b2) Our recomputation: Llama-3.1-8B on an 8x H100 NVLink node ==")
W_HBM, C = 3.35e12, 9.89e14
NVL = {"450 (theory)": 450e9, "370 (achievable)": 370e9}   # book Part 12
L, Dm, Fm = 32, 4096, 14336
P, WBYTES = 8.03e9, 16.06e9
for name, w in NVL.items():
    print(f"beta = W_hbm / W_nvlink = 3.35e12 / {w:.3g} = {W_HBM / w:.2f}   [{name} GB/s]")
print(f"TP rule for training/prefill (book Part 12): Y < F / 2200 = {Fm / 2200:.1f}")

print(f"8-way TP: each GPU holds 16.06 GB / 8 = {WBYTES / 8 / 1e9:.2f} GB of weights")
print("FSDP in decode (why not): every step all-gathers the weights over NVLink instead of reading HBM")
print(f"  local HBM read, 1 GPU: 16.06e9 / 3.35e12      = {WBYTES / W_HBM * 1e3:.1f} ms")
for name, w in NVL.items():
    print(f"  8-way FSDP all-gather: 16.06e9 * 7/8 / {w:.3g} = {WBYTES * 7 / 8 / w * 1e3:.1f} ms  [{name}]")
kv_w = 2 * Dm * 1024 * 2          # W_K + W_V: [4096, 8 heads * 128], bf16
qo_w = 2 * Dm * Dm * 2            # W_Q + W_O: [4096, 32 heads * 128], bf16
print(f"attention weights per layer: W_Q+W_O = {qo_w / 1e6:.1f} MB, W_K+W_V = {kv_w / 1e6:.1f} MB "
      f"(ratio {kv_w / qo_w:.2f}); all layers W_K+W_V = {kv_w * L / 1e6:.0f} MB")

print("\nBook's per-matrix bound Y* = F / (B * beta), F = 14336:")
print(f"{'B':>5} {'Y* @450':>9} {'Y* @370':>9}")
for b in (1, 32, 256):
    print(f"{b:5d} {Fm / (b * W_HBM / 450e9):9.1f} {Fm / (b * W_HBM / 370e9):9.1f}")


def allreduce(nbytes, y, w):
    """Bandwidth-only AllReduce in one node: 2x the AllGather cost B(Y-1)/(Y W)."""
    return 0.0 if y == 1 else 2 * nbytes * (y - 1) / (y * w)


print("\nWhole decode step (all 32 layers), per GPU. Megatron: 2 all-reduces/layer")
print("(after attention W_O and after MLP W_out), each of bf16[B, D] = 2*B*4096 bytes.")
print("T_weights = 16.06 GB / (Y * 3.35 TB/s); T_math = 2 * 8.03e9 * B / (Y * 989 TF);")
print("T_comm = 64 * 2 * (2*B*D) * (Y-1)/Y / W_nvlink.  Latency per collective NOT modelled.")
hdr = f"{'B':>4} {'TP':>3} {'T_weights':>10} {'T_math':>9} {'T_comm@450':>11} {'T_comm@370':>11}  comms > weights?"
print(hdr)
for b in (1, 32, 256):
    for y in (1, 2, 4, 8):
        tw = WBYTES / (y * W_HBM)
        tm = 2 * P * b / (y * C)
        act = 2 * b * Dm
        t450 = 2 * L * allreduce(act, y, 450e9)
        t370 = 2 * L * allreduce(act, y, 370e9)
        flag = "yes @370" if t370 > tw else "no"
        if t450 > tw:
            flag = "yes @450 and @370"
        print(f"{b:4d} {y:3d} {tw*1e6:8.0f}us {tm*1e6:7.0f}us {t450*1e6:9.1f}us {t370*1e6:9.1f}us  {flag}")
    print()

# Try this:
#   1. Add B = 512 to the batch list: at TP=8, T_comm@450 = 1044 us beats T_weights = 599 us,
#      and T_math (8.3 ms at TP=1 -> 1.04 ms at TP=8) > T_weights: now compute-bound, so the
#      training rule Y < F / 2200 (= 6.5 here) is the one that applies.
#   2. Set Fm = 28672 (Llama-3-70B's MLP) in the Y* table: Y* doubles (B=32 -> 120, B=256 -> 15
#      at 450 GB/s), so 8-way TP stays weight-load-bound to twice the batch. (70B also simply
#      needs > 1 GPU: ~140 GB of bf16 weights vs 80 GB of HBM.)
#   3. In (a), set Y = 8 or Y = 3 (with F = 255): any split of the hidden units works,
#      as long as the gate, up and out slices use the same indices.
