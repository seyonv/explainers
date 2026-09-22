"""Attention intensity lab: Scaling Book Part 7 "What about attention?" + Part 4 Q4 (GQA).

Run: python3 labs/attention-intensity.py   (stdlib only, CPU, well under a second)

One Flash-Attention fusion, bf16, per layer:
  read Q [B,T,N*H], read K and V [B,S,K*H], write O [B,T,N*H]
  FLOPs = 2BTSNH (QK^T) + 2BTSNH (AV) = 4BTSNH        (softmax/mask FLOPs ignored, as the book does)
  bytes = 2*BTNH (Q) + 2*BTNH (O) + 2*2*BSKH (K, V) = 4BTNH + 4BSKH
  intensity = TSG / (TG + S), G = N/K.  G = 1 (MHA) gives the book's ST/(S+T).
"""

CHIPS = {
    "TPU v5e": dict(flops=1.97e14, bw=8.2e11),   # book, Part 1/7
    "H100":    dict(flops=9.89e14, bw=3.35e12),  # dense bf16, _facts.md
}
# Llama-3.1-8B (running example, _facts.md)
L, D, F, N, K, H = 32, 4096, 14336, 32, 8, 128
WEIGHT_BYTES = 16.06e9
KV_PER_TOKEN = 2 * L * K * H * 2  # K and V, bf16 -> 131072 B
FREE_HBM = 80e9 - WEIGHT_BYTES


def ridge(c):
    return c["flops"] / c["bw"]


def attn(B, T, S, n=N, k=K, h=H):
    """FLOPs and HBM bytes for one layer of attention."""
    flops = 4 * B * T * S * n * h
    nbytes = 4 * B * T * n * h + 4 * B * S * k * h
    return flops, nbytes


def intensity(T, S, G):
    return T * S * G / (T * G + S)


def bound(I):
    return "  ".join(f"{name}: {'compute' if I > ridge(c) else 'memory'}" for name, c in CHIPS.items())


print("== Hardware ridge (FLOPs/byte) ==")
for name, c in CHIPS.items():
    print(f"{name:8s} {ridge(c):6.1f}")

print("\n== MHA (G=1), book formula ST/(S+T) ==")
print("prefill, S = T -> T/2")
for T in [128, 480, 590, 4096]:
    I = intensity(T, T, 1)
    print(f"  T=S={T:6d}: intensity {I:8.2f}   {bound(I)}")
for name, c in CHIPS.items():
    print(f"  prefill crossover {name}: T/2 = {ridge(c):.1f} -> T = {2 * ridge(c):.0f} tokens")
print("generation, T = 1 -> S/(S+1) ~ 1")
for S in [1024, 8192, 131072]:
    I = intensity(1, S, 1)
    print(f"  S={S:7d}: intensity {I:.5f}   {bound(I)}")

print("\n== Check with explicit bytes, one layer, B=1, as if Llama-3.1-8B were MHA (K=N=32) ==")
for label, T, S in [("prefill T=S=4096", 4096, 4096), ("decode T=1, S=8192", 1, 8192)]:
    fl, by = attn(1, T, S, k=N)
    q = 2 * T * N * H
    kv = 4 * S * N * H
    print(f"  {label}: Q read {q / 1e6:.3f} MB, KV read {kv / 1e6:.2f} MB, out {q / 1e6:.3f} MB, "
          f"FLOPs {fl:.3e}, intensity {fl / by:.4f}")

print("\n== GQA (Part 4 Q4): intensity = TSG/(TG+S) ==")
print("prefill S=T -> TG/(G+1);  generation T=1 -> SG/(G+S) -> G")
for G in [1, 4, 8, 32]:
    gen = [intensity(1, S, G) for S in (1024, 8192, 131072)]
    cross = {name: ridge(c) * (G + 1) / G for name, c in CHIPS.items()}
    print(f"  G={G:2d}: gen S=1k {gen[0]:6.3f}  S=8k {gen[1]:6.3f}  S=128k {gen[2]:6.3f}   "
          f"prefill crossover T: v5e {cross['TPU v5e']:.0f}, H100 {cross['H100']:.0f}")

print("\n== Llama-3.1-8B, G = N/K = 4, decode at batch 64, context 8192, whole model (32 layers) ==")
B, S = 64, 8192
fl, by = attn(B, 1, S)
fl, by = fl * L, by * L
h = CHIPS["H100"]
print(f"  attention FLOPs {fl:.3e}  bytes {by / 1e9:.2f} GB  intensity {fl / by:.3f} (H100 ridge {ridge(h):.0f})")
print(f"  H100: T_math {fl / h['flops'] * 1e3:.3f} ms   T_comms {by / h['bw'] * 1e3:.2f} ms")

print("\n== Bytes read per decode step: weights vs KV cache (H100, Llama-3.1-8B bf16) ==")
print(f"KV per token {KV_PER_TOKEN} B; free HBM after weights {FREE_HBM / 1e9:.2f} GB")
for ctx in [2048, 8192]:
    kv_seq = ctx * KV_PER_TOKEN
    print(f"context {ctx}: KV per sequence {kv_seq / 1e9:.3f} GB; KV = weights at batch {WEIGHT_BYTES / kv_seq:.1f}; "
          f"max batch that fits {int(FREE_HBM // kv_seq)}")
    for Bs in [1, 8, 16, 32, 64, 128, 256]:
        kv = Bs * kv_seq
        tot = kv + WEIGHT_BYTES
        fits = "fits" if kv <= FREE_HBM else "DOES NOT FIT in 80 GB"
        print(f"  B={Bs:4d}: weights {WEIGHT_BYTES / 1e9:6.2f} GB  KV {kv / 1e9:7.2f} GB  "
              f"KV share {kv / tot * 100:5.1f}%  min step {tot / h['bw'] * 1e3:6.2f} ms  ({fits})")

print("\n== Book's rough ratio: param bytes / KV bytes per sequence ~ 2DF/(SHK) ==")
for S in [2048, 8192]:
    approx = 2 * D * F / (S * H * K)
    exact = WEIGHT_BYTES / (S * KV_PER_TOKEN)
    print(f"  S={S}: 2DF/(SHK) = {approx:.2f}   exact 16.06 GB / KV(seq) = {exact:.2f}")

# Try this:
# 1. MQA: set K = 1 (G = 32) -> decode intensity -> ~32 and KV per token drops 8x (16 KiB);
#    still far below 295, but the batch where KV equals weights moves from ~15 to ~120 at 8k.
# 2. 128k context: add 131072 to the context loop -> one sequence's KV is 17.2 GB, more than the
#    weights; even batch 1 is KV-dominated and only 3 sequences fit.
# 3. Sweep G in intensity(1, S, G) up to 128 and see the decode intensity stay pinned at ~G.
