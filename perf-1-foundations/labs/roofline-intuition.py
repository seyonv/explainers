"""Roofline intuition: which resource limits an operation, and how long it takes.

T ~= max(FLOPs / C, bytes / W)      intensity I = FLOPs / bytes      ridge I* = C / W

Runs with: python3 labs/roofline-intuition.py   (stdlib only, < 1 s)
"""
import math

CHIPS = {
    # name: (peak dense FLOP/s C, HBM bandwidth W in B/s, HBM bytes)
    "H100 SXM (BF16)": (989e12, 3.35e12, 80e9),   # NVIDIA datasheet; Scaling Book Part 12
    "TPU v5e (BF16)": (1.97e14, 8.2e11, 16e9),    # Scaling Book Part 1
}

BYTES = 2  # BF16
D, F = 4096, 14336          # Llama-3.1-8B hidden size and MLP width
N_PARAMS = 8.03e9           # Llama-3.1-8B total parameters (_facts.md)


def matmul(B, D, F, b=BYTES):
    """X[B,D] @ W[D,F] -> Z[B,F]: read X and W, write Z (Scaling Book Part 1)."""
    flops = 2 * B * D * F
    nbytes = b * (B * D + D * F + B * F)
    return flops, nbytes


def ops():
    n = 1_000_000
    yield "(a) vector add, 1M BF16 elements", n, BYTES * 3 * n  # read x, y; write z
    yield "(b) MLP matmul 4096x14336, 1 token", *matmul(1, D, F)
    yield "(c) MLP matmul 4096x14336, 512 tokens", *matmul(512, D, F)
    # whole decode step at batch 1: every weight read once, 2 FLOPs per weight;
    # KV cache and activation traffic ignored (short context)
    yield "(d) whole decode step, batch 1", 2 * N_PARAMS, BYTES * N_PARAMS


def fmt_t(s):
    if s >= 1e-3:
        return f"{s * 1e3:8.3f} ms"
    if s >= 1e-6:
        return f"{s * 1e6:8.2f} us"
    return f"{s * 1e9:8.2f} ns"


def main():
    for chip, (C, W, hbm) in CHIPS.items():
        ridge = C / W
        print(f"\n=== {chip}: C = {C:.3g} FLOP/s, W = {W:.3g} B/s, ridge I* = C/W = {ridge:.1f} FLOPs/byte ===")
        print(f"{'operation':40s} {'FLOPs':>10s} {'bytes':>10s} {'I':>8s}  {'bound':7s} {'T_math':>11s} {'T_mem':>11s} {'attain TF/s':>11s}")
        for name, fl, by in ops():
            I = fl / by
            t_math, t_mem = fl / C, by / W
            bound = "compute" if t_math > t_mem else "memory"
            attain = min(C, I * W) / 1e12
            flag = "  (weights > HBM: does not fit)" if "decode" in name and by > hbm else ""
            print(f"{name:40s} {fl:10.4g} {by:10.4g} {I:8.3f}  {bound:7s} {fmt_t(t_math):>11s} {fmt_t(t_mem):>11s} {attain:11.2f}{flag}")

    C, W, _ = CHIPS["H100 SXM (BF16)"]
    ridge = C / W
    print(f"\n=== H100 text roofline: X[B,4096] @ W[4096,14336], BF16, ridge {ridge:.1f} ===")
    print(f"{'B':>5s} {'I (exact)':>10s} {'bound':8s} {'time':>11s} {'TF/s':>7s}  bar (log scale, # = attainable, full = 989 TF/s)")
    B = 1
    while B <= 1024:
        fl, by = matmul(B, D, F)
        I = fl / by
        t = max(fl / C, by / W)
        tf = fl / t / 1e12
        bound = "compute" if I > ridge else "memory"
        width = 40
        n = round(width * (math.log10(tf) - math.log10(1)) / (math.log10(989) - math.log10(1)))
        print(f"{B:5d} {I:10.1f} {bound:8s} {fmt_t(t):>11s} {tf:7.1f}  {'#' * n}")
        B *= 2
    # exact crossover: B*D*F / (B*D + D*F + B*F) = I*  ->  B = I* D F / (D F - I* (D + F))
    b_cross = ridge * D * F / (D * F - ridge * (D + F))
    print(f"crossover: intensity = {ridge:.1f} at B = {b_cross:.1f} tokens "
          f"(not 295: activation bytes B*(D+F) lower the intensity below B)")

    # decode vs FLOPs-only prediction on H100
    fl, by = 2 * N_PARAMS, BYTES * N_PARAMS
    print(f"\ndecode step, H100: memory {by / W * 1e3:.2f} ms vs compute {fl / C * 1e3:.4f} ms "
          f"-> FLOPs-only estimate is {(by / W) / (fl / C):.0f}x too fast; ceiling {W / by:.0f} tokens/s")
    print(f"math units busy {(fl / C) / (by / W) * 100:.2f}% of the decode step; "
          f"with FP8 weights (1 byte each) the step is {N_PARAMS / W * 1e3:.2f} ms")
    print(f"vector add on CUDA cores (~66e12 FLOP/s, Scaling Book Part 12): ridge {66e12 / W:.1f}")

    # ridge points and batch-1 decode floor across hardware (Scaling Book Parts 1, 12)
    print("\n=== ridge I* = C / W and batch-1 Llama-3.1-8B BF16 decode floor (16.06 GB / W) ===")
    for name, C, W in [("H100 SXM, BF16", 989e12, 3.35e12), ("H100 SXM, FP8", 1979e12, 3.35e12),
                       ("H200, BF16", 989e12, 4.8e12), ("B200, BF16", 2250e12, 8.0e12),
                       ("TPU v5e, BF16", 1.97e14, 8.2e11)]:
        print(f"{name:16s} C {C / 1e12:6.0f} TF/s  W {W / 1e12:5.2f} TB/s  I* {C / W:6.1f}  decode {by / W * 1e3:6.2f} ms")


if __name__ == "__main__":
    main()

# Try this:
# 1. FP8 compute: set H100 C = 1979e12. The ridge doubles to ~591. Then also set BYTES = 1
#    (FP8 weights): intensity doubles too, so the crossover batch barely moves.
# 2. H200: set W = 4.8e12 (same 989e12 FLOP/s). Ridge drops to ~206, and decode gets 1.43x faster.
# 3. Set D = F = 1024 (a small matmul): the activation bytes matter more, and the exact
#    crossover climbs from ~325 to ~697 tokens (Scaling Book Part 1, Question 3).
