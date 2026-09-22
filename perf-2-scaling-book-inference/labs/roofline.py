"""Roofline lab: reproduces the numbers in Scaling Book Part 1 ("All About Rooflines").

Run: python3 labs/roofline.py   (stdlib only, CPU, well under a second)

Every op takes at least max(T_math, T_comms) and at most T_math + T_comms.
T_math = FLOPs / (FLOP/s), T_comms = bytes / (bytes/s).
"""

# Chips (book values; H100/H200 from NVIDIA spec sheets as quoted in the book)
CHIPS = {
    "TPU v5e": dict(flops=1.97e14, bw=8.2e11),   # bf16 MXU, HBM
    "H100":    dict(flops=9.89e14, bw=3.35e12),  # dense bf16 (spec sheet 1.979e15 is with sparsity)
}
V5E_INT8 = 3.94e14   # v5e int8 OPs/s
ICI = 4.5e10         # v5e inter-chip bytes/s, one direction


def matmul(B, D, F, w_bytes=2, a_bytes=2):
    """X[B,D] @ Y[D,F] -> Z[B,F]. Returns (FLOPs, bytes moved to/from HBM)."""
    flops = 2 * B * D * F
    nbytes = a_bytes * B * D + w_bytes * D * F + a_bytes * B * F
    return flops, nbytes


def bound(flops, nbytes, chip):
    t_math = flops / chip["flops"]
    t_comms = nbytes / chip["bw"]
    return t_math, t_comms, max(t_math, t_comms), t_math + t_comms


def exact_bcrit(D, F, hw_intensity, w_bytes=2, a_bytes=2):
    """Smallest B where 2BDF / (a*BD + w*DF + a*BF) = hw_intensity (exact bytes, no B<<D approximation)."""
    # 2BDF = I (a B (D+F) + w DF)  ->  B = I w DF / (2DF - I a (D+F))
    den = 2 * D * F - hw_intensity * a_bytes * (D + F)
    return float("inf") if den <= 0 else hw_intensity * w_bytes * D * F / den


def us(t):
    return f"{t * 1e6:9.3f} us"


print("== Hardware intensity (peak FLOP/s / bandwidth) ==")
for name, c in CHIPS.items():
    print(f"{name:8s} {c['flops']:.3g} / {c['bw']:.3g} = {c['flops'] / c['bw']:.1f} FLOPs/byte")
print(f"v5e int8 {V5E_INT8:.3g} / 8.2e11 = {V5E_INT8 / 8.2e11:.1f} OPs/byte")
print(f"v5e ICI  1.97e14 / {ICI:.2g} = {1.97e14 / ICI:.1f} FLOPs/byte")

print("\n== Dot product x.y, bf16[N]: (2N-1) FLOPs / (2N+2N+2) bytes ==")
for N in [1, 10, 1000, 10**6]:
    print(f"N={N:>8}: {(2 * N - 1) / (4 * N + 2):.4f} FLOPs/byte")
print("limit N->inf: 0.5 FLOPs/byte  (far below 240 -> bandwidth-bound)")

print("\n== bf16 matmul, D = F = 4096: T_math, T_comms, bound ==")
D = F = 4096
for B in [1, 64, 240, 1024]:
    fl, by = matmul(B, D, F)
    I = fl / by
    row = f"B={B:5d}  FLOPs={fl:.3e}  bytes={by:.3e}  intensity={I:7.1f}"
    print(row)
    for name, c in CHIPS.items():
        tm, tc, lo, hi = bound(fl, by, c)
        which = "compute" if tm > tc else "memory"
        print(f"    {name:8s} T_math={us(tm)}  T_comms={us(tc)}  lower={us(lo)}  upper={us(hi)}  -> {which}-bound")

print("\n== Exact crossover batch (bf16, exact bytes) vs the B > I_hw approximation ==")
for name, c in CHIPS.items():
    I = c["flops"] / c["bw"]
    for DF in [4096, 1024]:
        print(f"{name:8s} D=F={DF:5d}: approx B > {I:.0f}, exact B > {exact_bcrit(DF, DF, I):.1f}")
    print(f"{name:8s} Llama-3.1-8B MLP D=4096 F=14336: exact B > {exact_bcrit(4096, 14336, I):.1f}")

print("\n== Cross-chip example: X[B,D] @ Y[D,F] split along D over 2 chips ==")
print("T_math = BDF/1.97e14, T_comms = 2BF/4.5e10 -> intensity D/2")
print(f"compute-bound when D/2 > {1.97e14 / ICI:.1f}  i.e. D > {2 * 1.97e14 / ICI:.1f}")

print("\n== Q1: int8 x int8 on v5e ==")
print(f"hardware int8 intensity = {V5E_INT8 / 8.2e11:.1f}; op intensity ~ 2B -> B > {V5E_INT8 / 8.2e11 / 2:.1f}")

print("\n== Q2: bf16 activations x int8 weights, bf16 compute (1.97e14) ==")
print(f"op intensity ~ 2B -> B > {1.97e14 / 8.2e11 / 2:.1f}")

print("\n== Q3: Q2 setup, exact bytes 2BD + DF + 2BF on v5e (book's plot, as a table) ==")
v5e = CHIPS["TPU v5e"]
print(f"{'B':>5} | {'D=F=4096 TFLOP/s':>17} | {'D=F=1024 TFLOP/s':>17}")
for B in [1, 16, 64, 128, 136, 200, 226, 256, 384, 511]:
    cells = []
    for DF in [4096, 1024]:
        fl, by = matmul(B, DF, DF, w_bytes=1, a_bytes=2)
        tm, tc, lo, _ = bound(fl, by, v5e)
        cells.append(fl / lo / 1e12)
    print(f"{B:5d} | {cells[0]:17.1f} | {cells[1]:17.1f}")
I = v5e["flops"] / v5e["bw"]
b_big = exact_bcrit(4096, 4096, I, w_bytes=1)
b_small = exact_bcrit(1024, 1024, I, w_bytes=1)
print(f"exact B_crit: D=F=4096 -> {b_big:.1f}, D=F=1024 -> {b_small:.1f}  (ratio {b_small / b_big:.2f}x)")

print("\n== Q4: per-example weights int8[B,D] . int8[B,D,F] ==")
for B in [1, 64, 1024]:
    fl = 2 * B * D * F
    by = B * D + B * D * F + B * F
    print(f"B={B:5d} D=F=4096: intensity = {fl / by:.3f}  (~2, independent of B)")

print("\n== Q5: H100 SXM ==")
print(f"9.89e14 / 3.35e12 = {9.89e14 / 3.35e12:.1f}")

print("\n== Roofline plot points (H100 vs H200: same 9.89e14 FLOP/s, two bandwidths) ==")
for bw_name, bw in [("H100 3.35e12", 3.35e12), ("H200 4.8e12", 4.8e12)]:
    print(f"{bw_name}: ridge at {9.89e14 / bw:.1f} FLOPs/byte")
for label, I in [("dot product", 0.5), ("matmul B=240", matmul(240, 4096, 4096)[0] / matmul(240, 4096, 4096)[1]),
                 ("matmul B=1024", matmul(1024, 4096, 4096)[0] / matmul(1024, 4096, 4096)[1])]:
    a = min(9.89e14, I * 3.35e12)
    b = min(9.89e14, I * 4.8e12)
    print(f"{label:14s} I={I:7.1f}: H100 {a / 1e12:6.1f} TF/s, H200 {b / 1e12:6.1f} TF/s")

# Try this:
# 1. FP8 on H100: set CHIPS["H100"]["flops"] = 1.979e15 -> hardware intensity ~591, so bf16-sized
#    batches that were compute-bound become memory-bound again unless the bytes also halve (fp8 weights).
# 2. H200 bandwidth: set CHIPS["H100"]["bw"] = 4.8e12 -> ridge drops to ~206; same FLOPs, earlier crossover.
# 3. Try D = F = 16384 in the exact-crossover loop and watch it approach the B > I_hw rule.
