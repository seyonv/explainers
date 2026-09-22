"""Critical batch size lab: Scaling Book Part 7, "Linear operations: what bottlenecks us?"

Run: python3 labs/critical-batch-size.py   (stdlib only, CPU, well under a second)

A linear layer X[B, D] @ W[D, F] -> Z[B, F] does 2BDF FLOPs and moves
(act_bytes*B*D + w_bytes*D*F + act_bytes*B*F) bytes to/from HBM.
It is compute-bound when T_math = FLOPs / C >= T_comms = bytes / W_hbm.
When B << D, F the weight bytes dominate, so intensity ~ 2B / w_bytes, and
B_crit = (C / W_hbm) * w_bytes / 2.
"""

CHIPS = {
    # name: bf16 dense FLOP/s, low-precision (int8/fp8) dense OP/s, HBM bytes/s
    "TPU v5e": dict(bf16=1.97e14, low=3.94e14, bw=8.2e11),   # book Parts 1, 7
    "H100":    dict(bf16=9.89e14, low=1.979e15, bw=3.35e12),  # book Part 12 (dense)
    "B200":    dict(bf16=2.25e15, low=4.5e15, bw=8.0e12),     # book Part 12
}

D, F = 4096, 14336  # one Llama-3.1-8B MLP matrix (W_in / W_gate is [D, F])


def matmul(B, D, F, w_bytes=2, a_bytes=2):
    flops = 2 * B * D * F
    nbytes = a_bytes * B * D + w_bytes * D * F + a_bytes * B * F
    return flops, nbytes


def exact_crossing(D, F, C, bw, w_bytes=2, a_bytes=2):
    """Smallest B with 2BDF/C = (a B (D+F) + w DF)/bw, no approximation."""
    I = C / bw
    den = 2 * D * F - I * a_bytes * (D + F)
    return float("inf") if den <= 0 else I * w_bytes * D * F / den


print("== Hardware intensity alpha_hbm = C / W_hbm (bf16) ==")
for name, c in CHIPS.items():
    print(f"{name:8s} {c['bf16']:.3g} / {c['bw']:.3g} = {c['bf16'] / c['bw']:.1f}")
print("(book: v5e 240; H100 295 in Part 12, 'about 280' in Part 7's takeaway; B200 281)")

print(f"\n== One Llama-3.1-8B MLP matrix, bf16[B,{D}] x bf16[{D},{F}] ==")
print(f"weight bytes = 2*D*F = {2 * D * F:,} B = {2 * D * F / 1e6:.1f} MB")
for name in ["TPU v5e", "H100"]:
    c = CHIPS[name]
    print(f"\n-- {name} (C = {c['bf16']:.3g} FLOP/s, W_hbm = {c['bw']:.3g} B/s) --")
    print(f"{'B':>5} | {'FLOPs':>10} | {'bytes':>10} | {'intensity':>9} | {'T_math':>9} | {'T_comms':>9} | {'bound':>7} | {'util':>6}")
    for B in [1, 16, 64, 256, 295, 512, 1024]:
        fl, by = matmul(B, D, F)
        tm, tc = fl / c["bf16"], by / c["bw"]
        bound = "compute" if tm >= tc else "memory"
        util = tm / max(tm, tc)
        print(f"{B:5d} | {fl:10.3e} | {by:10.3e} | {fl / by:9.1f} | {tm * 1e6:6.2f} us | {tc * 1e6:6.2f} us | {bound:>7} | {util * 100:5.1f}%")

print("\n== Exact crossing B (T_math = T_comms, exact bytes) for D=4096, F=14336 ==")
for name, c in CHIPS.items():
    I = c["bf16"] / c["bw"]
    x = exact_crossing(D, F, c["bf16"], c["bw"])
    print(f"{name:8s} approx B_crit = {I:.1f}  exact = {x:.1f}  ({(x / I - 1) * 100:+.1f}%)")
print("smaller matrices, D = F = 1024:")
for name, c in CHIPS.items():
    I = c["bf16"] / c["bw"]
    print(f"{name:8s} approx {I:.1f}  exact {exact_crossing(1024, 1024, c['bf16'], c['bw']):.1f}")

print("\n== Precision table: B_crit ~ (C_compute / W_hbm) * w_bytes / 2  (B << D, F) ==")
print("   plus the exact crossing for the Llama MLP matrix (D=4096, F=14336)")
CASES = [
    # label, weight bytes, activation bytes, which compute rate
    ("bf16 weights x bf16 compute", 2, 2, "bf16"),
    ("int8/fp8 weights, bf16 compute", 1, 2, "bf16"),
    ("int8/fp8 weights x int8/fp8 compute", 1, 1, "low"),
    ("bf16 weights, int8/fp8 compute", 2, 1, "low"),
]
for name, c in CHIPS.items():
    print(f"-- {name} --")
    for label, wb, ab, rate in CASES:
        C = c[rate]
        approx = C / c["bw"] * wb / 2
        exact = exact_crossing(D, F, C, c["bw"], w_bytes=wb, a_bytes=ab)
        beta = (wb * 8) / (16 if rate == "bf16" else 8)
        print(f"  {label:38s} beta={beta:3.1f}  alpha(bf16)={c['bf16'] / c['bw']:5.1f}  "
              f"B_crit ~ {approx:6.1f}   exact(Llama MLP) {exact:6.1f}")

print("\n== Why decode struggles to reach B_crit on one H100 (Llama-3.1-8B, bf16) ==")
free = 80e9 - 16.06e9
kv_per_tok = 131072
tokens = free / kv_per_tok
Bc = CHIPS["H100"]["bf16"] / CHIPS["H100"]["bw"]
print(f"free HBM {free / 1e9:.2f} GB / {kv_per_tok} B per token = {tokens:,.0f} KV tokens")
print(f"to decode {Bc:.0f} sequences at once, each can hold {tokens / Bc:,.0f} tokens of context")
print(f"at 8,192 tokens each only {tokens / 8192:.1f} sequences fit")
print(f"one 1,000-token prompt in prefill = 1,000 tokens in the step > {Bc:.0f}")

print("\n== MoE: E experts, k active -> B_crit x E/k (book Worked Problem Q5) ==")
print(f"E=16, k=2 on v5e: 240 * 16/2 = {240 * 16 / 2:.0f} tokens")

print("\n== Step-time curve for the SVG (H100, D=4096, F=14336) ==")
c = CHIPS["H100"]
for B in [0, 128, 256, 295, 325, 384, 512, 640, 768, 896, 1024]:
    fl, by = matmul(B, D, F)
    tm, tc = fl / c["bf16"], by / c["bw"]
    print(f"B={B:5d}  T_comms={tc * 1e6:6.2f} us  T_math={tm * 1e6:6.2f} us  step={max(tm, tc) * 1e6:6.2f} us")

# Try this:
# 1. Set D, F = 1024, 1024 at the top: the table's crossing moves well past 295 (the B << D, F
#    approximation breaks down when the matrix is small).
# 2. FP8 on H100: in the table loop use c["low"] for C and matmul(..., w_bytes=1, a_bytes=1).
#    The ridge doubles to ~591 but the weight bytes halve, so B_crit lands back near 295.
# 3. Put B200 in the table loop: 2.25e15 / 8e12 = 281, almost the same ratio as H100.
