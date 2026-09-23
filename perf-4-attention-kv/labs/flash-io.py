"""FlashAttention II: count HBM accesses, standard attention (Alg 0) vs FlashAttention (Alg 1).

Dao et al. 2022, "FlashAttention" (arXiv 2205.14135), Theorem 2 and Section 3.2.
We walk both algorithms loop by loop, exactly as the paper writes them, and count
every element read from or written to HBM. Forward pass only, one head.

Units: the paper's M is SRAM size counted in ELEMENTS (Alg 1 sets B_c = ceil(M / 4d),
so M is compared with element counts). SRAM bytes / 2 bytes (fp16/bf16) = M.
Bytes = elements x 2. The standard/flash ratio is the same in either unit, as
long as M is in elements.

Run: python3 labs/flash-io.py    (stdlib only, < 1 s)
"""
from math import ceil

BYTES = 2  # fp16 / bf16


def standard_accesses(N, d):
    """Algorithm 0, line by line. Each matmul reads its inputs once (ideal blocked matmul)."""
    line1 = 2 * N * d + N * N      # read Q, K; write S (N x N)
    line2 = N * N + N * N          # read S; write P
    line3 = N * N + N * d + N * d  # read P, V; write O
    return line1 + line2 + line3   # = 4Nd + 4N^2


def flash_accesses(N, d, M, Bc=None, Br=None, keep=None):
    """Algorithm 1, loop by loop. Returns (total, Bc, Br, Tc).
    keep(i, j) -> bool makes it Algorithm 5 (block-sparse): zero blocks are skipped."""
    if Bc is None:
        Bc = ceil(M / (4 * d))                  # line 1
    if Br is None:
        Br = min(ceil(M / (4 * d)), d)          # line 1
    Tc = ceil(N / Bc)
    Tr = ceil(N / Br)
    total = N * d + 2 * N                       # line 2: initialise O, l, m in HBM
    for j in range(Tc):                         # line 5
        bc = min(Bc, N - j * Bc)
        if keep is not None and not any(keep(i, j) for i in range(Tr)):
            continue                            # whole column of blocks is zero
        total += 2 * bc * d                     # line 6: load K_j, V_j
        for i in range(Tr):                     # line 7
            if keep is not None and not keep(i, j):
                continue                        # Alg 5: skip zero block
            br = min(Br, N - i * Br)
            total += 2 * br * d + 2 * br        # line 8: load Q_i, O_i, l_i, m_i
            total += br * d                     # line 12: write O_i
            total += 2 * br                     # line 13: write l_i, m_i
    return total, Bc, Br, Tc


def gb(elements, copies=1):
    return elements * copies * BYTES / 1e9


def show(title):
    print("\n" + title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
show("1. The paper's Fig 2 config: GPT-2 medium, N=1024, d=64, 16 heads, batch 64, A100")
N, d, heads, batch = 1024, 64, 16, 64
copies = heads * batch
M_a100 = 192 * 1024 // BYTES                   # 192 KB SRAM per SM (paper Sec 2.1) -> elements
std = standard_accesses(N, d)
fl, Bc, Br, Tc = flash_accesses(N, d, M_a100)
print(f"M = 192 KB / 2 B = {M_a100:,} elements -> B_c = {Bc}, B_r = {Br}, T_c = {Tc} passes over Q and O")
print(f"per head, standard  : {std:>12,} elements  (4Nd + 4N^2)")
print(f"per head, flash     : {fl:>12,} elements")
print(f"whole batch, fwd    : standard {gb(std, copies):.2f} GB   flash {gb(fl, copies):.2f} GB")
print(f"ratio (our count)   : {std / fl:.2f}x  forward only")
print(f"paper, measured     : 40.3 / 4.4 = {40.3 / 4.4:.2f}x  fwd+bwd, incl. dropout/mask/other ops")
print(f"paper runtime       : 41.7 / 7.3 = {41.7 / 7.3:.2f}x   GFLOPs 75.2 / 66.6 = {75.2 / 66.6:.2f}x more")
fwd_flops = 4 * N * N * d * copies
print(f"our fwd matmul FLOPs from these shapes: 4*N^2*d*heads*batch = {fwd_flops / 1e9:.1f} GFLOP")
print(f"intensity per head: standard {4*N*N*d / (std*BYTES):.0f} FLOP/B, flash {4*N*N*d / (fl*BYTES):.0f} FLOP/B")
# the paper's rounder "M around 100 KB"
fl100, Bc100, _, Tc100 = flash_accesses(N, d, 100 * 1000 // BYTES)
print(f"with M = 100 KB (paper's 'around 100KB'): B_c = {Bc100}, T_c = {Tc100}, ratio {std / fl100:.2f}x")

# ---------------------------------------------------------------------------
show("2. Block size sweep (Fig 2 middle, qualitatively): same config, vary B_c, B_r = 64")
print(f"{'B_c':>5} {'T_c':>4} {'flash GB':>9} {'vs std':>7} {'FLOP/B':>7}  mem ms  math ms  (A100 1.5 TB/s, 312 TF)")
for bc in (32, 64, 128, 256, 512, 1024):
    f, _, _, tc = flash_accesses(N, d, None, Bc=bc, Br=64)
    g = gb(f, copies)
    mem_ms = g * 1e9 / 1.5e12 * 1e3
    math_ms = fwd_flops / 312e12 * 1e3
    print(f"{bc:>5} {tc:>4} {g:>9.2f} {std / f:>6.1f}x {4*N*N*d / (f*BYTES):>7.0f}  {mem_ms:6.2f}  {math_ms:6.2f}")
print("B_c = 512 or 1024 means K_j + V_j = 128-256 KB at d=64: it no longer fits next to Q_i, O_i and S_ij.")

# ---------------------------------------------------------------------------
show("3. The ratio standard/flash vs the M/d^2 prediction")
print("Theta says ratio -> M/d^2 for large N. Alg 1's constants give about M/(3d^2):")
print("  flash ~ 3Nd*T_c = 3Nd * N*4d/M = 12 N^2 d^2 / M ;  standard ~ 4N^2")
rows = [("A100", 192, 64), ("A100", 192, 128), ("H100", 228, 64), ("H100", 228, 128)]
print(f"{'GPU':5} {'SRAM':>6} {'d':>4} {'M elems':>8} {'M/d^2':>7} {'bytes-as-M (wrong)':>19} {'M/3d^2':>7} {'N=64k exact':>12}")
for gpu, kb, dd in rows:
    M = kb * 1024 // BYTES
    s = standard_accesses(65536, dd)
    f, *_ = flash_accesses(65536, dd, M)
    print(f"{gpu:5} {kb:>4}KB {dd:>4} {M:>8,} {M / dd**2:>7.2f} {kb * 1024 / dd**2:>19.2f} {M / (3 * dd**2):>7.2f} {s / f:>11.2f}x")

# ---------------------------------------------------------------------------
show("4. Sweep over N (d=64, A100 M): standard grows as N^2, flash as N^2 d^2 / M")
print(f"{'N':>7} {'standard':>16} {'flash':>16} {'ratio':>7}   (elements per head)")
for n in (256, 1024, 4096, 16384, 65536):
    s = standard_accesses(n, 64)
    f, *_ = flash_accesses(n, 64, M_a100)
    print(f"{n:>7} {s:>16,} {f:>16,} {s / f:>6.2f}x")

# ---------------------------------------------------------------------------
show("5. Running example: Llama-3.1-8B prefill, N=8192, d=128, 32 query heads, H100")
N, d, H, L = 8192, 128, 32, 32
M_h100 = 228 * 1024 // BYTES
std = standard_accesses(N, d)
fl, Bc, Br, Tc = flash_accesses(N, d, M_h100)
flops = 4 * N * N * d
print(f"M = 228 KB / 2 B = {M_h100:,} elements -> B_c = {Bc}, B_r = {Br}, T_c = {Tc}")
print(f"per head : standard {std:,}  flash {fl:,}  ratio {std / fl:.2f}x")
print(f"per layer (32 heads): standard {gb(std, H):.2f} GB  flash {gb(fl, H):.2f} GB")
print(f"all 32 layers       : standard {gb(std, H * L):.0f} GB -> {gb(std, H * L) / 3.35e3 * 1e3:.0f} ms at 3.35 TB/s;"
      f" flash {gb(fl, H * L):.0f} GB -> {gb(fl, H * L) / 3.35e3 * 1e3:.0f} ms")
print(f"attention matmul FLOPs, all layers: {flops * H * L / 1e12:.1f} TFLOP -> {flops * H * L / 989e12 * 1e3:.0f} ms at 989 TF")
print(f"intensity: standard {flops / (std * BYTES):.0f} FLOP/B, flash {flops / (fl * BYTES):.0f} FLOP/B (H100 ridge 295)")
ideal = 4 * N * d  # read Q, K, V once, write O once: the M = Theta(Nd) end of Prop 3
print(f"ideal (Q, K, V read once, O written once): {ideal:,} elements -> {flops / (ideal * BYTES):.0f} FLOP/B = N/2 (perf-2 attention-intensity)")
print(f"N x N scores for one head in bf16: {N * N * BYTES / 1e6:.0f} MB (what standard attention writes, then reads back)")

# ---------------------------------------------------------------------------
show("6. Block-sparse (Prop 4): keep a fraction s of blocks, N=4096, d=64, A100 M")
N, d = 4096, 64
dense, Bc, Br, Tc = flash_accesses(N, d, M_a100)
Tr = ceil(N / Br)
print(f"blocks: T_r x T_c = {Tr} x {Tc}")
for stride in (1, 2, 4, 8):
    # illustrative mask: keep block (i, j) when (i + j) % stride == 0  -> s = 1/stride
    keep = lambda i, j, k=stride: (i + j) % k == 0
    kept = sum(keep(i, j) for i in range(Tr) for j in range(Tc))
    s_frac = kept / (Tr * Tc)
    f, *_ = flash_accesses(N, d, M_a100, keep=keep)
    print(f"s = {s_frac:.3f}: {f:>12,} elements  = {f / dense:.3f} of dense flash  (standard: {standard_accesses(N, d):,})")
print("Traffic falls ~ in proportion to s; the N*d init/output term stays (Prop 4: Theta(Nd + N^2 d^2 s / M)).")

# Try this:
# 1. Set BYTES = 4 (fp32): M in elements halves, T_c doubles, and the flash traffic roughly doubles again.
# 2. Change d to 256 in section 5: B_r = min(M/4d, d) and the ratio collapses toward 1 (d^2 approaches M).
# 3. Add block sparsity: multiply the loop body in flash_accesses by a fraction s of kept blocks (Prop 4).
