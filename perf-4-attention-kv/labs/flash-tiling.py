"""FlashAttention tiling lab: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness" (NeurIPS 2022, arXiv 2205.14135), Algorithm 0 vs Algorithm 1.

Run: python3 labs/flash-tiling.py   (numpy, CPU, a few seconds)

Part 1  Block sizes from Algorithm 1 line 1 for an SRAM of M floats (4-byte elements):
          B_c = ceil(M / 4d),  B_r = min(ceil(M / 4d), d),  T_r = ceil(N / B_r),  T_c = ceil(N / B_c)
        for the paper's A100 (192 KB per SM) and the H100 (228 KB shared memory per SM),
        at GPT-2's d = 64 and Llama-3.1-8B's d = 128. Our recomputation.
Part 2  Algorithm 0 (materialise S and P) vs Algorithm 1 exactly as written: outer loop over
        K_j, V_j blocks, inner loop over Q_i blocks, running m and l per row, O_i rescaled
        in place. Random Q, K, V; prints max abs error in float32 and float64.
Part 3  Trace of (m, l) for one query row across the column blocks: how the running max
        and running sum change, and the rescale factor e^(m_old - m_new) applied to old l and O.
Part 4  Extra memory: Algorithm 0 keeps S and P (N x N each), Algorithm 1 keeps only
        m and l (N each). Toy sizes and the running example (Llama-3.1-8B, 32 query heads,
        N = 8,192 and 131,072, S and P in BF16, statistics in FP32). Our recomputation.
"""
import math
import numpy as np


def block_sizes(M, d, N):
    Bc = math.ceil(M / (4 * d))
    Br = min(math.ceil(M / (4 * d)), d)
    return Bc, Br, math.ceil(N / Br), math.ceil(N / Bc)


def standard_attention(Q, K, V):
    """Algorithm 0: S and P are full N x N matrices (in HBM on a GPU)."""
    S = Q @ K.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return P @ V, S.nbytes + P.nbytes


def flash_attention(Q, K, V, M, trace_row=None):
    """Algorithm 1, line by line. No 1/sqrt(d) scaling, as in the paper's statement."""
    N, d = Q.shape
    dt = Q.dtype
    Bc, Br, Tr, Tc = block_sizes(M, d, N)                        # line 1
    O = np.zeros((N, d), dt)                                     # line 2
    l = np.zeros(N, dt)
    m = np.full(N, -np.inf, dt)
    trace = []
    for j in range(Tc):                                          # line 5: outer loop over K, V
        Kj, Vj = K[j * Bc:(j + 1) * Bc], V[j * Bc:(j + 1) * Bc]  # line 6: load to "SRAM"
        for i in range(Tr):                                      # line 7: inner loop over Q
            r = slice(i * Br, (i + 1) * Br)
            Qi, Oi, li, mi = Q[r], O[r], l[r], m[r]              # line 8
            Sij = Qi @ Kj.T                                      # line 9  (Br x Bc, on chip)
            mt = Sij.max(axis=1)                                 # line 10
            Pt = np.exp(Sij - mt[:, None])
            lt = Pt.sum(axis=1)
            mnew = np.maximum(mi, mt)                            # line 11
            lnew = np.exp(mi - mnew) * li + np.exp(mt - mnew) * lt
            O[r] = ((li * np.exp(mi - mnew))[:, None] * Oi      # line 12: rescale old O,
                    + np.exp(mt - mnew)[:, None] * (Pt @ Vj)) / lnew[:, None]  # add new block
            if trace_row is not None and r.start <= trace_row < r.stop:
                k = trace_row - r.start
                trace.append((j, float(mt[k]), float(mi[k]), float(mnew[k]),
                              float(np.exp(mi[k] - mnew[k])), float(lt[k]),
                              float(np.exp(mt[k] - mnew[k])), float(lnew[k])))
            l[r], m[r] = lnew, mnew                              # line 13
    return O, l, m, (Bc, Br, Tr, Tc), trace


def part1():
    print("Part 1: Algorithm 1 block sizes, M in 4-byte floats (our recomputation)")
    print(f"  {'SRAM per SM':<26}{'M floats':>9}{'d':>5}{'N':>7}{'B_c':>6}{'B_r':>6}{'T_r':>6}{'T_c':>6}{'tiles':>8}")
    rows = [("A100 192 KB (paper)", 192, 64, 1024), ("A100 192 KB (paper)", 192, 128, 8192),
            ("H100 228 KB", 228, 64, 1024), ("H100 228 KB", 228, 128, 8192)]
    for name, kb, d, N in rows:
        M = kb * 1024 // 4
        Bc, Br, Tr, Tc = block_sizes(M, d, N)
        print(f"  {name:<26}{M:>9,}{d:>5}{N:>7,}{Bc:>6}{Br:>6}{Tr:>6}{Tc:>6}{Tr * Tc:>8,}")
    M = 228 * 1024 // 4
    Bc, Br, Tr, Tc = block_sizes(M, 128, 8192)
    print(f"  Llama head, H100: M = 228*1024/4 = {M:,}; B_c = ceil({M:,}/(4*128)) = ceil({M / 512:.2f}) = {Bc}")
    print(f"    B_r = min({Bc}, 128) = {Br};  T_r = T_c = ceil(8192/{Bc}) = {Tc}  ({Tc * Bc:,} rows, last block {8192 - (Tc - 1) * Bc})")
    four = 4 * Bc * 128
    print(f"    Q_i + K_j + V_j + O_i = 4 x {Bc} x 128 = {four:,} floats = {four * 4 / 1024:.1f} KB;"
          f" S_ij = {Br}x{Bc} = {Br * Bc:,} floats = {Br * Bc * 4 / 1024:.1f} KB more")
    print()


def part2(M=228 * 1024 // 4):
    print(f"Part 2: Algorithm 0 vs Algorithm 1 (M = {M:,} floats), random Q, K, V ~ N(0, 1) scaled by d^-1/4")
    rng = np.random.default_rng(0)
    for N, d in [(512, 64), (512, 128), (2048, 128)]:
        for dt in (np.float32, np.float64):
            Q, K, V = (rng.standard_normal((N, d)) / d ** 0.25 for _ in range(3))
            Q, K, V = Q.astype(dt), K.astype(dt), V.astype(dt)
            O0, _ = standard_attention(Q, K, V)
            O1, l, m, (Bc, Br, Tr, Tc), _ = flash_attention(Q, K, V, M)
            err = np.abs(O0 - O1).max()
            print(f"  N={N:<5} d={d:<4} {np.dtype(dt).name:<8} B_c={Bc:<4} B_r={Br:<4} "
                  f"T_r={Tr:<3} T_c={Tc:<3} max|O_std - O_flash| = {err:.1e}")
    print()


def part3(N=512, d=128, row=0, seed=0):
    M = 228 * 1024 // 4
    rng = np.random.default_rng(seed)
    Q, K, V = (rng.standard_normal((N, d)) / d ** 0.25 for _ in range(3))
    O, l, m, (Bc, Br, Tr, Tc), tr = flash_attention(Q, K, V, M, trace_row=row)
    print(f"Part 3: trace of row {row} across the {Tc} column blocks (N={N}, d={d}, B_c={Bc}, float64)")
    print("  l new = e^(m old - m new) * l old + e^(m~ - m new) * l~")
    print(f"  {'j':>2} {'cols':>8} {'m~':>7} {'m old':>7} {'m new':>7} {'e^(mold-mnew)':>13} {'l~':>8} {'e^(m~-mnew)':>11} {'l new':>8}")
    for j, mt, mi, mn, sc, lt, sn, ln in tr:
        cols = f"{j * Bc}-{min((j + 1) * Bc, N) - 1}"
        print(f"  {j + 1:>2} {cols:>8} {mt:>7.4f} {mi:>7.4f} {mn:>7.4f} {sc:>13.4f} {lt:>8.4f} {sn:>11.4f} {ln:>8.4f}")
    s = Q[row] @ K.T
    print(f"  check: max_k S[{row},k] = {s.max():.4f};  sum_k e^(S-m) = {np.exp(s - s.max()).sum():.4f}")
    print()


def part4():
    print("Part 4: extra memory beyond Q, K, V, O (our recomputation)")
    for N, d in [(512, 64), (512, 128)]:
        print(f"  toy N={N}, d={d}, float32: Alg 0 S+P = 2*N^2 = {2 * N * N:,} floats"
              f" ({2 * N * N * 4 / 1e6:.2f} MB); Alg 1 m+l = 2*N = {2 * N:,} floats ({2 * N * 4 / 1e3:.1f} KB);"
              f" ratio {N:,}x")
    heads = 32
    for N in (8192, 131072):
        sp = 2 * heads * N * N * 2
        st = 2 * heads * N * 4
        o = heads * N * 128 * 2
        print(f"  Llama-3.1-8B, one layer, 32 heads, N={N:,}:")
        print(f"    Alg 0 S+P in BF16 = 2 x 32 x N^2 x 2 B = {sp:,} B = {sp / 1e9:,.2f} GB"
              f"  (x 32 layers kept for backward: {32 * sp / 1e9:,.0f} GB)")
        print(f"    Alg 1 m+l in FP32 = 2 x 32 x N x 4 B   = {st:,} B = {st / 1e6:.2f} MB"
              f"   (ratio {sp / st:,.0f}x; output O itself = {o / 1e6:.1f} MB)")
    print()


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()

# Try this:
#   part3(row=5, seed=3)        # another row: the max jumps at a different block, and old l and O shrink by e^(m_old-m_new)
#   part2(M=4*64*16)            # tiny SRAM: B_c = B_r = 16 at d = 64, 8 at d = 128; far more tiles, same exact answer
#   In flash_attention, drop the np.exp(mi - mnew) factor from line 12 (keep old O unrescaled):
#   the error jumps from ~1e-16 to ~0.1-0.5, because blocks summed under different maxima no longer agree
