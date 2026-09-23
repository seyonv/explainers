"""FlashAttention-2, Algorithm 1 and 2, in numpy: fewer non-matmul FLOPs.

Companion to perf-4-attention-kv/flashattention-2-algorithm.html.
Stdlib + numpy, CPU, runs in a few seconds:  python3 labs/fa2-nonmatmul.py

What it does
  1. Standard attention (materialises the N x N matrix) as the reference.
  2. FlashAttention-1 forward (FA paper, Alg 1): outer loop over K/V blocks,
     inner loop over Q blocks, output rescaled by 1/l at every step.
  3. FlashAttention-2 forward (FA2 paper, Alg 1): outer loop over Q blocks,
     un-scaled output, one divide by l at the end, stores only L = m + log l.
  4. Op counters for the non-matmul work in both, per score element.
  5. Causal FA2 with block skipping; block counts for Llama-3.1-8B prefill.
  6. FA2 backward (Alg 2) from L alone, checked against the textbook gradient.
  7. GQA: implicit head indexing, dK/dV summed across the shared query heads.
  8. The A100 cost model: T = F_matmul / 312 TF + F_other / 19.5 TF.

Counting convention (ours, not the paper's): every elementwise add, subtract,
multiply, divide, compare, exp and log counts as one non-matmul FLOP. QK^T and
PV are matmul FLOPs (2 per multiply-add). The 1/sqrt(d) scale is omitted, as in
the paper's exposition; it would add the same cost to both versions.
"""
import numpy as np

rng = np.random.default_rng(0)


def standard_attention(Q, K, V, causal=False):
    S = Q @ K.T
    if causal:
        S = np.where(np.tril(np.ones(S.shape, bool)), S, -np.inf)
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return P @ V, P


def fa1_forward(Q, K, V, Br, Bc):
    """FlashAttention (v1) Alg 1. O, l, m live in 'HBM' and are re-read every step."""
    N, d = Q.shape
    O = np.zeros((N, d)); l = np.zeros(N); m = np.full(N, -np.inf)
    ops = dict(softmax=0, rowstats=0, rescale=0)
    hbm_O_elems = 0  # O_i elements read + written across the loops
    for j in range(0, N, Bc):                    # outer: K_j, V_j
        Kj, Vj = K[j:j + Bc], V[j:j + Bc]
        for i in range(0, N, Br):                # inner: Q_i, O_i, l_i, m_i
            Qi = Q[i:i + Br]; Oi = O[i:i + Br]; li = l[i:i + Br]; mi = m[i:i + Br]
            hbm_O_elems += Oi.size               # read O_i
            S = Qi @ Kj.T
            mt = S.max(axis=1)                   # rowmax: 1 compare / score
            Pt = np.exp(S - mt[:, None])         # subtract + exp: 2 / score
            lt = Pt.sum(axis=1)                  # rowsum: 1 add / score
            ops['softmax'] += 4 * S.size
            mnew = np.maximum(mi, mt)            # 1 / row
            a = np.exp(mi - mnew); b = np.exp(mt - mnew)   # 2 sub + 2 exp / row
            lnew = a * li + b * lt               # 2 mul + 1 add / row
            ops['rowstats'] += 8 * len(mi)
            ca = li * a; inv = 1.0 / lnew        # 2 / row
            ops['rowstats'] += 2 * len(mi)
            PV = Pt @ Vj
            O[i:i + Br] = inv[:, None] * (ca[:, None] * Oi + b[:, None] * PV)
            ops['rescale'] += 4 * Oi.size        # 3 mul + 1 add per O element
            hbm_O_elems += Oi.size               # write O_i
            l[i:i + Br] = lnew; m[i:i + Br] = mnew
    return O, l, m, ops, hbm_O_elems


def fa2_forward(Q, K, V, Br, Bc, causal=False, counts=None):
    """FlashAttention-2 Alg 1. O_i stays on chip for the whole inner loop."""
    N, d = Q.shape
    O = np.zeros((N, d)); L = np.zeros(N)
    ops = dict(softmax=0, rowstats=0, rescale=0, final=0, mask=0)
    blocks = dict(computed=0, masked=0, skipped=0)
    for i in range(0, N, Br):                    # outer: Q_i
        Qi = Q[i:i + Br]; rows = np.arange(i, i + len(Qi))
        Oi = np.zeros((len(Qi), d)); li = np.zeros(len(Qi)); mi = np.full(len(Qi), -np.inf)
        for j in range(0, N, Bc):                # inner: K_j, V_j
            cols = np.arange(j, min(j + Bc, N))
            if causal and cols[0] > rows[-1]:    # whole block above the diagonal
                blocks['skipped'] += 1
                continue
            blocks['computed'] += 1
            S = Qi @ K[j:j + Bc].T
            if causal and cols[-1] > rows[0]:    # block straddles the diagonal
                S = np.where(cols[None, :] <= rows[:, None], S, -np.inf)
                ops['mask'] += S.size
                blocks['masked'] += 1
            mnew = np.maximum(mi, S.max(axis=1))  # 1 compare / score + 1 / row
            Pt = np.exp(S - mnew[:, None])       # 2 / score
            a = np.exp(mi - mnew)                # 2 / row
            li = a * li + Pt.sum(axis=1)         # 1 add / score + 2 / row
            ops['softmax'] += 4 * S.size
            ops['rowstats'] += 5 * len(mi)
            Oi = a[:, None] * Oi + Pt @ V[j:j + Bc]   # 1 mul + 1 add per O element
            ops['rescale'] += 2 * Oi.size
            mi = mnew
        O[i:i + Br] = Oi / li[:, None]           # the one divide (as 1 recip/row + 1 mul/elem)
        L[i:i + Br] = mi + np.log(li)            # log + add / row
        ops['final'] += Oi.size + 3 * len(li)
    if counts is not None:
        counts.update(blocks)
    return O, L, ops


def fa2_backward(Q, K, V, O, dO, L, Br, Bc):
    """FlashAttention-2 Alg 2: P rebuilt from L alone; dQ read-modify-written in 'HBM'."""
    N, d = Q.shape
    dQ = np.zeros_like(Q); dK = np.zeros_like(K); dV = np.zeros_like(V)
    D = (dO * O).sum(axis=1)                     # D = rowsum(dO o O)
    for j in range(0, N, Bc):                    # outer: K_j, V_j
        Kj, Vj = K[j:j + Bc], V[j:j + Bc]
        dKj = np.zeros_like(Kj); dVj = np.zeros_like(Vj)
        for i in range(0, N, Br):                # inner: Q_i, O_i, dO_i, dQ_i, L_i, D_i
            Qi, dOi = Q[i:i + Br], dO[i:i + Br]
            P = np.exp(Qi @ Kj.T - L[i:i + Br, None])   # 1 sub + 1 exp per score
            dVj += P.T @ dOi
            dP = dOi @ Vj.T
            dS = P * (dP - D[i:i + Br, None])
            dQ[i:i + Br] += dS @ Kj              # load dQ_i, add, write back
            dKj += dS.T @ Qi
        dK[j:j + Bc] = dKj; dV[j:j + Bc] = dVj
    return dQ, dK, dV


def reference_backward(Q, K, V, dO):
    _, P = standard_attention(Q, K, V)
    dV = P.T @ dO
    dP = dO @ V.T
    dS = P * (dP - (dP * P).sum(axis=1, keepdims=True))   # (diag(p) - p p^T) dp, row-wise
    return dS @ K, dS.T @ Q, dV


def per_score(ops, n_scores):
    return {k: v / n_scores for k, v in ops.items()}


print("=" * 72)
print("1. Correctness: FA1 and FA2 forward vs standard attention (float64)")
N, d, B = 1024, 128, 128
Q, K, V = (rng.standard_normal((N, d)) for _ in range(3))
O_ref, P_ref = standard_attention(Q, K, V)
O1, l1, m1, ops1, hbmO1 = fa1_forward(Q, K, V, B, B)
O2, L2, ops2 = fa2_forward(Q, K, V, B, B)
print(f"   N = {N}, d = {d}, Br = Bc = {B}")
print(f"   max |O_FA1 - O_std| = {np.abs(O1 - O_ref).max():.2e}")
print(f"   max |O_FA2 - O_std| = {np.abs(O2 - O_ref).max():.2e}")
print(f"   max |O_FA2 - O_FA1| = {np.abs(O2 - O1).max():.2e}")
print(f"   L = m + log(l) matches FA1's stored (m, l): max diff "
      f"{np.abs(L2 - (m1 + np.log(l1))).max():.2e}")
print(f"   FA1 stores 2 vectors of N (m, l) = {2 * N} floats; FA2 stores L = {N} floats")

print("=" * 72)
print("2. Non-matmul FLOPs per score element (one full forward pass)")
n_scores = N * N
s1, s2 = per_score(ops1, n_scores), per_score(ops2, n_scores)
steps = (N // B) ** 2
for name, s in (("FA1", s1), ("FA2", s2)):
    print(f"   {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in s.items()) +
          f"  -> total {sum(s.values()):.3f}")
# steady state = one inner step, excluding FA2's once-per-row finalisation
fa1_step = (4 * B * B + 10 * B + 4 * B * d) / (B * B)
fa2_step = (4 * B * B + 5 * B + 2 * B * d) / (B * B)
print(f"   per inner step, per score element: FA1 {fa1_step:.3f}   FA2 {fa2_step:.3f}")
print(f"     FA1 = 4 softmax + 10/{B} row stats + 4*{d}/{B} rescale of O")
print(f"     FA2 = 4 softmax +  5/{B} row stats + 2*{d}/{B} rescale of O  (+ one divide at the end)")
print(f"   FA2 end-of-row divide, amortised over a row of N = 8192: {(d + 3) / 8192:.4f} per score")
mm_per_score = 4 * d
print(f"   matmul FLOPs per score element: QK^T 2d + PV 2d = {mm_per_score}")
print(f"   non-matmul share of FLOPs: FA1 {fa1_step / (mm_per_score + fa1_step):.2%}, "
      f"FA2 {fa2_step / (mm_per_score + fa2_step):.2%}")

print("=" * 72)
print("3. A100 cost model: T = F_matmul / 312e12 + F_other / 19.5e12")
MM, NM = 312e12, 19.5e12
print(f"   cost ratio 312 / 19.5 = {MM / NM:.1f}x")
t_mm = mm_per_score / MM
for name, f in (("FA1", fa1_step), ("FA2", fa2_step)):
    t_nm = f / NM
    print(f"   {name}: matmul {t_mm * 1e12:.3f} ps + non-matmul {t_nm * 1e12:.3f} ps "
          f"per score -> non-matmul is {t_nm / (t_mm + t_nm):.1%} of ideal time; "
          f"ceiling {t_mm / (t_mm + t_nm):.1%} of matmul peak")
print(f"   in matmul-FLOP equivalents: FA1 {fa1_step * 16:.0f}, FA2 {fa2_step * 16:.0f} "
      f"(vs {mm_per_score} real matmul FLOPs)")

print("=" * 72)
print("4. Loop order: how often O is touched in HBM (our recomputation)")
Tc = N // B
print(f"   FA1: O_i read + written every inner step: {hbmO1:,} elements = 2 * Tc * N * d "
      f"with Tc = {Tc}")
print(f"   FA2: O_i written once: {N * d:,} elements -> {hbmO1 / (N * d):.0f}x fewer")

print("=" * 72)
print("5. Causal masking: skip blocks above the diagonal, mask only the diagonal block")
Nc = 1024
Qc, Kc, Vc = (rng.standard_normal((Nc, 64)) for _ in range(3))
Oc_ref, _ = standard_attention(Qc, Kc, Vc, causal=True)
cnt = {}
Oc, _, opsc = fa2_forward(Qc, Kc, Vc, 128, 128, causal=True, counts=cnt)
print(f"   N = {Nc}, d = 64, blocks 128x128: max |O_FA2 - O_std| = {np.abs(Oc - Oc_ref).max():.2e}")
print(f"   blocks computed {cnt['computed']}, of which masked {cnt['masked']}, skipped {cnt['skipped']}")


def causal_blocks(N, Bs):
    T = -(-N // Bs)
    computed = T * (T + 1) // 2
    return T, T * T, computed, T, T * (T - 1) // 2


for Nq in (8192, 1024, 131072):
    T, total, comp, masked, skipped = causal_blocks(Nq, 128)
    print(f"   N = {Nq:>6}, 128x128 blocks: {T} per side, {total:,} total; computed {comp:,} "
          f"({comp / total:.1%}), masked {masked} ({masked / comp:.1%} of computed), "
          f"skipped {skipped:,}; FLOP saving {total / comp:.2f}x")
print("   (paper, A100: 'around 1.7-1.8x speedup compared to attention without the causal mask')")

print("=" * 72)
print("6. Running example: Llama-3.1-8B prefill, N = 8192, all 32 layers x 32 query heads")
T, total, comp, masked, skipped = causal_blocks(8192, 128)
heads_layers = 32 * 32
scores = comp * 128 * 128 * heads_layers
t_mm = scores * mm_per_score / MM
t1 = scores * fa1_step / NM
t2 = scores * fa2_step / NM
print(f"   score elements computed (causal, block-skipped): {scores:.4e}")
print(f"   A100 ideal: matmul {t_mm * 1e3:.1f} ms + non-matmul FA1 {t1 * 1e3:.1f} ms / FA2 {t2 * 1e3:.1f} ms")
print(f"   totals: FA1 {(t_mm + t1) * 1e3:.1f} ms, FA2 {(t_mm + t2) * 1e3:.1f} ms "
      f"(saving {(t1 - t2) * 1e3:.1f} ms, {1 - (t_mm + t2) / (t_mm + t1):.1%})")
t_full = total * 128 * 128 * heads_layers * (mm_per_score / MM + (fa2_step + 1) / NM)
print(f"   without block skipping (compute + mask all {total} blocks): {t_full * 1e3:.1f} ms")

print("=" * 72)
print("7. Backward (Alg 2): P = exp(S - L), D = rowsum(dO o O)")
Nb, db = 512, 64
Qb, Kb, Vb, dOb = (rng.standard_normal((Nb, db)) for _ in range(4))
Ob, Lb, _ = fa2_forward(Qb, Kb, Vb, 64, 64)
g = fa2_backward(Qb, Kb, Vb, Ob, dOb, Lb, 64, 64)
r = reference_backward(Qb, Kb, Vb, dOb)
for name, a, b in zip(("dQ", "dK", "dV"), g, r):
    print(f"   max |{name}_FA2 - {name}_ref| = {np.abs(a - b).max():.2e}")
print("   rebuilding P per score: FA1 exp(S - m) / l = 3 ops; FA2 exp(S - L) = 2 ops")

print("=" * 72)
print("8. GQA: 4 query heads share 1 KV head (Llama-3.1-8B ratio), implicit indexing")
hq, hk, Ng, dg = 8, 2, 256, 64
Qg = rng.standard_normal((hq, Ng, dg))
Kg, Vg = rng.standard_normal((hk, Ng, dg)), rng.standard_normal((hk, Ng, dg))
dOg = rng.standard_normal((hq, Ng, dg))
grp = hq // hk
dK_sum = np.zeros_like(Kg); dV_sum = np.zeros_like(Vg); err = 0.0
for h in range(hq):
    kv = h // grp                                # implicit index, no copy of K or V
    Oh, Lh, _ = fa2_forward(Qg[h], Kg[kv], Vg[kv], 64, 64)
    err = max(err, np.abs(Oh - standard_attention(Qg[h], Kg[kv], Vg[kv])[0]).max())
    _, dk, dv = fa2_backward(Qg[h], Kg[kv], Vg[kv], Oh, dOg[h], Lh, 64, 64)
    dK_sum[kv] += dk; dV_sum[kv] += dv          # sum over the implicitly duplicated heads
Krep, Vrep = np.repeat(Kg, grp, axis=0), np.repeat(Vg, grp, axis=0)
dK_rep = np.stack([reference_backward(Qg[h], Krep[h], Vrep[h], dOg[h])[1] for h in range(hq)])
print(f"   forward max error vs standard: {err:.2e}")
print(f"   dK summed over the {grp} heads vs explicit-copy reference: "
      f"{np.abs(dK_sum - dK_rep.reshape(hk, grp, Ng, dg).sum(1)).max():.2e}")
print(f"   K/V memory with implicit indexing: {hk} heads, not {hq} ({hq // hk}x less)")

print("=" * 72)
print("9. Why a few exps matter: exp time vs matmul time per score element, d = 128")
for gpu, mm, ex, src in (("H100 SXM", 989e12, 3.9e12, "FA3: 16 ops/clk/SM x 132 x 1830 MHz"),
                         ("B200", 2.25e15, 16 * 148 * 1850e6, "FA4: 16 ops/clk/SM x 148 x 1850 MHz")):
    print(f"   {gpu}: matmul {mm / 1e12:,.0f} TF, exp {ex / 1e12:.2f} TF ({src}) -> "
          f"ratio {mm / ex:.0f}x; 1 exp per score costs {(1 / ex) / (512 / mm):.0%} of the 512 matmul FLOPs")

# Try this:
#   1. Set B = 64 in section 1 (Bc = 64 < d = 128): FA1's rescale grows to 4*d/Bc = 8 per score.
#   2. In causal_blocks, try block size 64 or 256 at N = 8192: the masked share shrinks as T grows.
#   3. Replace NM = 19.5e12 with 67e12 (H100 FP32 non-tensor) and MM with 989e12 to redo section 3.
