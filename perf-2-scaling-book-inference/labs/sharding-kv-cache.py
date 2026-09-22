"""Sharding the KV cache lab: Scaling Book Part 7, "Sharding the KV cache" + Appendix C.

Run: python3 labs/sharding-kv-cache.py   (numpy, CPU, well under a second)

Part 1 simulates the book's batch-sharded attention algorithm on a Y x Z mesh of "devices"
(plain numpy arrays in a dict). KV heads are split Y ways, the batch Z ways:
KV[2, B_Z, S, K_Y, H]. Q comes out of the W_Q matmul sharded over heads on both axes
(N_YZ); an AllToAll over Z swaps that to batch sharding, attention runs against the
local KV, a second AllToAll swaps the output back to head sharding, then W_O and an
AllReduce. An AllToAll here is just a reshuffle of numpy slices between devices.
It asserts the result equals plain unsharded attention.

Part 2 does the same with sequence sharding (AllGather Q, Flash-style combine).
Part 3 is the Appendix C latency-bound calculator (book numbers for TPU v5e, then our
recomputation for an 8xH100 NVLink node with Llama-3.1-8B-like dims).
"""
import numpy as np

# ---------------- Part 1: batch-sharded decode attention ----------------
# Small shapes so it runs instantly. Y*Z = 4 devices.
B, D, N, K, H, S = 8, 64, 8, 2, 16, 32   # batch, model dim, query heads, KV heads, head dim, context
Y, Z = 2, 2                               # Y shards KV heads, Z shards the batch
M = N // K                                # query heads per KV head (book: M = N/K)
assert K % Y == 0, "head sharding caps at K ways: Y must divide K"
assert B % Z == 0 and M % Z == 0, "Z must divide the batch and M"

rng = np.random.default_rng(0)
X = rng.standard_normal((B, D))              # activations for the one new token per sequence
WQ = rng.standard_normal((D, N, H)) / np.sqrt(D)
WO = rng.standard_normal((N, H, D)) / np.sqrt(N * H)
Kc = rng.standard_normal((B, S, K, H))       # KV cache (already filled by earlier steps)
Vc = rng.standard_normal((B, S, K, H))


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def reference():
    """Unsharded decode attention with GQA: head n = k*M + m uses KV head k."""
    Q = np.einsum("bd,dnh->bnh", X, WQ).reshape(B, K, M, H)
    s = np.einsum("bkmh,bskh->bskm", Q, Kc) / np.sqrt(H)
    p = softmax(s, axis=1)
    O = np.einsum("bskm,bskh->bkmh", p, Vc).reshape(B, N, H)
    return np.einsum("bnh,nhd->bd", O, WO)


devices = [(y, z) for y in range(Y) for z in range(Z)]
bZ, kY, mZ = B // Z, K // Y, M // Z
head_block = N // (Y * Z)


def q_heads(y, z):
    """Heads this device owns in Q[B, N_YZ, H]: N split over Y first, then Z."""
    start = y * (N // Y) + z * head_block
    return np.arange(start, start + head_block)


def o_heads(y, z):
    """Heads owned in O[B, K_Y, M_Z, H]: its KV heads, and its Z slice of M within each."""
    ks = np.arange(y * kY, (y + 1) * kY)
    ms = np.arange(z * mZ, (z + 1) * mZ)
    return (ks[:, None] * M + ms[None, :]).ravel()


def batch_rows(z):
    return np.arange(z * bZ, (z + 1) * bZ)


# Device-local KV cache: KV[B_Z, S, K_Y, H]. Nothing else about KV is ever moved.
kv_local = {(y, z): (Kc[batch_rows(z)][:, :, y * kY:(y + 1) * kY],
                     Vc[batch_rows(z)][:, :, y * kY:(y + 1) * kY]) for (y, z) in devices}

bytes_moved = {"AllToAll Q": 0, "AllToAll O": 0}
EL = 2  # count bytes as bf16

# line 3: Q[B, N_YZ, H] = X[B, D] * W_Q[D, N_YZ, H]   (X is replicated)
Q_loc = {(y, z): np.einsum("bd,dnh->bnh", X, WQ[:, q_heads(y, z)]) for (y, z) in devices}

# line 4: Q[B_Z, N_Y, H] = AllToAll_{Z->B}(Q[B, N_YZ, H])
Q_bs = {}
for (y, z) in devices:
    parts = []
    for z2 in range(Z):                        # gather my batch rows from every Z-peer
        piece = Q_loc[(y, z2)][batch_rows(z)]
        parts.append(piece)
        if z2 != z:
            bytes_moved["AllToAll Q"] += piece.size * EL
    Q_bs[(y, z)] = np.concatenate(parts, axis=1)   # [B_Z, N_Y, H], heads in N_Y order

# lines 5-8: reshape to [B_Z, K_Y, M, H], attend to the local KV, softmax over S
O_bs = {}
for (y, z) in devices:
    Qr = Q_bs[(y, z)].reshape(bZ, kY, M, H)
    Kl, Vl = kv_local[(y, z)]
    s = np.einsum("bkmh,bskh->bskm", Qr, Kl) / np.sqrt(H)
    p = softmax(s, axis=1)
    O_bs[(y, z)] = np.einsum("bskm,bskh->bkmh", p, Vl)   # [B_Z, K_Y, M, H]

# line 9: O[B, K_Y, M_Z, H] = AllToAll_{Z->M}(O[B_Z, K_Y, M, H])
O_hs = {}
for (y, z) in devices:
    parts = []
    for z2 in range(Z):                        # collect every Z-peer's batch rows for my M slice
        piece = O_bs[(y, z2)][:, :, z * mZ:(z + 1) * mZ]
        parts.append(piece)
        if z2 != z:
            bytes_moved["AllToAll O"] += piece.size * EL
    O_hs[(y, z)] = np.concatenate(parts, axis=0)   # [B, K_Y, M_Z, H]

# lines 10-12: reshape to O[B, N_YZ, H], multiply by W_O shard, AllReduce the partial sums
partials = [np.einsum("bnh,nhd->bd", O_hs[d].reshape(B, -1, H), WO[o_heads(*d)]) for d in devices]
X_out = np.sum(partials, axis=0)

ref = reference()
err = np.abs(X_out - ref).max()
assert np.allclose(X_out, ref, atol=1e-10), err
print("== Part 1: batch-sharded attention (book's algorithm) ==")
print(f"mesh Y={Y} (KV heads) x Z={Z} (batch) = {Y * Z} devices;  B={B} N={N} K={K} M={M} H={H} S={S}")
print(f"each device holds KV[2, B/{Z}={bZ}, S={S}, K/{Y}={kY}, H={H}]  ->  1/{Y * Z} of the cache, no copies")
print(f"max |sharded - unsharded| = {err:.1e}   OK: equals plain attention")
kv_total = 2 * B * S * K * H * EL
print(f"bytes moved by the two AllToAlls (bf16): Q {bytes_moved['AllToAll Q']} B, "
      f"O {bytes_moved['AllToAll O']} B;  KV cache stays put ({kv_total} B total)")

# ---------------- Part 2: sequence sharding (AllGather Q, Flash-style accumulate) ----------------
P = Y * Z
assert S % P == 0
Qfull = np.einsum("bd,dnh->bnh", X, WQ).reshape(B, K, M, H)   # after the AllGather, every device has all of Q
m_all, l_all, o_all = [], [], []
for p_ in range(P):
    sl = slice(p_ * S // P, (p_ + 1) * S // P)                  # this device's slice of the context
    s = np.einsum("bkmh,bskh->bskm", Qfull, Kc[:, sl]) / np.sqrt(H)
    m = s.max(axis=1)                                         # running max per query
    e = np.exp(s - m[:, None])
    m_all.append(m)
    l_all.append(e.sum(axis=1))                               # softmax denominator piece
    o_all.append(np.einsum("bskm,bskh->bkmh", e, Vc[:, sl]))  # unnormalised output piece
m_g = np.max(m_all, axis=0)                                   # combine like Flash Attention
num = sum(o * np.exp(m - m_g)[..., None] for o, m in zip(o_all, m_all))
den = sum(l * np.exp(m - m_g) for l, m in zip(l_all, m_all))
O_seq = (num / den[..., None]).reshape(B, N, H)
X_seq = np.einsum("bnh,nhd->bd", O_seq, WO)
assert np.allclose(X_seq, ref, atol=1e-10)
print("\n== Part 2: sequence-sharded KV (S split 4 ways) ==")
print(f"max |seq-sharded - unsharded| = {np.abs(X_seq - ref).max():.1e}   OK")

# ---------------- Part 3: Appendix C latency-bound calculator ----------------
print("\n== Part 3: Appendix C, latency-bound collectives (book, TPU v5e) ==")
W_ICI, T_MIN = 4.5e10, 1e-6        # one-way ICI bytes/s; ~1 us fixed cost per hop (book)
print(f"latency-bound when (bytes / n_shards) / {W_ICI:.1e} < {T_MIN:.0e} s")
print(f"  per-shard threshold  W_ICI * 1us = {W_ICI * T_MIN / 1e3:.0f} kB")
for Yn in [4, 8, 16]:
    print(f"  {Yn:2d}-way: buffer < {Yn * W_ICI * T_MIN / 1e3:.0f} kB")
Dm = 8192
for Bn in [4, 16, 64, 256]:
    nbytes = Bn * Dm                  # int8 activations: 1 byte each
    t_bw = nbytes / 8 / W_ICI
    print(f"  B={Bn:3d}, D={Dm} int8: {nbytes / 1e3:7.1f} kB, per shard (8-way) "
          f"{nbytes / 8 / 1e3:6.1f} kB -> {t_bw * 1e6:5.2f} us/hop  "
          f"{'LATENCY-bound' if t_bw < T_MIN else 'bandwidth-bound'};  "
          f"latency-bound once Y > BD/45,000 = {Bn * Dm / 45000:.1f}")
print("  same thing with T_total = max(T_min*|X|/2, B/W_ICI), |X| = 8 (bidirectional ring):")
for Bn in [4, 16, 64, 256]:
    lat, bw = T_MIN * 8 / 2, Bn * Dm / W_ICI
    print(f"    B={Bn:3d}: latency term {lat * 1e6:.1f} us, bandwidth term {bw * 1e6:5.2f} us -> "
          f"{'latency' if lat > bw else 'bandwidth'} wins")

print("\n== Our recomputation: 8xH100 NVLink node, Llama-3.1-8B dims (not in the book) ==")
W_NV, G = 450e9, 8                 # per-GPU egress, GPUs per node (book Part 12)
Dl, Nl, Hl, Kl = 4096, 32, 128, 8
print(f"head sharding caps at K = {Kl} ways; an 8-GPU node needs exactly {G}: no batch sharding needed")
print(f"bytes the bandwidth term moves in 1 us: {W_NV * 1e-6 / 1e3:.0f} kB")
for Bn in [4, 16, 64, 256]:
    act = Bn * Dl * 2                 # X[B, D] bf16, the AllReduce / AllGather payload
    t_ag = act * (G - 1) / (G * W_NV)   # book's in-node AllGather: B(N-1)/(N W)
    q = Bn * Nl * Hl * 2              # Q[B, N, H] bf16, the AllToAll payload
    t_a2a = q * (G - 1) / (G * G * W_NV)
    print(f"  B={Bn:3d}: X[B,D] bf16 {act / 1e3:7.1f} kB -> AllGather bw term {t_ag * 1e6:6.3f} us;  "
          f"Q AllToAll {q / 1e3:7.1f} kB -> {t_a2a * 1e6:6.3f} us")
b_1us = 1e-6 * W_NV * G / (G - 1) / (Dl * 2)
print(f"  AllGather bandwidth term reaches 1 us only at B ~ {b_1us:.0f}")
kv_layer = 16 * 8192 * 2 * Kl * Hl * 2
q_layer = 16 * Nl * Hl * 2
print(f"  per layer at B=16, S=8192: KV {kv_layer / 1e6:.0f} MB vs Q {q_layer / 1e3:.0f} kB "
      f"-> KV is {kv_layer / q_layer:.0f}x bigger (why moving activations wins)")

# Try this:
# 1. MQA: set K = 1, Y = 1, Z = 4. Head sharding is impossible (1 KV head), so the whole
#    4-way split is over the batch, and the two AllToAlls carry everything.
# 2. Compare the B = 4 and B = 64 rows in Part 3: at B=64 the TPU per-shard buffer is 65.5 kB (> 45 kB,
#    bandwidth-bound); at B=4 it is 4.1 kB, about 11x under the 1 us floor.
# 3. Set Y = 4 with K = 2: the assert fires, which is the book's "limited to K-way" rule.
