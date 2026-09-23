"""Multi-head latent attention (DeepSeek-V2 section 2.1), checked in numpy.

1. Weight absorption: attention computed the "explicit" way (rebuild per-head
   K and V from the cached latent c) equals the "absorbed" way (fold W^UK into
   the query side and W^UV into W^O, attend directly against c).
2. Why RoPE breaks absorption: rotate the keys by position and one merged
   matrix no longer works for every key.
3. KV cache per token (elements and BF16 bytes) for MHA / GQA / MQA / MLA.
4. Decode attention intensity (FLOPs per KV byte, one layer, one new token).

Run: python3 labs/mla.py   (stdlib + numpy, CPU, < 1 s)
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. absorption
# DeepSeek-V2-like shapes, scaled down (V2: d=5120, n_h=128, d_h=128, d_c=512, d_c'=1536, d_h^R=64)
d, n_h, d_h, d_c, d_cq, d_r, T = 64, 8, 16, 32, 48, 8, 10
f = np.float32
s = lambda *shape: (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(f)

h = rng.standard_normal((T, d)).astype(f)  # attention inputs of T tokens
W_DKV = s(d_c, d)                   # down-projection: h -> c^KV
W_UK = s(n_h * d_h, d_c)            # up-projection: c -> keys (all heads)
W_UV = s(n_h * d_h, d_c)            # up-projection: c -> values
W_DQ = s(d_cq, d)                   # query compression: h -> c^Q
W_UQ = s(n_h * d_h, d_cq)           # c^Q -> content queries
W_QR = s(n_h * d_r, d_cq)           # c^Q -> decoupled RoPE queries (per head)
W_KR = s(d_r, d)                    # h -> one shared RoPE key
W_O = s(d, n_h * d_h)               # output projection


def rope(x, pos):
    """Rotate consecutive pairs of the last axis by angle pos * theta_k."""
    k = x.shape[-1] // 2
    theta = pos[:, None] * (10000.0 ** (-np.arange(k) / k))[None, :]
    cos, sin = np.cos(theta).astype(f), np.sin(theta).astype(f)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


pos = np.arange(T, dtype=f)
c_kv = h @ W_DKV.T                                   # (T, d_c)       <- cached
k_r = rope(h @ W_KR.T, pos)                          # (T, d_r)       <- cached
t = T - 1                                            # decode the last token
c_q = W_DQ @ h[t]
q_c = (W_UQ @ c_q).reshape(n_h, d_h)
q_r = rope((W_QR @ c_q).reshape(n_h, d_r), np.full(n_h, t, dtype=f))
scale = 1 / np.sqrt(d_h + d_r)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# (a) explicit: rebuild per-head keys and values from c for every cached token
k_c = (c_kv @ W_UK.T).reshape(T, n_h, d_h)
v_c = (c_kv @ W_UV.T).reshape(T, n_h, d_h)
S_a = (np.einsum("hd,thd->ht", q_c, k_c) + q_r @ k_r.T) * scale
P_a = softmax(S_a)
o_a = np.einsum("ht,thd->hd", P_a, v_c)
u_a = W_O @ o_a.reshape(-1)

# (b) absorbed: q~_i = W^UK_i^T q_i lives in the d_c-dim latent space
W_UK_h = W_UK.reshape(n_h, d_h, d_c)
W_UV_h = W_UV.reshape(n_h, d_h, d_c)
q_lat = np.einsum("hdc,hd->hc", W_UK_h, q_c)         # (n_h, d_c)
S_b = (q_lat @ c_kv.T + q_r @ k_r.T) * scale         # scores straight against c
P_b = softmax(S_b)
o_lat = P_b @ c_kv                                   # (n_h, d_c): attend over c, not V
W_OV = np.einsum("xhd,hdc->hxc", W_O.reshape(d, n_h, d_h), W_UV_h)  # W^O_i W^UV_i, (n_h, d, d_c)
u_b = np.einsum("hxc,hc->x", W_OV, o_lat)

# (b') fully merged query matrix: W^UK_i^T W^UQ_i, precomputed once (App. C's wording)
W_UQ_h = W_UQ.reshape(n_h, d_h, d_cq)
W_QK = np.einsum("hdc,hde->hce", W_UK_h, W_UQ_h)     # (n_h, d_c, d_c')
S_c = (np.einsum("hce,e->hc", W_QK, c_q) @ c_kv.T + q_r @ k_r.T) * scale

print("1. Weight absorption (float32, n_h=8, d_h=16, d_c=32, d_h^R=8, 10 cached tokens)")
print(f"   scores  explicit vs absorbed          max |diff| = {np.abs(S_a - S_b).max():.1e}")
print(f"   scores  explicit vs merged W^UK^T W^UQ max |diff| = {np.abs(S_a - S_c).max():.1e}")
print(f"   output u explicit vs absorbed (W^O W^UV) max |diff| = {np.abs(u_a - u_b).max():.1e}")
print(f"   (score magnitude ~{np.abs(S_a).max():.2f}, output magnitude ~{np.abs(u_a).max():.2f})")

# ---------------------------------------------------------------- 2. RoPE on k^C
# Put RoPE on the compressed keys: k_j = R(j) W^UK c_j. The query-side matrix
# would have to be W^UK^T R(j)^T - different for every cached position j.
k_rot = np.stack([rope(k_c[j], np.full(n_h, j, dtype=f)) for j in range(T)])  # R(j) W^UK c_j
S_true = np.einsum("hd,thd->ht", q_c, k_rot)
# merge R(t) into the query side once (the best a single matrix can do), use it for every key
q_lat_t = np.einsum("hdc,hd->hc", W_UK_h, rope(q_c, np.full(n_h, -t, dtype=f)))
S_one = q_lat_t @ c_kv.T
err = np.abs(S_true - S_one)
print("\n2. RoPE on the compressed keys (no decoupling)")
print(f"   one merged matrix, right only for key position j = {t}: |diff| at j={t}: {err[:, t].max():.1e}, "
      f"max over other j: {err[:, :t].max():.2f}")
print("   -> every past key needs its own R(j): rebuild all keys each step, the saving is gone")

# ---------------------------------------------------------------- 3. KV per token
KiB, MiB = 1024, 1024 ** 2


def kv_elems(kind, l, n_h, d_h, n_g=None, d_c=512, d_r=64):
    if kind == "MHA": return 2 * n_h * d_h * l
    if kind == "GQA": return 2 * n_g * d_h * l
    if kind == "MQA": return 2 * d_h * l
    return (d_c + d_r) * l  # MLA: latent c^KV + shared RoPE key k^R


def fmt(b):
    return f"{b / MiB:6.2f} MiB" if b >= MiB else f"{b / KiB:6.1f} KiB"


rows = [
    ("DeepSeek-V2  MHA (hypothetical)", "MHA", 60, 128, 128, None),
    ("DeepSeek-V2  MLA (real)",         "MLA", 60, 128, 128, None),
    ("DeepSeek-V3  MHA (hypothetical)", "MHA", 61, 128, 128, None),
    ("DeepSeek-V3  GQA-8 (hypothetical)", "GQA", 61, 128, 128, 8),
    ("DeepSeek-V3  MQA (hypothetical)", "MQA", 61, 128, 128, None),
    ("DeepSeek-V3  MLA (real)",         "MLA", 61, 128, 128, None),
    ("Llama-3.1-8B MHA (hypothetical)", "MHA", 32, 32, 128, None),
    ("Llama-3.1-8B GQA-8 (real)",       "GQA", 32, 32, 128, 8),
    ("Llama-3.1-8B MQA (hypothetical)", "MQA", 32, 32, 128, None),
    ("Llama-3.1-8B MLA d_c=512 (hypothetical)", "MLA", 32, 32, 128, None),
]
print("\n3. KV cache per token (elements; bytes at BF16 = 2 B/element)")
kv = {}
for name, kind, l, nh, dh, ng in rows:
    e = kv_elems(kind, l, nh, dh, ng)
    kv[name] = e * 2
    print(f"   {name:42s} {e:>10,} el  {e * 2:>10,} B = {fmt(e * 2)}")
v3_mla, v3_mha = kv["DeepSeek-V3  MLA (real)"], kv["DeepSeek-V3  MHA (hypothetical)"]
ll_gqa, ll_mla = kv["Llama-3.1-8B GQA-8 (real)"], kv["Llama-3.1-8B MLA d_c=512 (hypothetical)"]
print(f"   V3: MLA / MHA = {v3_mla / v3_mha:.2%}  ({v3_mha / v3_mla:.1f}x smaller)")
print(f"   Llama: MLA vs GQA = {ll_gqa / ll_mla:.2f}x smaller; MLA = GQA with "
      f"{(512 + 64) / (2 * 128):.2f} groups (Table 1: 9/2 d_h l)")
free = 63.94e9
for name in ["Llama-3.1-8B GQA-8 (real)", "Llama-3.1-8B MLA d_c=512 (hypothetical)"]:
    print(f"   {name}: 8k-token sequences in 63.94 GB free HBM = {int(free // (kv[name] * 8192))}")

# DeepSeek 67B (HF config: 95 layers, 64 heads, 8 KV heads, hidden 8192 -> d_h 128)
ds67 = kv_elems("GQA", 95, 64, 128, 8)
v2 = kv_elems("MLA", 60, 128, 128)
print(f"   DeepSeek 67B GQA: {ds67:,} el; V2 MLA {v2:,} el -> -{1 - v2 / ds67:.1%} in elements")
print(f"   ...V2 at 6 bits vs 67B at 16 bits: {v2 * 6 / 8:,.0f} B vs {ds67 * 2:,} B -> "
      f"-{1 - (v2 * 6 / 8) / (ds67 * 2):.1%}  (paper: 'reduces the KV cache by 93.3%')")

# ---------------------------------------------------------------- 4. decode intensity
# One layer, one new token, per cached token j. FLOPs = multiply-adds x 2.
# Softmax ignored. Bytes = KV read at BF16. Absorbed MLA: per head, score = q~.c (d_c)
# + q^R.k^R (d_r), output accumulates p*c (d_c).
print("\n4. Decode attention, one layer, per cached token (BF16 KV; softmax ignored)")
ridge = 989e12 / 3.35e12


def show(name, flops, byts):
    print(f"   {name:44s} {flops:>12,} FLOPs / {byts:>6,} B = {flops / byts:8.1f} FLOPs/B")
    return flops / byts


show("Llama-3.1-8B GQA-8 (32 q heads, 8 KV heads)", 32 * 4 * 128, 2 * 8 * 128 * 2)
show("Llama-3.1-8B MLA absorbed (hypothetical)", 32 * (2 * 576 + 2 * 512), 576 * 2)
show("DeepSeek-V3 as MHA (hypothetical)", 128 * 4 * 128, 2 * 128 * 128 * 2)
i_v3 = show("DeepSeek-V3 MLA absorbed", 128 * (2 * 576 + 2 * 512), 576 * 2)
show("DeepSeek-V3 MLA explicit (rebuild K,V each step)",
     2 * 512 * 2 * 128 * 128 + 128 * (2 * 192 + 2 * 128), 576 * 2)
print(f"   H100 ridge = 989 TFLOPS / 3.35 TB/s = {ridge:.0f} FLOPs/B;  V3 MLA sits at {i_v3 / ridge:.0%} of it")
S = 8192
by, fl = S * 576 * 2, S * 128 * (2 * 576 + 2 * 512)
print(f"   V3 MLA, S = {S}, one layer, one sequence: read {by / 1e6:.2f} MB -> {by / 3.35e12 * 1e6:.2f} us; "
      f"{fl / 1e9:.2f} GFLOP -> {fl / 989e12 * 1e6:.2f} us at peak")
per_tok = 128 * 2 * 128 * 512 * 2  # q through W^UK and latent output through W^UV, per head
print(f"   plus fixed per-token absorbed projections (W^UK on q, W^UV on output): {per_tok / 1e6:.1f} MFLOPs/layer")

# Try this:
# - Set f = np.float64 at the top: the absorption diffs drop to ~1e-16 (it's exact algebra).
# - Change d_c in kv_elems() to 256 or 1024 and watch MLA cross GQA-2 / GQA-4.
# - In section 4, raise the number of heads (e.g. 256) and see absorbed MLA cross the H100 ridge.
