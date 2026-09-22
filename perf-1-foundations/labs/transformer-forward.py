"""The transformer forward pass: where the parameters live.

(a) Parameter breakdown of Llama-3.1-8B from its config.
(b) A tiny random-weight decoder in numpy (Llama-style: RMSNorm, RoPE, GQA,
    SwiGLU, untied LM head). Prints the tensor shape after every stage and
    the next-token probabilities.

Run: python3 labs/transformer-forward.py   (numpy only, CPU, < 1 s)
"""
import numpy as np

# ---------------------------------------------------------------- (a) counting
LLAMA_3_1_8B = dict(L=32, D=4096, F=14336, N=32, K=8, H=128, V=128256, tied=False)


def count_params(L, D, F, N, K, H, V, tied):
    attn = D * N * H + 2 * D * K * H + N * H * D      # Wq, Wk, Wv, Wo
    mlp = 3 * D * F                                    # W_gate, W_up, W_down
    norms = 2 * D                                      # two RMSNorm gains per layer
    layer = attn + mlp + norms
    emb = V * D                                        # token embedding table
    head = 0 if tied else D * V                        # LM head (unembedding)
    final_norm = D
    total = L * layer + emb + head + final_norm
    return dict(attn=attn, mlp=mlp, norms=norms, layer=layer, emb=emb,
                head=head, final_norm=final_norm, total=total, L=L)


def show_counts(name, cfg):
    c = count_params(**cfg)
    t = c["total"]
    L = c["L"]
    print(f"== (a) parameter breakdown: {name} ==")
    print(f"config: {cfg}")
    print(f"attention / layer : D*N*H + 2*D*K*H + N*H*D = {c['attn']:>15,}  ({c['attn']/1e6:.2f}M)")
    print(f"MLP / layer       : 3*D*F                   = {c['mlp']:>15,}  ({c['mlp']/1e6:.2f}M)")
    print(f"norms / layer     : 2*D                     = {c['norms']:>15,}")
    print(f"one layer         :                           {c['layer']:>15,}  ({c['layer']/1e6:.2f}M)")
    print(f"all {L} layers     :                           {L*c['layer']:>15,}")
    print(f"embedding V*D     :                           {c['emb']:>15,}  ({c['emb']/1e6:.2f}M)")
    print(f"LM head   D*V     :                           {c['head']:>15,}  ({c['head']/1e6:.2f}M)")
    print(f"final norm D      :                           {c['final_norm']:>15,}")
    print(f"TOTAL             :                           {t:>15,}  ({t/1e9:.2f}B)")
    print(f"share  MLP        : {L*c['mlp']/t:6.1%}   ({L*c['mlp']/1e9:.2f}B)")
    print(f"share  attention  : {L*c['attn']/t:6.1%}   ({L*c['attn']/1e9:.2f}B)")
    print(f"share  emb+head   : {(c['emb']+c['head'])/t:6.1%}   ({(c['emb']+c['head'])/1e9:.2f}B)")
    print(f"share  norms      : {(L*c['norms']+c['final_norm'])/t:8.3%}   ({L*c['norms']+c['final_norm']:,})")
    print(f"BF16 weights      : {2*t/1e9:.2f} GB  (2 bytes/param)")
    print()
    return c


show_counts("Llama-3.1-8B", LLAMA_3_1_8B)

# ---------------------------------------------------------------- (b) tiny model
TINY = dict(L=2, D=64, F=224, N=4, K=2, H=16, V=100, tied=False)
L, D, F, N, K, H, V = (TINY[k] for k in "LDFNKHV")
G = N // K                       # query heads per KV head (GQA group size)
tokens = np.array([17, 42, 7, 99, 3])   # 5 made-up token ids
T = len(tokens)

rng = np.random.default_rng(0)
def w(*shape):
    return rng.normal(0, 1 / np.sqrt(shape[0]), size=shape).astype(np.float32)

params = dict(
    emb=w(V, D) * np.sqrt(V),    # rows ~ unit scale
    layers=[dict(n1=np.ones(D, np.float32), wq=w(D, N * H), wk=w(D, K * H), wv=w(D, K * H),
                 wo=w(N * H, D), n2=np.ones(D, np.float32),
                 wg=w(D, F), wu=w(D, F), wd=w(F, D)) for _ in range(L)],
    nf=np.ones(D, np.float32),
    head=w(D, V),
)


def rmsnorm(x, g, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def rope(x):                      # x: [T, heads, H]
    half = H // 2
    freqs = 1.0 / (500000.0 ** (np.arange(half) / half))   # Llama 3 uses theta = 500,000
    ang = np.arange(T)[:, None] * freqs[None, :]           # [T, H/2]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def silu(x):
    return x / (1 + np.exp(-x))


def shp(label, a):
    print(f"  {label:<44} {str(list(a.shape)):>14}")


print(f"== (b) tiny random-weight decoder: {TINY}, T={T} tokens ==")
tc = count_params(**TINY)
print(f"tiny model params: {tc['total']:,}  (MLP {L*tc['mlp']/tc['total']:.1%}, "
      f"attention {L*tc['attn']/tc['total']:.1%}, emb+head {(tc['emb']+tc['head'])/tc['total']:.1%})")
shp("token ids", tokens)
x = params["emb"][tokens]
shp("embedding lookup  emb[V,D][ids]", x)

for i, p in enumerate(params["layers"]):
    print(f" block {i}")
    h = rmsnorm(x, p["n1"]);                         shp("RMSNorm", h)
    q = (h @ p["wq"]).reshape(T, N, H);              shp("q = h @ Wq[D, N*H]  -> [T, N, H]", q)
    k = (h @ p["wk"]).reshape(T, K, H);              shp("k = h @ Wk[D, K*H]  -> [T, K, H]", k)
    v = (h @ p["wv"]).reshape(T, K, H);              shp("v = h @ Wv[D, K*H]  -> [T, K, H]", v)
    q, k = rope(q), rope(k)
    k_rep, v_rep = np.repeat(k, G, axis=1), np.repeat(v, G, axis=1)   # share each KV head with G query heads
    scores = np.einsum("tnh,snh->nts", q, k_rep) / np.sqrt(H)
    scores = scores + np.triu(np.full((T, T), -np.inf), 1)           # causal mask
    shp("scores q.k / sqrt(H)  [N, T, T]", scores)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    ctx = np.einsum("nts,snh->tnh", probs, v_rep).reshape(T, N * H)
    shp("softmax . v  -> [T, N*H]", ctx)
    a = ctx @ p["wo"];                               shp("attn out = ctx @ Wo[N*H, D]", a)
    x = x + a;                                       shp("x + attn  (residual)", x)
    h = rmsnorm(x, p["n2"]);                         shp("RMSNorm", h)
    gate, up = h @ p["wg"], h @ p["wu"];             shp("gate, up = h @ W_gate/W_up[D, F]", up)
    m = silu(gate) * up;                             shp("SiLU(gate) * up", m)
    m = m @ p["wd"];                                 shp("down = m @ W_down[F, D]", m)
    x = x + m;                                       shp("x + mlp  (residual)", x)

x = rmsnorm(x, params["nf"]);                        shp("final RMSNorm", x)
logits = x @ params["head"];                         shp("logits = x @ LM head[D, V]", logits)
last = logits[-1]
p_next = np.exp(last - last.max()); p_next /= p_next.sum()
shp("softmax of last row -> next-token probs", p_next)
top = np.argsort(-p_next)[:5]
print("top-5 next tokens (random weights, so meaningless):")
for t in top:
    print(f"  token {t:>3}  p = {p_next[t]:.3f}")
print(f"sum of probs = {p_next.sum():.6f}; greedy pick = token {top[0]}")
print()

# ---------------------------------------------------------------- variations
print("== what-ifs on Llama-3.1-8B shapes ==")
for name, cfg in [("as MHA (K = N = 32)", dict(LLAMA_3_1_8B, K=32)),
                  ("F doubled (28672)", dict(LLAMA_3_1_8B, F=28672))]:
    c = count_params(**cfg)
    t = c["total"]
    print(f"{name:<22} total {t/1e9:.2f}B | attn/layer {c['attn']/1e6:.2f}M | "
          f"MLP {c['L']*c['mlp']/t:.1%} | attention {c['L']*c['attn']/t:.1%} | emb+head {(c['emb']+c['head'])/t:.1%}")

# Try this:
# 1. Set K=4 in TINY (MHA): the k/v shapes become [T, 4, 16] and attention params grow.
#    Do the same for LLAMA_3_1_8B (K=32) and watch attention/layer go 41.94M -> 67.11M.
# 2. Double F in LLAMA_3_1_8B: the MLP share climbs from 70.2% toward ~82%.
# 3. Set tied=True (one matrix for embedding and LM head, as small models often do):
#    the embeddings share drops by half.
