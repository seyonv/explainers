"""KV-quantization lab: FP8 KV and KIVI (Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit
Quantization for KV Cache", ICML 2024, arXiv 2402.02750).

Run: python3 labs/kv-quant.py   (numpy, CPU, about a second)

Part 1  Capacity and decode speed for Llama-3.1-8B on one H100 SXM, 8,192-token sequences,
        with the KV cache stored as BF16, FP8 (one scale per tensor), INT4 in groups of 32, or
        KIVI-2 (2-bit, groups of 32, residual window R = 128 kept in BF16).
          bits per value include the scale and zero-point (FP16 each, our assumption)
          KIVI's residual: worst case R tokens of K and R tokens of V in BF16
          sequences that fit in the 63.94 GB left after BF16 weights
          decode step = B*KV_seq/BW + max(2*B*P/C, W/BW)   (perf-2 step formula)
        Our recomputation, not in the paper. It ignores activations, allocator slack and the
        cost of (de)quantizing, so every "fit" and tok/s is an upper bound.

Part 2  SYNTHETIC keys with 4 outlier channels (seeded numpy): round-to-nearest quantization
        (KIVI's formula: z = min, s = (max - min)/(2^b - 1)) per-token vs per-channel, groups of 32.
        Relative errors ||X - X'||_F / ||X||_F of the keys, the logits q.K^T, and softmax(q.K^T).
        (The paper's Table 2 uses its own averaged relative-error metric on Llama-2-13B; our values
        are on a different scale and only the ratios are comparable in spirit.)

Part 3  SYNTHETIC values, per-token vs per-channel, error of the attention output A.V under a
        sparse attention (each query puts 90% of its weight on 8 tokens). Three cases: tokens with
        the same scale; token scales that vary (log-normal, sigma 1); and a stress case where the
        attended tokens are 10x smaller than the rest. The last two are our modelling choices,
        not measurements of a real model.

Part 4  The whole pipeline at 2 bits on the synthetic data: per-token K and V, KIVI without its
        window, KIVI with R = 128, error of the attention output.

Everything in parts 2-4 is synthetic and illustrative: it shows the mechanism, not model quality.
"""
import math
import numpy as np

# ---------------- Part 1: Llama-3.1-8B, H100 (course _facts.md) ----------------
L, K_HEADS, H = 32, 8, 128
VALS = 2 * L * K_HEADS * H      # K and V numbers per token = 65,536
P = 8.03e9
W = 16.06e9
FREE = 80e9 - W                 # 63.94 GB
BW = 3.35e12
C = 989e12
CTX = 8192
G = 32                          # group size
R = 128                         # KIVI residual window
KIB = 1024


def scheme(name, bits, group_overhead_bits, residual=0):
    """bits per value = payload + scale/zero bits spread over the group"""
    bpv = bits + group_overhead_bits
    quant_tokens = CTX - residual
    kv_seq = VALS * (quant_tokens * bpv / 8 + residual * 2)   # residual stays BF16
    return dict(name=name, bpv=bpv, tok_q=VALS * bpv / 8, kv_seq=kv_seq, eff_tok=kv_seq / CTX)


def step(B, kv_seq):
    t_kv = B * kv_seq / BW
    t_c = 2 * B * P / C
    return t_kv + max(t_c, W / BW), t_kv, t_c


def part1():
    rows = [
        scheme("BF16", 16, 0),
        scheme("FP8 per-tensor", 8, 0),                  # 2 scales per layer: negligible
        scheme("INT4 g32", 4, 32 / G),                    # FP16 scale + FP16 zero per 32 values
        scheme(f"KIVI-2 g32 R={R}", 2, 32 / G, residual=R),
    ]
    base = rows[0]
    print(f"=== Part 1: Llama-3.1-8B KV, {CTX:,}-token sequences, H100, free HBM {FREE/1e9:.2f} GB ===")
    print(f"{'scheme':<18}{'bits/val':>9}{'quant/tok':>11}{'eff/tok':>10}{'per seq':>10}{'x smaller':>10}"
          f"{'fit':>6}{'step':>9}{'tok/s':>8}{'  step @BF16 B':>14}")
    B0 = math.floor(FREE / base["kv_seq"])     # BF16's batch (59 at 8k)
    for d in rows:
        B = math.floor(FREE / d["kv_seq"])
        s, t_kv, t_c = step(B, d["kv_seq"])
        s59, _, _ = step(B0, d["kv_seq"])
        d.update(B=B, step=s, tput=B / s, t_c=t_c, s59=s59)
        print(f"{d['name']:<18}{d['bpv']:>9.2f}{d['tok_q']/KIB:>7.1f} KiB{d['eff_tok']/KIB:>6.1f} KiB"
              f"{d['kv_seq']/1e9:>7.4f} GB{base['kv_seq']/d['kv_seq']:>9.2f}x{B:>6}"
              f"{s*1e3:>6.1f} ms{B/s:>8,.0f}{s59*1e3:>11.2f} ms")
    k = rows[3]
    res = VALS * R * 2
    print(f"\nKIVI-2 arithmetic: 2 bits + (16+16)/{G} = {k['bpv']:.0f} bits/value -> "
          f"{VALS:,} x 3/8 = {k['tok_q']:,.0f} B = {k['tok_q']/KIB:.0f} KiB per quantized token "
          f"(16 KiB payload + 8 KiB scales)")
    print(f"  residual {R} tokens x 128 KiB = {res/2**20:.0f} MiB per sequence (worst case; the paper expects "
          f"R/2 of K + R of V = {VALS*(R//2+R)/2**20:.0f} MiB)")
    print(f"  per seq = {CTX-R:,} x {k['tok_q']/KIB:.0f} KiB + {res/2**20:.0f} MiB = {k['kv_seq']/1e9:.4f} GB; "
          f"fit floor({FREE/1e9:.2f} / {k['kv_seq']/1e9:.4f}) = {k['B']}")
    print(f"  step ({k['B']} x {k['kv_seq']/1e9:.4f} + 16.06) GB / 3.35 TB/s = {k['step']*1e3:.1f} ms; "
          f"compute 2*P*B/C = {k['t_c']*1e3:.2f} ms vs weight read {W/BW*1e3:.2f} ms"
          + (": right at the compute edge" if k['t_c'] > 0.95 * W / BW else ": memory-bound"))
    for d in rows:
        print(f"  {d['name']:<18} batch x{d['B']/rows[0]['B']:.2f}  throughput x{d['tput']/rows[0]['tput']:.2f}"
              f"  same-batch ({B0}) step speedup x{rows[0]['s59']/d['s59']:.2f}")
    n = 161 + 338    # the paper's ShareGPT workload: average prompt + output
    for r in (128, 32):
        eff = ((n - r) * 3 / 16 + r) / n
        print(f"  KIVI-2 at a {n}-token sequence, R = {r}: KV = {eff:.3f} of BF16 -> {1/eff:.2f}x smaller "
              f"(scales included, residual full)")
    return rows


# ---------------- quantizer (KIVI section 3.1) ----------------
def quant(X, bits, axis, g=G):
    """Asymmetric round-to-nearest, groups of g along `axis`. axis=1: per-token
    (a group is g channels of one token). axis=0: per-channel (g tokens of one channel)."""
    Xt = X if axis == 1 else X.T
    n, d = Xt.shape
    Y = Xt.reshape(n, d // g, g)
    z = Y.min(axis=2, keepdims=True)
    s = (Y.max(axis=2, keepdims=True) - z) / (2 ** bits - 1)
    s = np.where(s == 0, 1, s)
    Yq = np.round((Y - z) / s) * s + z
    Yq = Yq.reshape(n, d)
    return Yq if axis == 1 else Yq.T


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


T, D, NQ = 1024, 128, 64          # tokens, head dim, queries
OUT = [3, 37, 70, 101]            # outlier channels


def make_keys(rng):
    Kc = rng.normal(0, 1, (T, D))
    sign = np.array([1, -1, 1, -1])
    Kc[:, OUT] = sign * rng.normal(10, 3, (T, len(OUT)))   # fixed large channels
    return Kc


def part2(seed=0):
    rng = np.random.default_rng(seed)
    Kc = make_keys(rng)
    q = rng.normal(0, 1, (NQ, D))
    S = q @ Kc.T / math.sqrt(D)
    A = softmax(S)
    print(f"\n=== Part 2 (synthetic): keys {T}x{D}, channels {OUT} ~ +/-N(10, 3), others N(0, 1); "
          f"{NQ} queries, groups of {G} ===")
    print(f"{'bits':<6}{'grouping':<14}{'K error':>9}{'q.K^T error':>13}{'softmax error':>15}")
    out = {}
    for bits in (4, 2):
        for name, ax in (("per-token", 1), ("per-channel", 0)):
            Kq = quant(Kc, bits, ax)
            Sq = q @ Kq.T / math.sqrt(D)
            e = (rel(Kc, Kq), rel(S, Sq), rel(A, softmax(Sq)))
            out[(bits, name)] = e
            print(f"{bits:<6}{name:<14}{e[0]:>9.3f}{e[1]:>13.3f}{e[2]:>15.3f}")
        r = [out[(bits, 'per-token')][i] / out[(bits, 'per-channel')][i] for i in range(3)]
        print(f"      per-token / per-channel: K {r[0]:.1f}x, logits {r[1]:.1f}x, softmax {r[2]:.1f}x")
    # why: the range per-token groups must cover
    grp = Kc[0, :32]
    print(f"  token 0, channels 0-31: min {grp.min():.2f}, max {grp.max():.2f} -> 2-bit step "
          f"{(grp.max()-grp.min())/3:.2f} for normal values of size ~1")
    ch = Kc[:32, 0]
    print(f"  channel 0, tokens 0-31: min {ch.min():.2f}, max {ch.max():.2f} -> 2-bit step {(ch.max()-ch.min())/3:.2f}")
    return out


def sparse_attention(rng, top=8, mass=0.9):
    A = np.full((NQ, T), (1 - mass) / (T - top))
    hot = set()
    for i in range(NQ):
        idx = rng.choice(T, top, replace=False)
        A[i, idx] = mass / top
        hot.update(idx.tolist())
    return A, sorted(hot)


def make_values(rng, sigma, small=None):
    V = rng.normal(0, 1, (T, D))
    scale = np.exp(rng.normal(0, sigma, (T, 1))) if sigma else np.ones((T, 1))
    if small is not None:
        scale[small] = 0.1                       # the attended tokens are the small ones
    return V * scale


def part3(seed=0):
    rng = np.random.default_rng(seed)
    A, hot = sparse_attention(rng)
    top_share = np.sort(A, axis=1)[:, -8:].sum(axis=1).mean()
    print(f"\n=== Part 3 (synthetic): values {T}x{D}, sparse attention ({top_share:.0%} of each row on 8 tokens; "
          f"{(A < 1/T).mean():.1%} of entries below uniform 1/T) ===")
    print(f"{'token scales':<22}{'bits':>5}{'V err tok':>11}{'V err chan':>11}{'A.V err tok':>13}{'A.V err chan':>14}{'ratio':>8}")
    out = {}
    for label, sigma, small in (("all equal", 0, None), ("log-normal, sigma 1", 1.0, None),
                                ("attended tokens 0.1x", 0, hot)):
        V = make_values(rng, sigma, small)
        O = A @ V
        for bits in (4, 2):
            Vt, Vc = quant(V, bits, 1), quant(V, bits, 0)
            et, ec = rel(O, A @ Vt), rel(O, A @ Vc)
            out[(label, bits)] = (rel(V, Vt), rel(V, Vc), et, ec)
            print(f"{label:<22}{bits:>5}{rel(V, Vt):>11.3f}{rel(V, Vc):>11.3f}{et:>13.3f}{ec:>14.3f}{ec/et:>7.1f}x")
    return out


def part4(seed=0):
    rng = np.random.default_rng(seed)
    Kc = make_keys(rng)
    V = make_values(rng, 1.0)
    # queries that look mostly at the most recent tokens plus a few old ones (illustrative)
    q = rng.normal(0, 1, (NQ, D))
    S = q @ Kc.T / math.sqrt(D)
    recency = np.linspace(0, 3.0, T)            # logit bonus rising toward the newest token
    A = softmax(S + recency)
    O = A @ V
    recent = A[:, -R:].sum(axis=1).mean()
    print(f"\n=== Part 4 (synthetic): whole attention at 2 bits; {recent:.0%} of attention on the last {R} tokens ===")

    def run(kax, vax, window):
        n = T - window
        Kq, Vq = Kc.copy(), V.copy()
        Kq[:n] = quant(Kc[:n], 2, kax)
        Vq[:n] = quant(V[:n], 2, vax)
        Aq = softmax(q @ Kq.T / math.sqrt(D) + recency)
        return rel(O, Aq @ Vq)

    res = {
        "per-token K and V": run(1, 1, 0),
        "per-channel K and V": run(0, 0, 0),
        "KIVI, no window": run(0, 1, 0),
        f"KIVI, R = {R}": run(0, 1, R),
    }
    for k, v in res.items():
        print(f"  {k:<22} output error {v:.3f}")
    return res


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()

# Try this:
#   CTX = 32768 at the top: KIVI's 16 MiB window matters less, and the gap over BF16 grows to 5.24x
#   part2(seed=1)                 # another random draw: same ordering
#   in make_keys, set the outlier mean to 0 (N(0, 3)): per-channel's advantage on keys shrinks
