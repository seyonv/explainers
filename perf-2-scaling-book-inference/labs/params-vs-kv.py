"""Params vs KV cache lab: Scaling Book Part 7 "What about memory?"

Run: python3 labs/params-vs-kv.py   (stdlib only, CPU, well under a second)

Parameter count (the book's table, untied input/output embeddings):
  FFW       = 3 * D * F * L                       (SwiGLU gate, up, down)
  vocab     = 2 * V * D                           (input + output embeddings)
  attention = (2 * D * N * H + 2 * D * K * H) * L (q + out, k + v)
KV cache for T tokens = 2 * bytes * H * K * L * T  (the 2 = keys and values)
"""

GB, GiB = 1e9, 2**30

MODELS = {
    # name: L, D, F, N, K, H, V
    "LLaMA 2-13B":  dict(L=40, D=5120, F=13824, N=40, K=40, H=128, V=32000),   # book, Part 7
    "Llama-3.1-8B": dict(L=32, D=4096, F=14336, N=32, K=8,  H=128, V=128256),  # course running example
    "LLaMA 3-70B":  dict(L=80, D=8192, F=28672, N=64, K=8,  H=128, V=128256),  # Part 8 uses 70e9 params
}
CHIPS = {"H100 (80 GB)": 80e9, "8x TPU v5e (128 GiB)": 128 * GiB}


def params(m):
    ffw = 3 * m["D"] * m["F"] * m["L"]
    vocab = 2 * m["V"] * m["D"]
    attn = (2 * m["D"] * m["N"] * m["H"] + 2 * m["D"] * m["K"] * m["H"]) * m["L"]
    return ffw, vocab, attn, ffw + vocab + attn


def kv_bytes(m, T, nbytes=2):
    return 2 * nbytes * m["H"] * m["K"] * m["L"] * T


# 1. The book's LLaMA 2-13B table
m = MODELS["LLaMA 2-13B"]
ffw, vocab, attn, total = params(m)
print("== LLaMA 2-13B parameters (book's table) ==")
print(f"FFW   5120^2 x 2.7 x 3 x 40             = {5120**2 * 2.7 * 3 * 40:.4e}  (exact with F=13824: {ffw:.4e})")
print(f"vocab 2 x 32000 x 5120                  = {vocab:.4e}")
print(f"attn  (2*5120*40*128 + 2*5120*40*128)*40 = {attn:.4e}")
print(f"total                                   = {total:.4e}")
w13 = total * 2
print(f"bf16 weights: {w13/GB:.2f} GB = {w13/GiB:.2f} GiB")
print(f"Adam's two fp32 moments alone: 8 B x {total:.3e} = {8*total/GB:.0f} GB (book: training 'around 100GB')")
act = 8192 * 5120 * 2
print(f"one activation, 8k prefill: 8192 x 5120 x 2 = {act:,} B = {act/1e6:.1f} MB = {act/2**20:.0f} MiB")
kv13_tok = kv_bytes(m, 1)
kv13_8k = kv_bytes(m, 8192)
print(f"KV per token: 2 x 2 x 128 x 40 x 40 = {kv13_tok:,} B = {kv13_tok/1024:.0f} KiB")
print(f"KV per 8k sequence: 8192 x 40 x 128 x 40 x 2 x 2 = {kv13_8k:,} B = {kv13_8k/GB:.2f} GB = {kv13_8k/GiB:.2f} GiB")
print(f"4 caches = {4*kv13_8k/GB:.2f} GB  vs weights {w13/GB:.2f} GB  -> sequences whose KV = weights: {w13/kv13_8k:.2f}")

# 2. Side-by-side for the three models (bf16)
print("\n== Three models, bf16 ==")
print(f"{'model':14} {'params':>8} {'weights GB':>10} {'KV/token':>10} {'KV 8k GB':>9} {'KV=weights at 8k':>17}")
for name, m in MODELS.items():
    p = params(m)[3]
    w = 70e9 * 2 if name == "LLaMA 3-70B" else p * 2
    t = kv_bytes(m, 1)
    s = kv_bytes(m, 8192)
    print(f"{name:14} {p/1e9:7.2f}B {w/GB:10.2f} {t:>8,} B {s/GB:9.3f} {w/s:17.2f}")
print("  (70B weights use the book's round 70e9 params -> 140 GB; exact count above)")
m70 = MODELS["LLaMA 3-70B"]
print(f"LLaMA 3-70B KV/token int8 = {kv_bytes(m70, 1, 1):,} B (book: 160kB); bf16 = {kv_bytes(m70, 1):,} B = "
      f"{kv_bytes(m70, 1)/1024:.0f} KiB (book table: 324kB)")

# 3. How many sequences fit
print("\n== Sequences that fit: floor((HBM - weights) / KV per sequence), bf16 unless i8 ==")
print(f"{'model':14} {'chip':22} {'free GB':>8} {'2k':>5} {'8k':>5} {'32k':>5}")
for name, m in MODELS.items():
    w = 70e9 * 2 if name == "LLaMA 3-70B" else params(m)[3] * 2
    if name == "Llama-3.1-8B":
        w = 16.06e9  # _facts.md value
    for chip, hbm in CHIPS.items():
        free = hbm - w
        if free <= 0:
            print(f"{name:14} {chip:22} {free/GB:8.2f}   weights alone don't fit")
            continue
        fits = [int(free // kv_bytes(m, T)) for T in (2048, 8192, 32768)]
        print(f"{name:14} {chip:22} {free/GB:8.2f} {fits[0]:5} {fits[1]:5} {fits[2]:5}")
m = MODELS["LLaMA 3-70B"]
for chip, hbm in CHIPS.items():  # Part 8's int8 setup: 70 GB of weights, int8 KV
    free = hbm - 70e9
    fits = [int(free // kv_bytes(m, T, 1)) for T in (2048, 8192, 32768)]
    print(f"{'LLaMA 3-70B i8':14} {chip:22} {free/GB:8.2f} {fits[0]:5} {fits[1]:5} {fits[2]:5}")

# 4. HBM budget, Llama-3.1-8B on one H100
print("\n== HBM budget: Llama-3.1-8B on H100, 8k sequences ==")
m = MODELS["Llama-3.1-8B"]
seq = kv_bytes(m, 8192)
for B in (1, 16, 32, 59, 60):
    used = 16.06e9 + B * seq
    tag = "OOM" if used > 80e9 else "fits"
    print(f"B={B:3}: weights 16.06 + KV {B} x {seq/GB:.3f} = {B*seq/GB:6.2f} -> total {used/GB:6.2f} GB  "
          f"free {max(0, 80e9-used)/GB:5.2f} GB  {tag}")

# 5. The book's v5e-8 table in consistent units
print("\n== LLaMA 2-13B on 8x v5e (128 GiB), consistent bytes ==")
for B in (16, 17):
    used = w13 + B * kv13_8k
    print(f"B={B}: {used/GB:.1f} GB = {used/GiB:.1f} GiB  {'fits' if used <= 128*GiB else 'OOM'}")

# Try this:
# 1. KV in int8 / fp8: call kv_bytes(m, T, nbytes=1). Every "fits" count roughly doubles and
#    "KV = weights" doubles (LLaMA 2-13B: 3.9 -> 7.8 sequences at 8k).
# 2. 128k context: kv_bytes(MODELS["Llama-3.1-8B"], 131072) = 17.2 GB, more than the 16.06 GB of
#    weights; only 3 such sequences fit on an H100.
# 3. Set K = N for Llama-3.1-8B (undo GQA) and watch the 8k fit on H100 fall from 59 to 14.
