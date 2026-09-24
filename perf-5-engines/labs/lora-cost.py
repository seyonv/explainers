"""Serving many LoRA adapters: what an adapter costs next to Llama-3.1-8B.

Run: python3 labs/lora-cost.py   (stdlib only, CPU, instant)

LoRA replaces a weight W (d_in x d_out) with W + A.B, where A is d_in x r and B is r x d_out.
In serving we keep W shared and compute y = x.W + (x.A).B for each request's own A, B:
  - bytes per adapted matrix  = r * (d_in + d_out) * 2 bytes (BF16)
    (= 2 * r * d for a square d x d matrix like q_proj or o_proj)
  - FLOPs per token per matrix = 2 * r * (d_in + d_out)   (shrink x.A, then expand v.B)

Assumptions (ours, not from the papers): Llama-3.1-8B shapes from the course facts, BF16
adapters, the default target set q, k, v, o in all 32 layers (S-LoRA: "Typically, this
adjustment is only applied to the query, key, value, and output projection matrices"),
one H100 SXM (80 GB, 3.35 TB/s), decode step = one read of every weight it needs.
"""

GB, MB = 1e9, 1e6
L, D, F, KV_DIM = 32, 4096, 14336, 8 * 128     # layers, hidden, MLP width, K heads x head dim
N = 8.03e9                                      # parameters
BASE_BYTES = 16.06e9                            # BF16 weights
KV_TOK = 131072                                 # KV bytes per token (128 KiB)
HBM, BW = 80e9, 3.35e12                         # H100 SXM
RANKS = [8, 16, 64]

ATTN = {"q_proj": (D, D), "k_proj": (D, KV_DIM), "v_proj": (D, KV_DIM), "o_proj": (D, D)}
MLP = {"gate_proj": (D, F), "up_proj": (D, F), "down_proj": (F, D)}
TARGETS = ATTN                                  # try: TARGETS = {**ATTN, **MLP}

per_layer_dims = sum(di + do for di, do in TARGETS.values())   # sum of (d_in + d_out)


def adapter_params(r):
    return r * per_layer_dims * L


def adapter_bytes(r):
    return adapter_params(r) * 2


print("== 1. Bytes per adapter (Llama-3.1-8B, BF16, targets: " + ", ".join(TARGETS) + ") ==")
print("per layer: sum of (d_in + d_out) = " + " + ".join(str(di + do) for di, do in TARGETS.values())
      + f" = {per_layer_dims:,}")
print(f"{'rank':>5} {'params':>12} {'bytes':>10} {'% of base':>10} {'KV-token equiv':>15}")
for r in RANKS:
    p, b = adapter_params(r), adapter_bytes(r)
    print(f"{r:>5} {p:>12,} {b / MB:>8.2f} MB {100 * b / BASE_BYTES:>9.3f}% {b / KV_TOK:>12.0f} tok")
print(f"e.g. r=16: 16 x {per_layer_dims:,} x {L} x 2 B = {adapter_bytes(16) / MB:.2f} MB"
      f"  (KV-token equiv = bytes / 131,072 B per token of KV)")

print("\n== 2. How many adapters fit ==")
print(f"{'rank':>5} {'in 1 GB':>9} {'in 10 GB':>9} {'in 63.9 GB (all free HBM)':>26}")
for r in RANKS:
    b = adapter_bytes(r)
    print(f"{r:>5} {int(1e9 // b):>9,} {int(10e9 // b):>9,} {int(63.9e9 // b):>26,}")

print("\n== 3. Extra FLOPs per token vs the base model (2 x N) ==")
base_flops = 2 * N
for r in RANKS:
    f = 2 * adapter_params(r)
    print(f"r={r:>2}: 2 x {adapter_params(r):,} = {f / 1e6:.1f} MFLOP vs {base_flops / 1e9:.2f} GFLOP"
          f" = {100 * f / base_flops:.2f}%")

print("\n== 4. Memory: one merged copy per adapter vs one shared base + adapters ==")
r = 16
for n in [1, 4, 64, 2000]:
    merged = n * BASE_BYTES
    shared = BASE_BYTES + n * adapter_bytes(r)
    fits = f"leaves {(HBM - shared) / GB:.1f} GB for KV" if shared <= HBM else "does not fit"
    print(f"{n:>5} adapters, r=16: merged copies {merged / GB:>9.2f} GB"
          f" | shared base + adapters {shared / GB:>7.2f} GB ({fits})")
print(f"merged copies that fit in 80 GB with no room for KV: {int(HBM // BASE_BYTES)}")

print("\n== 5. Decode step cost: bytes read per step (memory-bound, batch of B requests) ==")
base_ms = BASE_BYTES / BW * 1e3
print(f"base weights, read once for the whole batch: {BASE_BYTES / GB:.2f} GB / 3.35 TB/s = {base_ms:.2f} ms")
for k in [1, 3, 8, 64]:
    extra = k * adapter_bytes(r)
    print(f"  + {k:>2} distinct r=16 adapters in the batch: {extra / MB:>8.1f} MB = {extra / BW * 1e3:.3f} ms"
          f"  (+{100 * extra / BASE_BYTES:.1f}% of the base read)")
print("One merged replica per adapter instead: each replica reads its own 16.06 GB per step,"
      " and a batch only holds requests for that one adapter.")

print("\n== 6. S-LoRA tensor parallelism: extra communication (S-LoRA Sec. 6.2) ==")
print("base: one all-reduce of B tokens x h      = 2(N-1)Bh/N")
print("LoRA: 3 all-gathers + 1 all-reduce of B x r = 3(N-1)Br/N + 2(N-1)Br/N = 5(N-1)Br/N")
for r in RANKS:
    print(f"r={r:>2}, h=4096: LoRA/base = 5r / 2h = {5 * r} / {2 * D} = {100 * 5 * r / (2 * D):.2f}%")

# Try this:
# 1. TARGETS = {**ATTN, **MLP}: adapting all 7 matrices makes an r=16 adapter 83.9 MB
#    (3.1x bigger), 0.52% of the base, and 0.52% extra FLOPs per token.
# 2. Set r = 64 in section 4 and 5: 64 distinct r=64 adapters read 6.98 GB per step, +43% of the
#    base read -- this is when adapter reads stop being "free" and adapter clustering pays.
# 3. Change BW to 2.0e12 (A100 80GB): the ratios in sections 3, 5 and 6 don't change, only the ms.
