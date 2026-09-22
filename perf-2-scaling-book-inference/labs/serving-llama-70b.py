"""Serving LLaMA 3-70B end to end: Scaling Book Part 8 ("Serving LLaMA 3-70B").

Run: python3 labs/serving-llama-70b.py   (numpy + stdlib, CPU, well under a second)

Reproduces every worked step of the chapter on TPU v5e, runs the book's printed
numpy roofline code (printing its points instead of plotting them), and then
redoes the sizing for H100s (our recomputation, not in the book).
"""
import math
import numpy as np

# LLaMA 3-70B (book's table)
L, D, F, N, K, H, V = 80, 8192, 28672, 64, 8, 128, 128256
P = 70e9  # the book rounds the parameter count to 70e9

# TPU v5e (book Parts 1, 7, 8)
V5E_BF16, V5E_INT8, V5E_HBM, V5E_BW = 1.97e14, 3.94e14, 16e9, 8.2e11
V5E_ICI = 9e10  # bidirectional ICI bandwidth the book uses in Part 8's T_ici check


def hdr(s):
    print(f"\n== {s} ==")


hdr("1. KV cache per token (int8)")
kv_tok = 2 * K * H * L  # bytes, int8 = 1 byte per value
print(f"2 * K * H * L = 2 * {K} * {H} * {L} = {kv_tok:,} B  (book: 160 kB)")
seq32 = 160e3 * 32768
print(f"32k sequence: 160e3 * 32,768 = {seq32 / 1e9:.2f} GB (book: 5.3 GB; exact bytes {kv_tok * 32768 / 1e9:.2f} GB)")
bs240 = 5.3e9 * 240
print(f"batch 240: 5.3e9 * 240 = {bs240 / 1e12:.2f} TB -> (70e9 + 1.3e12) / 16e9 = {(70e9 + 1.3e12) / 16e9:.1f} v5e chips (book: 86)")

hdr("2. Batch 32, 8,192 context, everything int8")
kv_total = 160e3 * 8192 * 32
tot = kv_total + P
print(f"KV = 160e3 * 8192 * 32 = {kv_total / 1e9:.1f} GB ; params = 70 GB ; total = {tot / 1e9:.1f} GB")
print(f"chips = {tot / 1e9:.1f}e9 / 16e9 = {tot / V5E_HBM:.2f} -> 4x2 (tight), 4x4 realistic")

hdr("3. Step time on 4x2 (int8 weights+KV, bf16 FLOPs)")
for chips, name in [(8, "4x2"), (16, "4x4")]:
    t_kv = kv_total / (chips * V5E_BW)
    t_par = P / (chips * V5E_BW)
    t_fl = 2 * 32 * P / (chips * V5E_BF16)
    step = t_kv + max(t_par, t_fl)
    print(f"{name}: KV {t_kv * 1e3:.2f} ms + max(params {t_par * 1e3:.2f} ms, FLOPs {t_fl * 1e3:.2f} ms) = {step * 1e3:.2f} ms"
          f" -> {32 / step:,.0f} tok/s total, {32 / step / chips:.0f} tok/s/chip")
print(f"book shortcut: 112e9 / (8 * 8.2e11) = {112e9 / (8 * V5E_BW) * 1e3:.1f} ms ; 32/0.017 = {32 / 0.017:.0f} ; /8 = {32 / 0.017 / 8:.0f}")
print(f"ICI check (2 axes): Y_max = 2 * F / 2200 = {2 * F / 2200:.1f}  (> 8 and > 16 chips: fine)")

hdr("4. Critical batch size on v5e")
print(f"bf16 w, bf16 FLOPs : {V5E_BF16 / V5E_BW:.0f}")
print(f"int8 w, bf16 FLOPs : {V5E_BF16 / V5E_BW / 2:.0f}")
print(f"int8 w, int8 FLOPs : {V5E_INT8 / V5E_BW / 2:.0f}")

hdr("5. Minimum slice at 8k context (book table; book KV/token values 324/162/81 kB)")
book_kv = {"bf16": 324e3, "int8": 162e3, "int4": 81e3}
exact_kv = {"bf16": 2 * kv_tok, "int8": kv_tok, "int4": kv_tok / 2}
bpp = {"bf16": 2, "int8": 1, "int4": 0.5}
slices = {"bf16": ("4x4", 16), "int8": ("4x2", 8), "int4": ("2x2", 4)}
dbl = {"bf16": ("4x8", 32), "int8": ("4x4", 16), "int4": ("2x4", 8)}
T_STEP, STEPS = 0.019, 512
rows = {}
for dt in ["bf16", "int8", "int4"]:
    psz = P * bpp[dt]
    name, n = slices[dt]
    free = n * 16 - psz / 1e9
    ncache = math.floor(free * 1e9 / (book_kv[dt] * 8192))
    ncache_exact = math.floor(free * 1e9 / (exact_kv[dt] * 8192))
    qps = ncache / (T_STEP * STEPS * n)
    rows[dt] = (psz, n, free, ncache, qps)
    print(f"{dt}: params {psz / 1e9:.0f} GB, KV/token {book_kv[dt] / 1e3:.0f} kB (exact {exact_kv[dt] / 1e3:.1f} kB),"
          f" min chips {psz / V5E_HBM:.2f} -> {name} = {n}, free {free:.0f} GB, caches@8k {ncache} (exact bytes: {ncache_exact})")
print("note: int4 min chips = 35/16 = 2.19; the book prints 2.81 (typo), the 2x2 answer is unchanged")

hdr("6. Step latency with HBM full, and a 512-token answer")
t_full = V5E_HBM / V5E_BW
print(f"16e9 / 8.2e11 = {t_full * 1e3:.1f} ms/step ; x 512 = {t_full * 512:.1f} s (book: 19 ms, 'about 9 s' using 0.019*512 = {0.019 * 512:.2f} s)")
print(f"int4 params only on 2x2: 35e9 / (4 * 8.2e11) = {35e9 / (4 * V5E_BW) * 1e3:.1f} ms (book: ~10 ms)")

hdr("7. QPS per chip = B / (step * 512 * N), step = 19 ms")
print(f"43 / (0.019 * 512) = {43 / (T_STEP * STEPS):.2f} / N")
for dt, (psz, n, free, nc, qps) in rows.items():
    print(f"{dt:5s} on {slices[dt][0]} (N={n:2d}): QPS/chip = {qps:.3f}")
print("book: 0.27 / 0.55 / 1.11 ; 'real numbers likely around 1/2 of this'")

hdr("8. Doubled topology")
for dt in ["bf16", "int8", "int4"]:
    name, n = dbl[dt]
    free = n * 16 - P * bpp[dt] / 1e9
    nc = math.floor(free * 1e9 / (book_kv[dt] * 8192))
    qps = nc / (T_STEP * STEPS * n)
    gain = qps / rows[dt][4]
    print(f"{dt:5s} on {name} (N={n:2d}): free {free:.0f} GB, batch {nc}, {nc / (T_STEP * STEPS):.2f}/N -> QPS/chip {qps:.3f} ({gain:.2f}x the min slice)")
print("book: 0.44 / 0.90 / 1.80 (14.39/32 = 0.4497; the book truncates to 0.44)")
print("QPS bar widths, log scale 0.1..2 QPS/chip:")
for q in [0.276, 0.553, 1.105, 0.45, 0.90, 1.80]:
    print(f"  {q:.3f} -> {100 * math.log10(q / 0.1) / math.log10(20):.1f}%")

hdr("9. Sharding bf16 on a 4x8 during generation")
print(f"pure model parallelism, 2 axes: Y = 2 * 28672 / 2200 = {2 * F / 2200:.1f} (so 16 chips ok, 32 not)")
print(f"small batch, B = 64: Y = F / (8 * B) = 28672 / 512 = {F / (8 * 64):.0f}")
B, Y = 64, 32
t_ici = 2 * B * D / V5E_ICI
t_hbm = 2 * D * F / (Y * V5E_BW)
t_math = 2 * B * D * F / (Y * V5E_BF16)
print(f"T_ici = 2*64*8192/9e10 = {t_ici * 1e6:.1f} us ; T_hbm = 2*8192*28672/(32*8.2e11) = {t_hbm * 1e6:.1f} us ;"
      f" T_math = {t_math * 1e6:.2f} us (book: 11 / 18 / 4 us) -> HBM-bound")

hdr("10. Prefill, evictions, prefill:generate ratio")
t_pre = 2 * P * 8192 / (16 * V5E_BF16 * 0.4)
print(f"prefill 8192 tokens on 16 chips at 40% MFU: 2*70e9*8192/(16*1.97e14*0.4) = {t_pre:.2f} s")
print(f"evictions: B*(P+G)/G = 32*(8192+4096)/4096 = {32 * (8192 + 4096) / 4096:.0f} tokens/step ; {32 / 4096:.4f} sequences/step")
ratio = 0.91 * 32 / (0.019 * 512)
print(f"P/0.91 = 32 G/(0.019*512) -> P = {ratio:.2f} G  (book: P = 3G)")

hdr("11. The book's printed numpy code, verbatim logic, printed not plotted")
num_chips = 16
bytes_per_param = 1
param_count = 70e9
param_size = bytes_per_param * param_count
hbm_bandwidth = 8.20E+11
flops = 1.97E+14


def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80


def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))


def get_max_batch_size(num_chips, sequence_length, param_size):
    batch_sizes = np.arange(1, 1024, 4)
    kv_sizes = kv_cache_size(sequence_length * batch_sizes)
    required_chips = min_topology(kv_sizes + param_size)
    max_idx = np.where(required_chips <= num_chips)[0][-1]
    return max_idx


def curve(sequence_length, fix_index=False):
    max_idx = get_max_batch_size(num_chips, sequence_length, param_size)
    top = int(np.arange(1, 1024, 4)[max_idx]) if fix_index else int(max_idx)
    batch_sizes = np.arange(1, 512, 1)[:top]
    kv_sizes = kv_cache_size(sequence_length * batch_sizes)
    kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)
    param_comms_time = np.asarray([param_size / (num_chips * hbm_bandwidth)] * batch_sizes.shape[0])
    flops_time = 2 * param_size * batch_sizes / (num_chips * flops)
    mlp_time = np.maximum(flops_time, param_comms_time)
    attn_time = kv_comms_time
    latency = 1000 * (mlp_time + attn_time)            # ms per step
    throughput = batch_sizes / (latency * num_chips)   # tokens / ms / chip
    return batch_sizes, latency, throughput, kv_comms_time, param_comms_time, flops_time, max_idx


for S in [2048, 8192, 32768]:
    b, lat, thr, *_, mi = curve(S)
    print(f"S={S:6d}: verbatim code -> max_idx {mi}, so batch sizes 1..{b[-1]} "
          f"(max fitting batch is actually {np.arange(1, 1024, 4)[mi]}; the index into a step-4 array is reused as a count)")

print("\nlatency (ms) vs throughput (tokens/s/chip), index fixed so batch runs to the true max:")
for S in [2048, 8192, 32768]:
    b, lat, thr, *_ = curve(S, fix_index=True)
    pick = [x for x in [1, 8, 16, 32, 48, 64, 96, 120, 137, 160, 240, 320, 400, 480, 511] if x <= b[-1]] + [int(b[-1])]
    pts = sorted(set(pick))
    print(f"S={S}: " + "  ".join(f"B{x}:({lat[x - 1]:.2f} ms, {thr[x - 1] * 1000:.0f})" for x in pts))
    for x in sorted({min(120, int(b[-1])), int(b[-1])}):
        print(f"   B=1 -> B={x}: cost per token / {thr[x - 1] / thr[0]:.0f}, latency x {lat[x - 1] / lat[0]:.2f}")

print("\nstacked breakdown at S=8192, 16 chips (ms): param load | KV load | FLOPs | step = KV + max(param, FLOPs)")
b, lat, thr, kvt, pt, ft, _ = curve(8192, fix_index=True)
for x in [1, 16, 32, 64, 120, 137]:
    i = x - 1
    bound = "compute" if ft[i] > pt[i] else "memory"
    print(f"B={x:3d}: {pt[i] * 1e3:5.2f} | {kvt[i] * 1e3:5.2f} | {ft[i] * 1e3:5.2f} | {lat[i]:5.2f} ms  MLP {bound}")
s_cross = (2 * param_size / flops) / (2 * 128 * 8 * 80) * hbm_bandwidth
print(f"KV load > FLOPs when S > 2*P/C * W_hbm / kv_bytes = {s_cross:,.0f} tokens (int8 KV); "
      f"{s_cross / 2:,.0f} with bf16 KV (book says 'greater than 2048')")

hdr("12. Same model on H100s (our recomputation, not in the book)")
H_BF16, H_FP8, H_BW, H_HBM = 9.89e14, 1.979e15, 3.35e12, 80e9
print(f"FLOPs per dollar (book's Part 8 table): H100 9.9e14 * 3600 / $10.8 = {9.9e14 * 3600 / 10.8:.2e} ;"
      f" v5e 1.97e14 * 3600 / $1.2 = {1.97e14 * 3600 / 1.2:.2e}")
for label, wbytes, kvbytes, C in [("BF16 weights + BF16 KV, bf16 math", 2, 2 * kv_tok, H_BF16),
                                  ("FP8 weights + FP8 KV, fp8 math", 1, kv_tok, H_FP8)]:
    print(f"-- {label}: params {P * wbytes / 1e9:.0f} GB, KV/token {kvbytes:,} B, 8k seq {kvbytes * 8192 / 1e9:.2f} GB,"
          f" B_crit {C / H_BW * wbytes / 2:.0f} --")
    for n in [1, 2, 4, 8]:
        free = n * H_HBM - P * wbytes
        if free <= 0:
            print(f"  {n}xH100: {n * 80} GB < {P * wbytes / 1e9:.0f} GB of params -> does not fit")
            continue
        B = math.floor(free / (kvbytes * 8192))
        floor_ms = P * wbytes / (n * H_BW) * 1e3
        t_kv = B * kvbytes * 8192 / (n * H_BW)
        t_par = P * wbytes / (n * H_BW)
        t_fl = 2 * B * P / (n * C)
        step = t_kv + max(t_par, t_fl)
        qps = B / (step * 512 * n)
        print(f"  {n}xH100: free {free / 1e9:4.0f} GB -> batch {B:3d} @8k ; batch-1 floor {floor_ms:5.2f} ms ;"
              f" full-batch step {t_kv * 1e3:.1f} + max({t_par * 1e3:.1f}, {t_fl * 1e3:.1f}) = {step * 1e3:.1f} ms ;"
              f" {B / step:,.0f} tok/s ; QPS/chip {qps:.2f} ({qps * 3600 / 10.8:,.0f} queries/$) ; 512-token answer {step * 512:.1f} s")
print(f"v5e for comparison, queries per $ at $1.2/chip-hour: bf16 4x8 {0.4497 * 3600 / 1.2:,.0f} ; int8 4x4 {0.899 * 3600 / 1.2:,.0f} ; int4 2x4 {1.799 * 3600 / 1.2:,.0f}")
print(f"HBM bytes/s per $/h: H100 {3.35e12 / 10.8:.2e} ; v5e {8.2e11 / 1.2:.2e}")
print("1xH100 FP8: the 10 GB left is before CUDA context, activations and allocator headroom; treat as no real room")
print(f"H100 B_crit: bf16 {H_BF16 / H_BW:.0f} ; fp8 w + bf16 math {H_BF16 / H_BW / 2:.0f} ; fp8 x fp8 {H_FP8 / H_BW / 2:.0f}")

# Try this:
# 1. 32k context: in section 11 the 16-chip int8 curve stops at batch 33 (the verbatim code's
#    max_idx is 8, so it only draws 1..8). KV loading dominates the step at every batch size.
# 2. int4 KV: change kv_cache_size to return bs * 128 * 8 * 80 (half of int8) and rerun; the max
#    batch at 8k roughly doubles and the step at max batch barely changes (HBM is still full).
# 3. Set bytes_per_param = 2 (bf16) in section 11: param load doubles to 10.7 ms and B_crit
#    goes back to 240, which a 16-chip slice can't fit at 8k context.
