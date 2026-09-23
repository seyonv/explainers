"""Matmul bounds lab: siboehm's CUDA matmul worklog, intro + Kernel 1.

Run: python3 labs/matmul-bounds.py   (stdlib only, CPU, well under a second)

Prints:
  1. siboehm's lower bound for a 4092^2 fp32 SGEMM on an RTX A6000
  2. the naive kernel's zero-cache traffic (548 GB)
  3. every row of his results table as % of cuBLAS and % of advertised peak
  4. our recomputation of the same bound for a Colab T4 and an H100
  5. one Llama-3.1-8B MLP up-projection at decode (B=1) vs prefill (B=4096)
"""

S = 4092  # siboehm's matrix size (his README and repo benchmark use 4096)
FP32 = 4  # bytes


def sgemm_bound(n, flops_peak, bw, bytes_per_el=FP32):
    """C = alpha*A@B + beta*C for n x n matrices."""
    flops = 2 * n**3 + n**2          # multiply+add per step of each dot product, plus the beta*C add
    reads = 3 * n**2 * bytes_per_el  # A, B and the old C, each read once
    store = n**2 * bytes_per_el      # the new C, written once
    t_math = flops / flops_peak
    t_mem = (reads + store) / bw
    return flops, reads, store, t_math, t_mem


print("== 1. siboehm's lower bound: 4092^2 fp32, RTX A6000 (30 TFLOP/s fp32, 768 GB/s advertised) ==")
flops, reads, store, t_math, t_mem = sgemm_bound(S, 30e12, 768e9)
minb = reads + store
print(f"FLOPs      2*{S}^3 + {S}^2          = {flops:,}  = {flops/1e9:.1f} GFLOP")
print(f"reads      3*{S}^2*4 B               = {reads/1e6:.1f} MB")
print(f"store      {S}^2*4 B                 = {store/1e6:.1f} MB")
print(f"minimum traffic                     = {minb/1e6:.1f} MB   (article: 268 MB; later text says 278 MB)")
print(f"compute at 30 TFLOP/s               = {t_math*1e3:.2f} ms  (article: 4.5 ms)")
print(f"memory  at 768 GB/s                 = {t_mem*1e3:.3f} ms (article: 0.34 ms)")
print(f"compute / memory                    = {t_math/t_mem:.1f}x  -> compute-bound while traffic < ~{t_math/t_mem:.0f}x the minimum (article: ~10x)")
print(f"min arithmetic intensity            = {flops/minb:.0f} FLOP/B;  A6000 ridge = 30e12/768e9 = {30e12/768e9:.1f} FLOP/B")
print(f"cuBLAS moves 500 MB -> {flops/500e6:.0f} FLOP/B by our division (article states ~245 FLOP/B), {500e6/minb:.2f}x the minimum")
t_tc = flops / 309e12
print(f"tensor cores 309 TFLOP/s            = {t_tc*1e3:.2f} ms  (article: 0.44 ms; now close to the {t_mem*1e3:.2f} ms memory floor)")

print("\n== 2. Naive kernel, zero caching ==")
per_thread = 2 * S + 1
naive = per_thread * S**2 * FP32
print(f"floats per thread  2*{S}+1           = {per_thread:,}  (one row of A, one column of B, one C)")
print(f"threads            {S}^2             = {S**2:,}")
print(f"traffic  {per_thread:,} * {S**2:,} * 4 B = {naive/1e9:.1f} GB  (article: 548 GB)")
print(f"vs minimum          {naive/1e9:.1f} GB / {minb/1e6:.1f} MB  = {naive/minb:,.0f}x")
print(f"at 768 GB/s that alone would take   = {naive/768e9:.2f} s")
t_naive = flops / 309.0e9
print(f"measured 309 GFLOP/s -> runtime     = {flops/1e9:.1f} / 309 = {t_naive:.3f} s  (article: 'about 0.5s')")
print(f"548 GB in {t_naive:.3f} s would need    = {naive/t_naive/1e9:,.0f} GB/s > 768 GB/s, so caches must absorb much of it")
print(f"profiler 15 GB/s GMEM over {t_naive:.3f} s = {15e9*t_naive/1e9:.1f} GB actually read from DRAM (our inference)")
print(f"309 GFLOP/s = {309/30000*100:.1f}% of the advertised 30 TFLOP/s")
for n in (4092, 4096):
    g = -(-n // 32)  # CEIL_DIV(n, 32)
    launched = g * g * 1024
    print(f"tile quantization at {n}: grid {g}x{g} = {g*g:,} blocks x 1024 = {launched:,} threads for {n*n:,} entries "
          f"-> {launched - n*n:,} idle ({(launched - n*n)/launched*100:.2f}%)")

print("\n== 3. Results table, A6000, 4092^2 (article rows + README-only rows 7, 8, 11) ==")
rows = [  # (kernel, name, GFLOP/s, in_article)
    ("1", "Naive", 309.0, True),
    ("2", "GMEM coalescing", 1986.5, True),
    ("3", "SMEM caching", 2980.3, True),
    ("4", "1D blocktiling", 8474.7, True),
    ("5", "2D blocktiling", 15971.7, True),
    ("6", "Vectorized mem access", 18237.3, True),
    ("7", "Bank conflicts (linearize)", 16213.4, False),
    ("8", "Bank conflicts (offset)", 16459.2, False),
    ("9", "Autotuning", 19721.0, True),
    ("10", "Warptiling", 21779.3, True),
    ("11", "Double buffering", 17278.3, False),
    ("0", "cuBLAS", 23249.6, True),
]
cub = 23249.6
print(f"{'#':>3} {'kernel':28s} {'GFLOP/s':>9} {'% cuBLAS':>9} {'% 30 TF':>8} {'x naive':>8}  where")
for k, name, g, art in rows:
    print(f"{k:>3} {name:28s} {g:9.1f} {g/cub*100:8.1f}% {g/30000*100:7.1f}% {g/309.0:7.1f}x  {'article' if art else 'README only'}")

print("\n== 4. Our recomputation: same 4092^2 fp32 bound on the reader's GPUs ==")
gpus = [
    ("A6000 fp32 (siboehm)", 30e12, 768e9, FP32),
    ("T4 fp32", 8.1e12, 320e9, FP32),
    ("H100 SXM fp32 (non-tensor)", 67e12, 3.35e12, FP32),
    ("H100 SXM bf16 tensor", 989e12, 3.35e12, 2),
]
print(f"{'GPU':28s} {'ridge F/B':>9} {'min bytes':>10} {'t_math':>9} {'t_mem':>9} {'ratio':>6} {'naive@bw':>9}")
for name, fl, bw, b in gpus:
    f_, r_, s_, tm, tb = sgemm_bound(S, fl, bw, b)
    nv = (2 * S + 1) * S**2 * b
    print(f"{name:28s} {fl/bw:9.1f} {(r_+s_)/1e6:8.1f}MB {tm*1e3:7.2f}ms {tb*1e3:7.3f}ms {tm/tb:5.1f}x {nv/bw:8.2f}s")
print("ratio = t_math / t_mem: an optimized kernel stays compute-bound while it moves < ratio x the minimum")
print("naive@bw = time to stream the zero-cache 548 GB (274 GB in bf16) at that GPU's bandwidth")

for n in (4096, 1024):
    f_, r_, s_, tm, tb = sgemm_bound(n, 8.1e12, 320e9)
    print(f"T4 at {n}^2 (the notebook sizes): 2*n^3 = {2*n**3/1e9:.2f} GFLOP, floor {2*n**3/8.1e12*1e3:.2f} ms at 8.1 TFLOP/s, memory {tb*1e3:.3f} ms")

print("\n== 5. Llama-3.1-8B MLP up-projection, bf16[B,4096] x bf16[4096,14336], H100 ==")
D, F = 4096, 14336
C, BW = 989e12, 3.35e12
print(f"H100 bf16 ops:byte = 989e12 / 3.35e12 = {C/BW:.0f}")
for B, label in ((1, "decode, batch 1"), (4096, "prefill, 4096 tokens")):
    fl = 2 * B * D * F
    by = 2 * (B * D + D * F + B * F)
    ai = fl / by
    tm, tb = fl / C, by / BW
    bound = "compute" if tm > tb else "memory"
    print(f"B={B:5d} ({label}): FLOPs 2*{B}*{D}*{F} = {fl:.4g}; bytes 2*({B}*{D}+{D}*{F}+{B}*{F}) = {by/1e6:.1f} MB; "
          f"AI = {ai:.1f} FLOP/B; t_math {tm*1e6:.2f} us vs t_mem {tb*1e6:.2f} us -> {bound}-bound, "
          f"{min(1, tm/tb)*100:.1f}% of peak FLOPs usable")

# Try this:
# 1. Set S = 16384 in section 1: the compute/memory ratio grows with n (FLOPs ~ n^3, bytes ~ n^2).
# 2. In section 5, try B = 64 and B = 325 to find where the up-projection crosses the ridge.
# 3. Replace 30e12 with the 38.7e12 the article's later sidenote gives for the A6000.
