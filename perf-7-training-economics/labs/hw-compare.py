"""Choosing hardware: which spec matters for which regime, and what a token costs on each GPU.

A calculator (no measurements). For eight datacenter GPUs it computes:
  1. the sparse-vs-dense trap (NVIDIA headline numbers are 2:4 sparse)
  2. the ops:byte ridge (dense FLOP/s / HBM bytes/s) = B_crit for BF16 weights
  3. $ per TB/s-hour (sets the decode price) and $ per dense PFLOP-hour (sets the prefill price)
  4. Llama-3.1-8B BF16 decode on one GPU, 2k context, with course 2's step formula
         step = B * KV_seq / BW + max(2 * B * N / C, W / BW)
     at batch 1 and at the KV-limited batch (HBM full of weights + KV)
  5. which GPU wins per regime, and whether Llama-3.1-70B BF16 fits on one GPU
  6. the MLPerf v5.0 Llama-2-70B arithmetic and the NVIDIA B200-vs-H100 cost claim

Specs: vendor spec pages (NVIDIA H100/H200/A100/L40S/DGX B200/GB200 NVL72, AMD MI300X/MI325X
datasheets), dense numbers. Prices: on-demand $/GPU-hr seen 2026-09-24 (Lambda, RunPod, Modal).
Every tok/s and $/M here is a ceiling (100% of peak bandwidth or FLOPs), not a measurement.

Run: python3 labs/hw-compare.py   (stdlib only, < 1 s)
GB = 1e9 bytes throughout.
"""

import math

GB, TB = 1e9, 1e12

# name: (BF16 dense TF, FP8 dense TF, FP4 dense TF, HBM GB, TB/s, TDP, scale-up link,
#        on-demand $/GPU-hr, provider, Modal $/hr)       None = not offered / not in our sources
GPUS = {
    "A100 80GB": (312, None, None, 80, 2.039, "400 W", "NVLink 600 GB/s", 2.79, "Lambda 8x", 2.50),
    "L40S":      (362, 733, None, 48, 0.864, "350 W", "PCIe Gen4 64 GB/s, no NVLink", 1.09, "RunPod", 1.95),
    "H100 SXM":  (989, 1979, None, 80, 3.35, "700 W", "NVLink 900 GB/s", 3.99, "Lambda 8x", 3.95),
    "H200":      (989, 1979, None, 141, 4.8, "700 W", "NVLink 900 GB/s", 4.59, "RunPod", 4.54),
    "B200":      (2250, 4500, 9000, 180, 8.0, "~14.3 kW per 8", "NVLink 1.8 TB/s", 6.69, "Lambda 8x", 6.25),
    "GB200":     (2500, 5000, 10000, 186, 8.0, "not listed", "NVLink 1.8 TB/s, 72-GPU domain", None, None, None),
    "MI300X":    (1307.4, 2614.9, None, 192, 5.3, "750 W", "Infinity Fabric 7 x 128 GB/s", None, None, None),
    "MI325X":    (1307.4, 2614.9, None, 256, 6.0, "1000 W", "Infinity Fabric 7 x 128 GB/s", None, None, None),
}
# The vendor page's headline (2:4 sparse) BF16 number, to show the trap
HEADLINE_SPARSE_BF16 = {"A100 80GB": 624, "L40S": 733, "H100 SXM": 1979, "H200": 1979,
                        "B200": 4500, "GB200": 5000, "MI300X": 2614.9, "MI325X": 2614.9}


def llama_params(L, h, F, a, K, hd, V):
    attn = h * a * hd * 2 + h * K * hd * 2
    return L * (attn + 3 * h * F + 2 * h) + 2 * V * h + h


N8 = llama_params(32, 4096, 14336, 32, 8, 128, 128256)       # 8.03e9
N70 = llama_params(80, 8192, 28672, 64, 8, 128, 128256)      # 70.55e9
W8 = 2 * N8                                                   # BF16 bytes
KV8 = 2 * 32 * 8 * 128 * 2                                    # 131,072 B = 128 KiB per token
CTX = 2048


def step(B, bw, C):
    return B * CTX * KV8 / bw + max(2 * B * N8 / C, W8 / bw)


def per_m(price, tok_s):
    return price / (tok_s * 3600) * 1e6


def main():
    print(f"Llama-3.1-8B: N = {N8 / 1e9:.2f}B, W = {W8 / GB:.2f} GB BF16, KV = {KV8:,} B/token,"
          f" {CTX * KV8 / GB:.3f} GB per 2k sequence. Llama-3.1-70B: N = {N70 / 1e9:.2f}B")

    print("\n1. Sparse vs dense (BF16 TFLOPS): the headline is 2:4 sparse, serving is dense")
    for g, s in HEADLINE_SPARSE_BF16.items():
        print(f"   {g:<10} headline {s:>7,.1f}  dense {GPUS[g][0]:>7,.1f}  ratio {s / GPUS[g][0]:.1f}x")

    print("\n2. Ridge (ops:byte) = dense FLOP/s / HBM B/s  [= B_crit for BF16 weights]")
    for g, v in GPUS.items():
        bf, f8 = v[0], v[1]
        r8 = f"{f8 * 1e12 / (v[4] * TB):5.0f}" if f8 else "    -"
        print(f"   {g:<10} BF16 {bf * 1e12 / (v[4] * TB):5.0f}   FP8 {r8}")

    print("\n3. Price per unit of the resource that binds (ours; on-demand, 2026-09-24)")
    print("   GPU        $/hr  provider     $/TB/s-hr  $/BF16 PF-hr  $/FP8 PF-hr | Modal $/hr  $/TB/s-hr")
    for g, v in GPUS.items():
        p = v[7]
        if p is None:
            print(f"   {g:<10} no on-demand price in our sources")
            continue
        f8 = f"{p / (v[1] / 1000):11.2f}" if v[1] else "          -"
        print(f"   {g:<10} {p:5.2f}  {v[8]:<11} {p / v[4]:9.2f}  {p / (v[0] / 1000):12.2f}  {f8} |"
              f" {v[9]:9.2f}  {v[9] / v[4]:9.2f}")

    print(f"\n4. Llama-3.1-8B BF16 decode, one GPU, {CTX} context, step formula (ceilings)")
    print("   GPU        W/BW ms  b1 ceil  b1 step   $/M b1 | B_kv  step ms   tok/s   $/M B_kv  max() term, KV share of step")
    rows = {}
    for g, v in GPUS.items():
        bf, bw, hbm, p = v[0] * 1e12, v[4] * TB, v[3] * GB, v[7]
        ceil1 = bw / W8
        t1 = step(1, bw, bf)
        bkv = math.floor((hbm - W8) / (CTX * KV8))
        tk = step(bkv, bw, bf)
        kvshare = bkv * CTX * KV8 / bw / tk
        bound = ("compute" if 2 * bkv * N8 / bf > W8 / bw else "weights") + f", KV {kvshare:.0%}"
        rows[g] = (ceil1, 1 / t1, bkv, bkv / tk, p)
        m1 = f"{per_m(p, 1 / t1):8.2f}" if p else "       -"
        mk = f"{per_m(p, bkv / tk):9.3f}" if p else "        -"
        print(f"   {g:<10} {W8 / bw * 1e3:7.2f}  {ceil1:7.0f}  {1 / t1:7.0f}  {m1} | {bkv:4d}  {tk * 1e3:7.2f}"
              f"  {bkv / tk:6,.0f}  {mk}  {bound}")
    h = rows["H100 SXM"]
    print(f"   check vs course facts: H100 b1 ceiling {h[0]:.0f} (209), step {h[1]:.0f} (205),"
          f" $/M {per_m(3.99, h[1]):.2f} (5.40), B_kv {h[2]} (238), tok/s {h[3]:,.0f} (9,973)")

    print("\n   Prefill ceiling (compute-bound): tokens/s = C / 2N, $/M input = price / that")
    for g, v in GPUS.items():
        tps = v[0] * 1e12 / (2 * N8)
        m = f"${per_m(v[7], tps):.4f}/M" if v[7] else "-"
        print(f"   {g:<10} {tps:9,.0f} tok/s   {m}")

    print("\n5. Winners per regime (among GPUs with a price)")
    priced = {g: r for g, r in rows.items() if r[4]}
    print(f"   fastest single stream (batch 1): {max(rows, key=lambda g: rows[g][1])}"
          f" ({max(r[1] for r in rows.values()):.0f} tok/s)")
    c1 = min(priced, key=lambda g: per_m(priced[g][4], priced[g][1]))
    ck = min(priced, key=lambda g: per_m(priced[g][4], priced[g][3]))
    print(f"   cheapest at batch 1: {c1}; cheapest at the KV-limited batch: {ck}")
    print("   Llama-3.1-70B BF16 weights on one GPU?")
    for g, v in GPUS.items():
        w = 2 * N70
        spare = v[3] * GB - w
        s = f"fits, {spare / GB:5.1f} GB left = {math.floor(spare / (CTX * 327680))} x 2k seqs" if spare > 0 \
            else f"no ({w / GB:.1f} GB > {v[3]} GB) -> needs {math.ceil(w / (v[3] * GB))}+ GPUs"
        print(f"   {g:<10} {s}")

    print("\n6. Published numbers, our arithmetic")
    for name, tps, price in (("8xH200 offline", 34988, 4.59), ("8xB200 offline", 98858, 6.69)):
        print(f"   MLPerf v5.0 Llama-2-70B {name}: {tps:,} tok/s = {tps * 3600 / 1e6:.1f}M tok/hr,"
              f" {tps / 8:,.0f}/GPU; at 8 x ${price} = ${8 * price:.2f}/hr -> ${per_m(8 * price, tps):.3f}/M")
    print(f"   B200/H200 throughput: {98858 / 34988:.2f}x; B200/H200 price: {6.69 / 4.59:.2f}x")
    print(f"   NVIDIA claim, GPT-OSS-120B: H100 vLLM $0.09/M at 66 tok/s/user vs B200 TRT-LLM $0.02/M"
          f" at 55 tok/s/user -> {0.09 / 0.02:.1f}x (different engine AND different interactivity)")
    kb = 192e9 - W8
    print(f"   Kiely's B200 (192 GB, ~5 PF FP8) vs DGX-derived (180 GB, 4.5 PF): B_kv for 8B would be"
          f" {math.floor(kb / (CTX * KV8))} vs {rows['B200'][2]}")


if __name__ == "__main__":
    main()

# Try this:
# 1. Set CTX = 32768: the KV-limited batch collapses (H100: 14 sequences) and $/M rises ~16x on H100;
#    the GPUs with the most HBM (MI325X, B200) keep the biggest batches: "buy GB for long context".
# 2. Replace W8 = 2 * N8 with W8 = N8 (FP8 weights) and use v[1] for C in step(): the weight read
#    halves, batch-1 tok/s roughly doubles, and B_crit halves, as course 2 says.
# 3. Put a price on MI300X (e.g. whatever your provider quotes) in GPUS and see where it lands.
