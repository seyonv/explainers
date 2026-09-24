"""Sizing a deployment: how many GPUs, and which layout.

A calculator (no measurements). Four questions, in order:
  1. Memory: do the weights plus enough KV cache fit?  -> minimum GPUs
  2. Traffic: N users x tok/s each at a context length -> KV needed, tok/s needed -> replicas
  3. Money: $/hr per layout and $ per million output tokens
  4. Blast radius: how much capacity one GPU failure takes down

The decode-step model is the same one labs/layouts.py uses (the pipeline card), so the
numbers agree with it:
    step = max(HBM bytes / BW, FLOPs / peak) + 2L all-reduces x (latency + ring bytes / 450 GB/s)
HBM bytes = this GPU's weight shard + its share of every sequence's KV cache.
All-reduce latency is ILLUSTRATIVE (NVIDIA publishes none): 1 us per ring hop, 2(n-1)
hops inside a node = 14 us at TP8, 6 at TP4, 2 at TP2 (course facts).

Memory budget follows vLLM: the engine may use gpu_memory_utilization (default 0.92,
vLLM engine-args docs) of each GPU; weights come out of that, the rest is KV cache.
vLLM also subtracts activations and CUDA graphs it measures at start-up; we leave those
out, like the scheduler card in course 5, so real KV room is somewhat smaller.

Run:  python3 labs/sizing.py                      # the card's tables
      python3 labs/sizing.py --model llama-70b --prec fp8 --gpu h200 --users 500 --tok-s 20 --ctx 4096
stdlib only, < 1 s.
"""
import argparse
import math

# ---- models (course facts) ----
def llama(L, D, F, N, K, H, V):
    params = L * (D * N * H * 2 + D * K * H * 2 + 3 * D * F + 2 * D) + 2 * V * D + D
    return dict(params=params, active=params, L=L, D=D, K=K, kv_token=2 * L * K * H * 2, moe=False)

MODELS = {
    "llama-8b": llama(32, 4096, 14336, 32, 8, 128, 128256),     # 8.03B, KV 131,072 B/token
    "llama-70b": llama(80, 8192, 28672, 64, 8, 128, 128256),    # 70.55B, KV 327,680 B/token
    # DeepSeek-V3: 671B total, 37B active; MLA KV = (512 + 64) x 61 layers x 2 B (ds-v3-params.py)
    "deepseek-v3": dict(params=671e9, active=37e9, L=61, D=7168, K=None,
                        kv_token=(512 + 64) * 61 * 2, moe=True),
}
BYTES = {"bf16": 2.0, "fp8": 1.0, "int4": 0.5}     # int4 = weight-only (W4A16), scales ignored
PEAK = {"bf16": 989e12, "fp8": 1979e12, "int4": 989e12}   # W4A16 computes in BF16

# ---- GPUs: H100 SXM (course facts), H200 (NVIDIA spec: 141 GB HBM3e, 4.8 TB/s, same compute)
GPUS = {"h100": dict(hbm=80e9, bw=3.35e12), "h200": dict(hbm=141e9, bw=4.8e12)}
W_NVL = 450e9                 # NVLink per direction per GPU (H100 and H200)
HOP = 1e-6                    # ILLUSTRATIVE latency per ring hop inside a node
UTIL = 0.92                   # vLLM default gpu_memory_utilization
PRICE = 3.99                  # $/GPU-hr, Lambda 8x H100 SXM on-demand, seen 2026-09-24
FAIL_GPU_H = 50_000           # Kiely p.196: ~1 failure per 50,000 GPU-hours (from Llama 3)


def allreduce(n, msg):
    return 0.0 if n == 1 else 2 * (n - 1) * HOP + 2 * (n - 1) / n * msg / W_NVL


def weights(m, prec):
    return m["params"] * BYTES[prec]


def replica(m, prec, gpu, tp, ctx, util=UTIL):
    """One TP group inside a node. Returns KV pool, sequences that fit, and a step(B) function."""
    g = GPUS[gpu]
    W = weights(m, prec)
    pool = util * g["hbm"] * tp - W                    # KV bytes across the group
    kv_seq = m["kv_token"] * ctx
    kv_split = min(tp, m["K"]) if m["K"] else tp        # KV heads can't split past K (Llama: 8)
    seqs = max(0, int(pool / tp // (kv_seq / kv_split))) if pool > 0 else 0

    def step(B):
        # MoE at big batch touches ~every expert, so read all weights (EP comms NOT modelled)
        mem = (W / tp + B * kv_seq / kv_split) / g["bw"]
        flop = 2 * m["active"] / tp * B / PEAK[prec]
        comm = 0.0 if m["moe"] else 2 * m["L"] * allreduce(tp, B * m["D"] * 2)
        return max(mem, flop) + comm
    return dict(W=W, pool=pool, seqs=seqs, step=step, kv_seq=kv_seq)


def min_gpus(m, prec, gpu, util=UTIL, headroom=0.0):
    """Smallest power of 2 whose budget holds the weights plus headroom x weights of KV."""
    n = 1
    while util * GPUS[gpu]["hbm"] * n - weights(m, prec) <= headroom * weights(m, prec):
        n *= 2
    return n


def size_traffic(m, prec, gpu, tp, users, tok_s, ctx, util=UTIL):
    """Replicas of TP=tp needed so every user holds ctx tokens of KV and gets >= tok_s."""
    r = replica(m, prec, gpu, tp, ctx, util)
    tpot_max = 1.0 / tok_s
    B = r["seqs"]
    while B > 0 and r["step"](B) > tpot_max:           # shrink batch until TPOT meets the target
        B -= 1
    if B == 0:
        return dict(r, B=0, reps=None)
    reps = math.ceil(users / B)
    return dict(r, B=B, tpot=r["step"](B), reps=reps, gpus=reps * tp,
                per_user=1 / r["step"](B))


def gb(x):
    return x / 1e9


def report(args):
    m, prec, gpu = MODELS[args.model], args.prec, args.gpu
    print(f"\n== your scenario: {args.model} {prec} on {gpu}, {args.users} users x "
          f"{args.tok_s} tok/s at {args.ctx}-token context (util {args.util}) ==")
    print(f"weights {gb(weights(m, prec)):.1f} GB; KV per sequence {gb(m['kv_token'] * args.ctx):.3f} GB; "
          f"KV for all users {gb(m['kv_token'] * args.ctx * args.users):.1f} GB; "
          f"output needed {args.users * args.tok_s:,} tok/s")
    print(f"minimum GPUs: weights fit {min_gpus(m, prec, gpu, args.util)}, "
          f"with 50% KV headroom {min_gpus(m, prec, gpu, args.util, 0.5)}")
    print(f"{'layout':<10}{'KV pool GB':>11}{'seqs fit':>9}{'batch':>7}{'TPOT ms':>9}{'replicas':>9}"
          f"{'GPUs':>6}{'$/hr':>9}{'$/M out':>9}")
    lay = "EP" if m["moe"] else "TP"
    for tp in (1, 2, 4, 8):
        s = size_traffic(m, prec, gpu, tp, args.users, args.tok_s, args.ctx, args.util)
        if s["reps"] is None:
            print(f"{lay}{tp:<8}{gb(max(s['pool'], 0)):>11.1f}{s['seqs']:>9}   doesn't fit or misses the TPOT target")
            continue
        cost = s["gpus"] * args.price
        print(f"{lay}{tp:<8}{gb(s['pool']):>11.1f}{s['seqs']:>9}{s['B']:>7}{s['tpot']*1e3:>9.2f}"
              f"{s['reps']:>9}{s['gpus']:>6}{cost:>9.2f}{cost / (args.users * args.tok_s * 3600 / 1e6):>9.2f}")
    print(f"  $ uses --price {args.price}/GPU-hr (default: Lambda's 8x H100 price; pass your own for H200)")
    if m["moe"]:
        print("  MoE: EP all-to-all is not modelled, and each GPU reads all experts at big batch;"
              " treat TPOT as a floor.")


def card():
    print(f"assumptions: vLLM gpu_memory_utilization {UTIL}; all-reduce latency 1 us/hop "
          f"(illustrative); ${PRICE}/GPU-hr (Lambda 8x H100, seen 2026-09-24)")

    # ---- 1. weights and minimum GPUs ----
    print("\n== 1. weights and minimum GPUs (powers of 2; 'fits' = weights < util x HBM x n; "
          "'+50%' = KV room >= half the weights, Kiely p.77) ==")
    print(f"{'model':<13}{'prec':<6}{'weights GB':>11}   {'H100 fits':>9}{'+50%':>6}{'KV GB':>8}"
          f"   {'H200 fits':>9}{'+50%':>6}{'KV GB':>8}")
    for name in MODELS:
        m = MODELS[name]
        for prec in BYTES:
            row = f"{name:<13}{prec:<6}{gb(weights(m, prec)):>11.1f}"
            for gpu in GPUS:
                n1 = min_gpus(m, prec, gpu)
                n2 = min_gpus(m, prec, gpu, headroom=0.5)
                kv = UTIL * GPUS[gpu]["hbm"] * n2 - weights(m, prec)
                row += f"   {n1:>9}{n2:>6}{gb(kv):>8.1f}"
            print(row)
    for gpu in GPUS:
        print(f"  {gpu}: budget per GPU {UTIL} x {gb(GPUS[gpu]['hbm']):.0f} GB = "
              f"{gb(UTIL * GPUS[gpu]['hbm']):.2f} GB; left outside it {gb((1-UTIL) * GPUS[gpu]['hbm']):.1f} GB")
    m70 = MODELS["llama-70b"]
    for tp in (1, 2, 4, 8):
        r = replica(m70, "fp8", "h100", tp, 4096)
        print(f"  70B FP8 TP{tp} on H100: {UTIL} x 80 x {tp} - 70.55 = {gb(r['pool']):6.1f} GB of KV "
              f"-> {r['seqs']} sequences of 4k")
    ds = MODELS["deepseek-v3"]
    for gpu, n in (("h200", 8), ("h100", 16)):
        pool = UTIL * GPUS[gpu]["hbm"] * n - weights(ds, "fp8")
        print(f"  DeepSeek-V3 FP8 on {n} x {gpu}: KV pool {gb(pool):.1f} GB = "
              f"{pool // (ds['kv_token'] * 4096):,.0f} sequences of 4k (MLA: {ds['kv_token']:,} B/token)")

    # ---- consistency check with layouts.py ----
    print("\n== check: same model as labs/layouts.py (util 1.0, 8k context, every sequence full) ==")
    for name, prec, tp, dp in (("TP8", "bf16", 8, 1), ("TP4 x DP2", "bf16", 4, 2),
                               ("FP8 TP2 x DP4", "fp8", 2, 4)):
        r = replica(m70, prec, "h100", tp, 8192, util=1.0)
        b1 = r["step"](1)
        tok = r["seqs"] / r["step"](r["seqs"]) * dp
        print(f"  {name:<14} seqs {r['seqs'] * dp:>4}  batch-1 {b1*1e3:5.2f} ms  total {tok:,.0f} tok/s"
              f"   (layouts.py: TP8 185 / 7.62 / 6,626; TP4xDP2 132 / 11.70 / 5,220; FP8 TP2xDP4 132 / 11.26 / 5,440)")

    # ---- 2. traffic example ----
    users, tok_s, ctx = 500, 20, 4096
    print(f"\n== 2. traffic: Llama-3.1-70B FP8, {users} concurrent chat users x {tok_s} tok/s, "
          f"{ctx}-token context (every user at full context: worst case) ==")
    kv_all = m70["kv_token"] * ctx * users
    print(f"  KV needed  {users} x {ctx} x {m70['kv_token']:,} B = {gb(kv_all):.1f} GB (BF16 KV)")
    print(f"  output     {users} x {tok_s} = {users * tok_s:,} tok/s; per-user TPOT <= {1000 / tok_s:.0f} ms")
    print(f"  weights    {gb(weights(m70, 'fp8')):.1f} GB per replica")
    print(f"{'GPU':<6}{'layout':<7}{'KV pool GB':>11}{'seqs fit':>9}{'batch':>7}{'TPOT ms':>9}"
          f"{'user tok/s':>11}{'replicas':>9}{'GPUs':>6}{'N+1 GPUs':>9}{'left after 1 failure':>22}")
    for gpu, tps in (("h100", (1, 2, 4, 8)), ("h200", (1, 2, 4, 8))):
        for tp in tps:
            s = size_traffic(m70, "fp8", gpu, tp, users, tok_s, ctx)
            if s["reps"] is None:
                print(f"{gpu:<6}TP{tp:<5}{gb(max(s['pool'], 0)):>11.1f}{s['seqs']:>9}   too few sequences / misses TPOT")
                continue
            left = (s["reps"] - 1) * s["B"]
            print(f"{gpu:<6}TP{tp:<5}{gb(s['pool']):>11.1f}{s['seqs']:>9}{s['B']:>7}{s['tpot']*1e3:>9.2f}"
                  f"{s['per_user']:>11.1f}{s['reps']:>9}{s['gpus']:>6}{s['gpus'] + tp:>9}"
                  f"{left:>12} of {users} users")
    out_m_hr = users * tok_s * 3600 / 1e6
    for g in (16, 20, 24):
        print(f"  {g} H100 = ${g * PRICE:.2f}/hr -> ${g * PRICE / out_m_hr:.2f} per M output tokens "
              f"at {users * tok_s:,} tok/s ({out_m_hr:.0f}M/hr)")

    # ---- 3. $/hr per layout, from the layouts.py numbers ----
    print("\n== 3. $/hr and tokens per $ for the layouts on the pipeline card (8k context, util 1.0) ==")
    rows = [("TP8", 8, 6626), ("TP4 x PP2", 8, 6000), ("TP4 x DP2", 8, 5220), ("FP8 TP2 x DP4", 8, 5440),
            ("TP16 over IB", 16, 6811), ("TP8 x PP2", 16, 13717), ("TP8 x DP2", 16, 13252)]
    print(f"{'layout':<15}{'GPUs':>5}{'$/hr':>8}{'tok/s':>8}{'M tok/hr':>9}{'$/M tok':>9}{'k tok per $':>12}")
    for name, n, tok in rows:
        hr = n * PRICE
        print(f"{name:<15}{n:>5}{hr:>8.2f}{tok:>8,}{tok * 3600 / 1e6:>9.2f}{hr / (tok * 3600 / 1e6):>9.3f}"
              f"{tok * 3600 / hr / 1e3:>12.1f}")

    # ---- 4. blast radius ----
    print(f"\n== 4. blast radius: ~1 failure per {FAIL_GPU_H:,} GPU-hours (Kiely p.196) ==")
    year = 8760
    print(f"  one 8-GPU node for a year = 8 x {year:,} = {8 * year:,} GPU-hours "
          f"-> {8 * year / FAIL_GPU_H:.2f} failures expected")
    print(f"{'group (GPUs that fail together)':<34}{'mean hours between failures':>28}{'= days':>8}")
    for name, n in (("TP2 replica", 2), ("TP4 replica", 4), ("TP8 replica (one node)", 8),
                    ("TP8 x PP2 / EP16 (two nodes)", 16), ("DeepSeek decode EP144", 144),
                    ("DeepSeek decode EP320 (paper)", 320)):
        h = FAIL_GPU_H / n
        print(f"{name:<34}{h:>28,.0f}{h / 24:>8.1f}")
    print("  capacity lost when one GPU of an 8-GPU node fails:")
    for name, tp in (("TP8 x DP1", 8), ("TP4 x DP2", 4), ("FP8 TP2 x DP4", 2)):
        print(f"    {name:<14} {tp} GPUs down = {tp / 8:.1%} of the node")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--prec", choices=BYTES, default="fp8")
    p.add_argument("--gpu", choices=GPUS, default="h100")
    p.add_argument("--users", type=int, default=500, help="concurrent users (streams)")
    p.add_argument("--tok-s", type=float, default=20, help="output tok/s each user must get")
    p.add_argument("--ctx", type=int, default=4096, help="tokens of KV per user")
    p.add_argument("--util", type=float, default=UTIL, help="vLLM gpu_memory_utilization")
    p.add_argument("--price", type=float, default=PRICE, help="$ per GPU-hour")
    a = p.parse_args()
    if a.model:
        report(a)
    else:
        card()


if __name__ == "__main__":
    main()

# Try this:
# 1. python3 labs/sizing.py --model llama-70b --prec fp8 --gpu h200 --users 500 --tok-s 20 --ctx 4096
#    H200's 141 GB lets FP8 TP2 hold 140 users per replica: 4 replicas, one 8-GPU node.
# 2. python3 labs/sizing.py --model llama-70b --prec fp8 --ctx 2048
#    Halving the context doubles the users per replica: TP4 fits 333, so 2 replicas = 8 H100.
# 3. python3 labs/sizing.py --model llama-70b --prec fp8 --tok-s 40
#    A 25 ms TPOT target now binds before memory: TP8's batch drops from 386 to 333.
