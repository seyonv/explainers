"""Pipeline, data and tensor parallel layouts for serving Llama-3.1-70B.

A ceiling model (no measurements): each decode step costs
    max(HBM bytes / bandwidth, FLOPs / peak) + all-reduces + pipeline hops
on H100 SXM. Every sequence sits at an 8,192-token context. The batch is
whatever fits in the HBM left after weights. Communication is not overlapped.

The latency values are illustrative (NVIDIA publishes no NVLink hop latency). They
follow the collectives card and _facts.md: alpha = 1 us per ring hop (the Scaling
Book's TPU figure), so an n-GPU ring all-reduce inside a node pays 2(n-1) hops =
14 us at TP8; an all-reduce that spans nodes is taken as 28 us (illustrative 2x).
Change them and rerun.

Run: python3 labs/layouts.py   (stdlib only, < 1 s)
"""

# ---- model: Llama-3.1-70B (Meta config; numbers from _facts.md) ----
L, D, F, N, K, H, V = 80, 8192, 28672, 64, 8, 128, 128256
ATTN = D * N * H * 2 + D * K * H * 2          # W_Q, W_O + W_K, W_V
MLP = 3 * D * F                                # gate, up, down
PARAMS = L * (ATTN + MLP + 2 * D) + 2 * V * D + D   # + norms, untied embedding and LM head
KV_TOKEN = 2 * L * K * H * 2          # bytes per token, BF16 K and V = 327,680
CTX = 8192                           # tokens per sequence (worst case: all at 8k)
KV_SEQ = KV_TOKEN * CTX              # 2.68 GB
AR_PER_TOKEN = 2 * L                 # 2 all-reduces per layer forward (our inference from Megatron)

# ---- hardware: H100 SXM ----
HBM, BW = 80e9, 3.35e12
FLOPS = {"bf16": 989e12, "fp8": 1979e12}
W_NVL = 450e9                        # NVLink, per direction, per GPU
W_NODE = 400e9                       # IB egress per node: 8 NICs x 50 GB/s. Scaling Book: a
                                     # hierarchical cross-node all-reduce costs ~2 x bytes / 400e9

# ---- illustrative latencies (not published) ----
HOP = 1e-6                           # one hop inside a node (a PP send is 1 hop)
ALPHA_IB = 28e-6                     # one all-reduce that spans two nodes
HOP_IB = 2 * HOP                     # one PP send across nodes (illustrative 2x)


def allreduce(n, msg, link):
    if n == 1:
        return 0.0
    if link == "nvl":                # ring inside the node
        return 2 * (n - 1) * HOP + 2 * (n - 1) / n * msg / W_NVL
    return ALPHA_IB + 2 * msg / W_NODE   # spans nodes: node-egress bound


def layout(name, tp, pp, dp, prec="bf16", tp_link="nvl", pp_link="nvl"):
    """One replica = tp*pp GPUs; dp replicas. Returns a dict of results."""
    wbytes = PARAMS * (1 if prec == "fp8" else 2)
    gpus = tp * pp
    w_gpu = wbytes / gpus
    free_gpu = HBM - w_gpu
    kv_shard = min(tp, K)            # KV heads can't split beyond K=8; beyond that they replicate
    # sequences one replica can hold: per GPU, one sequence's KV takes KV_SEQ/(kv_shard*pp)
    seqs = int(free_gpu // (KV_SEQ / (kv_shard * pp))) if free_gpu > 0 else 0
    w_hop, a_hop = (W_NVL, HOP) if pp_link == "nvl" else (W_NODE, HOP_IB)

    def stage_time(mb):
        """Time for one pipeline stage to process a micro-batch of mb sequences."""
        mem = (w_gpu + mb * KV_SEQ / (kv_shard * pp)) / BW
        flop = 2 * PARAMS / gpus * mb / FLOPS[prec]
        comm = (AR_PER_TOKEN / pp) * allreduce(tp, mb * D * 2, tp_link)
        return max(mem, flop) + comm, mem, flop, comm

    def hop(mb):
        # activation to the next stage (each of the tp GPUs sends 1/tp of it, so a node's
        # NICs work in parallel); the last stage sends back only token ids (~free)
        return 0.0 if pp == 1 else (pp - 1) * (a_hop + mb * D * 2 / w_hop)

    # batch 1: one sequence walks every stage in turn; other stages sit idle
    s1, m1, f1, c1 = stage_time(1)
    lat1 = pp * s1 + hop(1)
    # big batch: fill KV; split into pp micro-batches so every stage stays busy
    B = seqs
    mb = B / pp
    sB, mB, fB, cB = stage_time(mb)
    cycle = pp * sB + hop(mb)        # time for every sequence in the replica to get one token
    tok_s = B / cycle * dp
    return dict(name=name, gpus=gpus * dp, w_gpu=w_gpu, free=free_gpu, seqs=seqs * dp,
                lat1=lat1, mem1=pp * m1, comm1=pp * c1, hop1=hop(1),
                tpot=cycle, tok_s=tok_s, tok_s_gpu=tok_s / (gpus * dp),
                memB=pp * mB, flopB=pp * fB, commB=pp * cB, hopB=hop(mb), B=B)


def show(rows, title):
    print(f"\n== {title} ==")
    print(f"{'layout':<22}{'GPUs':>5}{'wt/GPU GB':>10}{'free GB':>8}{'8k seqs':>8}"
          f"{'b=1 ms':>8}{'b=1 tok/s':>10}{'bigB TPOT ms':>13}{'total tok/s':>12}{'tok/s/GPU':>10}")
    for r in rows:
        free = f"{r['free']/1e9:8.1f}" if r["free"] > 0 else "   n/a  "
        print(f"{r['name']:<22}{r['gpus']:>5}{r['w_gpu']/1e9:>10.1f}{free}{r['seqs']:>8}"
              f"{r['lat1']*1e3:>8.2f}{1/r['lat1']:>10.0f}{r['tpot']*1e3:>13.2f}"
              f"{r['tok_s']:>12,.0f}{r['tok_s_gpu']:>10,.0f}")
    print("  breakdown, batch 1 (ms): weights+KV read | all-reduces | hops")
    for r in rows:
        print(f"    {r['name']:<20}{r['mem1']*1e3:7.2f} |{r['comm1']*1e3:6.2f} |{r['hop1']*1e3:6.3f}")
    print("  breakdown, big batch per token-step (ms): HBM read | FLOPs | all-reduces | hops  (B)")
    for r in rows:
        print(f"    {r['name']:<20}{r['memB']*1e3:7.2f} |{r['flopB']*1e3:6.2f} |{r['commB']*1e3:6.2f} |"
              f"{r['hopB']*1e3:6.3f}   B={r['B']}")


print(f"Llama-3.1-70B: {PARAMS/1e9:.2f}B params, BF16 {PARAMS*2/1e9:.1f} GB, FP8 {PARAMS/1e9:.1f} GB")
print(f"KV per token {KV_TOKEN:,} B; per 8k sequence {KV_SEQ/1e9:.2f} GB")
print(f"illustrative latencies: {HOP*1e6:.0f} us per hop -> all-reduce in node 2(n-1) hops "
      f"({14*HOP*1e6:.0f} us at TP8, {6*HOP*1e6:.0f} at TP4, {2*HOP*1e6:.0f} at TP2); across nodes {ALPHA_IB*1e6:.0f} us")

# ---- part 1: one 8xH100 node ----
node = [
    layout("TP8 (1 replica)", 8, 1, 1),
    layout("TP4 x PP2", 4, 2, 1),
    layout("TP4 x DP2", 4, 1, 2),
    layout("FP8 TP2 x DP4", 2, 1, 4, prec="fp8"),
]
show(node, "8 x H100, one node (NVLink)")

fp8 = PARAMS / 2 / 1e9
print(f"\nFP8 TP2 fit check: {PARAMS/1e9:.1f} GB / 2 = {fp8:.1f} GB per GPU, "
      f"free {80-fp8:.1f} GB/GPU -> {int((80-fp8)*2e9 // KV_SEQ)} BF16-KV sequences of {CTX} tokens per replica")

# ---- part 2: two nodes, 16 GPUs ----
two = [
    layout("TP16 over IB", 16, 1, 1, tp_link="ib"),
    layout("TP8 x PP2 (IB hop)", 8, 2, 1, pp_link="ib"),
    layout("TP8 x DP2 (replicas)", 8, 1, 2),
]
show(two, "16 x H100, two nodes (IB between them)")

tp8pp2 = two[1]["lat1"]
fixed16 = (PARAMS * 2 / 16 + KV_SEQ / K) / BW + AR_PER_TOKEN * (2 * D * 2 / W_NODE)
print(f"\nbreak-even IB all-reduce latency for TP16 to match TP8xPP2 at batch 1: "
      f"{(tp8pp2 - fixed16) / AR_PER_TOKEN * 1e6:.1f} us")
print(f"TP16 KV: only {K} KV heads, so KV is sharded 8 ways and each copy is held twice "
      f"-> {two[0]['seqs']} sequences vs {two[1]['seqs']} for TP8 x PP2")

# ---- part 3: pipeline bubble from unequal micro-batches (Sarathi-Serve §3.3 numbers) ----
def pipeline(stage_ms, iters, stages=2):
    """stage_ms[mb][it] = time for micro-batch mb's iteration it on each stage.
    A micro-batch's next iteration can't start until it leaves the last stage
    (it needs the token that stage produces)."""
    free = [0.0] * stages
    ready = [0.0] * len(stage_ms)
    busy = [0.0] * stages
    spans = [[] for _ in range(stages)]
    for it in range(iters):
        for mb in range(len(stage_ms)):
            t = ready[mb]
            for s in range(stages):
                start = max(t, free[s])
                t = start + stage_ms[mb][it]
                free[s] = t
                busy[s] += stage_ms[mb][it]
                spans[s].append(("AB"[mb] + str(it), start, t))
            ready[mb] = t
    end = max(free)
    return end, busy, spans


uni = [[200] * 3, [200] * 3]
end_u, busy_u, sp_u = pipeline(uni, 3)
mix = [[1150, 200, 200], [200, 200, 200]]
end_m, busy_m, sp_m = pipeline(mix, 3)
print("\n== pipeline bubble, 2 stages, 2 micro-batches, 3 iterations (Falcon-180B stage times from Sarathi-Serve) ==")
print(f"uniform 200 ms micro-batches: finish {end_u:.0f} ms, stage-1 idle {end_u-busy_u[0]:.0f} ms, "
      f"stage-2 idle {end_u-busy_u[1]:.0f} ms (fill/drain only)")
print(f"A starts with a 1150 ms prefill: finish {end_m:.0f} ms, stage-1 idle {end_m-busy_m[0]:.0f} ms, "
      f"stage-2 idle {end_m-busy_m[1]:.0f} ms")
print(f"  extra stage-1 idle vs uniform: {(end_m-busy_m[0])-(end_u-busy_u[0]):.0f} ms "
      f"(the 1150 - 200 = {1150-200} ms bubble)")
for nm, sp in (("uniform", sp_u), ("mixed", sp_m)):
    for s_, row in enumerate(sp):
        print(f"  {nm:<8}stage {s_+1}: " + "  ".join(f"{m} {a:.0f}-{b:.0f}" for m, a, b in row))

# ---- part 4: FSDP-style decode, for contrast ----
ag = 7 / 8 * PARAMS * 2 / W_NVL
print(f"\nFSDP-style decode on 8 GPUs: all-gather 7/8 of 141.1 GB per token = {ag*1e3:.0f} ms "
      f"vs {PARAMS*2/8/BW*1e3:.2f} ms to read a TP8 shard from HBM")

# Try this:
# 1. CTX = 2048: every layout holds 4x the sequences; FP8 TP2 x DP4 reaches 21.3k tok/s vs
#    TP8's 22.1k (97%) with the shortest big-batch TPOT, and TP4 x PP2 edges past TP8.
# 2. ALPHA_IB = 40e-6: TP16's batch-1 step grows to 9.15 ms, now slower than TP8 x PP2 (7.62 ms).
# 3. HOP = 5e-6: TP8 at batch 1 grows to 16.58 ms and FP8 TP2 (12.54 ms) becomes the fastest
#    layout on the node: all-reduce latency is TP's tax, and it grows with n.
