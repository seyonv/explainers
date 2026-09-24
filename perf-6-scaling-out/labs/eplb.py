"""Expert parallelism: bytes per token, DeepEP timings and the EPLB balancer.

A calculator and simulator for the card "Expert parallelism: dispatch, combine
and EPLB". Stdlib + numpy, CPU only, well under a second.

  1. Bytes each token moves per MoE layer in DeepSeek-V3 (dispatch FP8, combine BF16)
  2. DeepEP V1 low-latency table (H800): our bytes / its latency vs its bandwidth column
  3. What those latencies cost per decode step if nothing overlaps them
  4. EP vs TP: bytes per token per layer as the GPU count grows (our recomputation)
  5. EPLB, a line-for-line numpy port of deepseek-ai/EPLB eplb.py:
     the README example reproduced exactly, then per-GPU loads, hierarchical vs global
  6. A bigger skewed load (illustrative): 256 experts on 32 GPUs (prefill) and 144 GPUs (decode)

Run:  python3 labs/eplb.py
"""
import numpy as np

KiB = 1024

# ---------------------------------------------------------------------------
# 1. Bytes per token per MoE layer (DeepSeek-V3 shapes)
# ---------------------------------------------------------------------------
HIDDEN = 7168          # V3 hidden size
TOPK = 8               # routed experts per token
MAX_NODES = 4          # node-limited routing: a token reaches at most 4 nodes
FP8, BF16 = 1, 2       # bytes per value
SCALE_BYTES = (HIDDEN // 128) * 4   # one FP32 scale per 1x128 tile (our assumption on the scale format)

disp_copy = HIDDEN * FP8                 # one FP8 copy of the hidden vector
disp_copy_scaled = disp_copy + SCALE_BYTES
comb_copy = HIDDEN * BF16                # one BF16 expert output coming back
print("== 1. Bytes per token per MoE layer (DeepSeek-V3: hidden 7168, top-8) ==")
print(f"dispatch, one copy  : 7168 x 1 B (FP8)          = {disp_copy:>7,} B")
print(f"  + FP32 scales     : 56 tiles x 4 B            = {SCALE_BYTES:>7,} B  -> {disp_copy_scaled:,} B")
print(f"combine, one copy   : 7168 x 2 B (BF16)         = {comb_copy:>7,} B")
print(f"dispatch x 8 experts: {TOPK * disp_copy:>7,} B = {TOPK * disp_copy / KiB:.0f} KiB   (with scales {TOPK * disp_copy_scaled:,} B)")
print(f"combine  x 8 experts: {TOPK * comb_copy:>7,} B = {TOPK * comb_copy / KiB:.0f} KiB")
print(f"round trip          : {TOPK * (disp_copy + comb_copy):>7,} B = {TOPK * (disp_copy + comb_copy) / KiB:.0f} KiB")
print(f"cross-node (IB) cap, node-limited to 4 nodes: dispatch {MAX_NODES * disp_copy:,} B, "
      f"combine {MAX_NODES * comb_copy:,} B (V3 forwards once per node over IB, then NVLink)")
print(f"whole batch at 4,096 tokens (DeepEP normal-kernel setting): dispatch "
      f"{4096 * TOPK * disp_copy / 1e6:.0f} MB, combine {4096 * TOPK * comb_copy / 1e6:.0f} MB per layer")

# ---------------------------------------------------------------------------
# 2. DeepEP V1 low-latency kernels on H800 (docs/legacy.md): 128 tokens, top-8
# ---------------------------------------------------------------------------
print("\n== 2. DeepEP V1 low-latency kernels, H800 + CX7 (docs/legacy.md) ==")
EP = [8, 16, 32, 64, 128, 256]
DISP_US = [77, 118, 155, 173, 192, 194]
DISP_GBS = [98, 63, 48, 43, 39, 39]
COMB_US = [114, 195, 273, 314, 369, 360]
COMB_GBS = [127, 74, 53, 46, 39, 40]
TOK = 128
disp_bytes = TOK * TOPK * disp_copy_scaled
comb_bytes = TOK * TOPK * comb_copy
print(f"bytes per GPU per call: dispatch 128 x 8 x {disp_copy_scaled} = {disp_bytes / 1e6:.2f} MB, "
      f"combine 128 x 8 x {comb_copy} = {comb_bytes / 1e6:.2f} MB")
print(f"{'EP':>4} {'disp us':>8} {'ours GB/s':>9} {'table':>6} | {'comb us':>8} {'ours GB/s':>9} {'table':>6}")
for e, du, dg, cu, cg in zip(EP, DISP_US, DISP_GBS, COMB_US, COMB_GBS):
    print(f"{e:>4} {du:>8} {disp_bytes / du / 1e3:>9.1f} {dg:>6} | {cu:>8} {comb_bytes / cu / 1e3:>9.1f} {cg:>6}")
print(f"growth EP8 -> EP256: dispatch {DISP_US[-1] / DISP_US[0]:.2f}x, combine {COMB_US[-1] / COMB_US[0]:.2f}x")

# ---------------------------------------------------------------------------
# 3. What those latencies cost a decode step if nothing hides them
# ---------------------------------------------------------------------------
MOE_LAYERS = 58
print("\n== 3. Communication per decode step, 58 MoE layers, no overlap (our recomputation) ==")
for e, du, cu in zip(EP, DISP_US, COMB_US):
    print(f"EP{e:<4} (dispatch {du} + combine {cu}) us x 58 = {(du + cu) * MOE_LAYERS / 1e3:5.1f} ms")
print("DeepSeek's service averages 20-22 tok/s per user, i.e. about 45-50 ms per token (Day 6 post)")

# ---------------------------------------------------------------------------
# 4. EP vs TP: bytes crossing the interconnect per token per MoE layer
# ---------------------------------------------------------------------------
print("\n== 4. EP vs TP, total bytes on the wire per token per MoE layer (our recomputation) ==")
print("TP: row-parallel output all-reduce, ring, BF16: n GPUs each send 2(n-1)/n x 7168 x 2 B")
print("EP: 8 copies out (FP8) and back (BF16); a copy is free if the expert is on the token's own GPU")
print(f"{'n GPUs':>7} {'TP ring':>10} {'EP top-8':>10} {'EP/TP':>6}")
for n in [8, 16, 32, 64]:
    tp = n * 2 * (n - 1) / n * HIDDEN * BF16
    ep = TOPK * (n - 1) / n * (disp_copy + comb_copy)
    print(f"{n:>7} {tp / 1e3:>8.1f}kB {ep / 1e3:>8.1f}kB {ep / tp:>6.2f}")

# ---------------------------------------------------------------------------
# 5. EPLB: numpy port of deepseek-ai/EPLB eplb.py (same steps, same greedy rules)
# ---------------------------------------------------------------------------
def balanced_packing(weight, num_packs):
    """Sort items heaviest first; each goes to the lightest pack that still has room (n/m items per pack)."""
    num_layers, n = weight.shape
    per_pack = n // num_packs
    if per_pack == 1:
        return np.tile(np.arange(n), (num_layers, 1)), np.zeros((num_layers, n), dtype=np.int64)
    pack_index = np.full((num_layers, n), -1, dtype=np.int64)
    rank_in_pack = np.full((num_layers, n), -1, dtype=np.int64)
    for i in range(num_layers):
        loads, counts = [0.0] * num_packs, [0] * num_packs
        for item in np.argsort(-weight[i], kind="stable"):
            pack = min((p for p in range(num_packs) if counts[p] < per_pack), key=loads.__getitem__)
            pack_index[i, item], rank_in_pack[i, item] = pack, counts[pack]
            loads[pack] += weight[i, item]
            counts[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(weight, num_phy):
    """Each extra slot goes to the expert whose per-replica load (load / copies) is highest."""
    n, num_log = weight.shape
    phy2log = np.tile(np.arange(num_phy), (n, 1))
    rank = np.zeros((n, num_phy), dtype=np.int64)
    logcnt = np.ones((n, num_log), dtype=np.int64)
    rows = np.arange(n)
    for i in range(num_log, num_phy):
        hot = np.argmax(weight / logcnt, axis=-1)
        phy2log[:, i] = hot
        rank[:, i] = logcnt[rows, hot]
        logcnt[rows, hot] += 1
    return phy2log, rank, logcnt


def inverse(perm):
    inv = np.empty_like(perm)
    np.put_along_axis(inv, perm, np.tile(np.arange(perm.shape[1]), (perm.shape[0], 1)), axis=1)
    return inv


def rebalance_hierarchical(weight, num_phy, num_groups, num_nodes, num_gpus):
    num_layers, num_log = weight.shape
    group_size = num_log // num_groups
    groups_per_node = num_groups // num_nodes
    phy_per_gpu = num_phy // num_gpus
    # step 1: pack expert groups onto nodes
    tokens_per_group = weight.reshape(num_layers, num_groups, group_size).sum(-1)
    gpack, grank = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (((gpack * groups_per_node + grank) * group_size)[:, :, None]
                + np.arange(group_size)).reshape(num_layers, -1)
    mlog2log = inverse(log2mlog)
    # step 2: replicate hot experts inside each node
    tokens_per_mlog = np.take_along_axis(weight, mlog2log, -1).reshape(-1, num_log // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_phy // num_nodes)
    # step 3: pack replicas onto the GPUs of each node
    tokens_per_phy = np.take_along_axis(tokens_per_mlog / mlogcnt, phy2mlog, -1)
    pack, rank_in = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack * phy_per_gpu + rank_in
    pphy2phy = inverse(phy2pphy)
    pphy2mlog = np.take_along_axis(phy2mlog, pphy2phy, -1)
    pphy2mlog = (pphy2mlog.reshape(num_layers, num_nodes, -1)
                 + np.arange(0, num_log, num_log // num_nodes)[None, :, None]).reshape(num_layers, -1)
    pphy2log = np.take_along_axis(mlog2log, pphy2mlog, -1)
    logcnt = np.take_along_axis(mlogcnt.reshape(num_layers, -1), log2mlog, -1)
    return pphy2log, logcnt


def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    weight = np.asarray(weight, dtype=np.float64)
    if num_groups % num_nodes == 0:
        return rebalance_hierarchical(weight, num_replicas, num_groups, num_nodes, num_gpus)
    return rebalance_hierarchical(weight, num_replicas, 1, 1, num_gpus)   # global policy


def gpu_loads(weight_row, phy2log_row, logcnt_row, num_gpus):
    """Each replica serves load / copies; a GPU's load is the sum over its replica slots."""
    per_slot = weight_row[phy2log_row] / logcnt_row[phy2log_row]
    return per_slot.reshape(num_gpus, -1).sum(-1)


print("\n== 5. EPLB README example (2 layers x 12 experts, 16 replicas, 4 groups, 2 nodes x 4 GPUs) ==")
W = np.array([[90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
              [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27]], dtype=np.float64)
README = np.array([[5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],
                   [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1]])
phy2log, logcnt = rebalance_experts(W, 16, 4, 2, 8)
print("phy2log (ours)  :", phy2log.tolist())
print("matches README  :", bool((phy2log == README).all()))

row = W[0]
print(f"layer 0 total load {row.sum():.0f}, ideal per GPU {row.sum():.0f}/8 = {row.sum() / 8:.1f}")
groups = row.reshape(4, 3).sum(-1)
print("group loads g0..g3:", groups.astype(int).tolist())
print("replicated experts (2 copies):", [int(e) for e in np.where(logcnt[0] > 1)[0]])
for policy, (g, nn) in {"hierarchical": (4, 2), "global": (1, 1)}.items():
    p2l, cnt = rebalance_experts(W, 16, g, nn, 8)
    loads = gpu_loads(row, p2l[0], cnt[0], 8)
    slots = p2l[0].reshape(8, 2).tolist()
    print(f"{policy:>12}: experts per GPU {slots}")
    print(f"{'':>12}  loads {loads.tolist()}")
    print(f"{'':>12}  node loads {loads[:4].sum():.0f} | {loads[4:].sum():.0f}; "
          f"max {loads.max():.1f} vs ideal {row.sum() / 8:.1f} -> {loads.max() / (row.sum() / 8):.3f}x")
# no redundancy: 12 experts can't split evenly over 8 GPUs, so compare with the single hottest expert
print(f"without replicas the hottest expert alone (183) is {183 / (row.sum() / 8):.2f}x the ideal GPU load")

# ---------------------------------------------------------------------------
# 6. Bigger skewed load, DeepSeek-sized (loads are ILLUSTRATIVE, seeded)
# ---------------------------------------------------------------------------
print("\n== 6. 256 experts, skewed load (illustrative lognormal, seed 0) ==")
rng = np.random.default_rng(0)
load = rng.lognormal(mean=0.0, sigma=0.8, size=(1, 256)) * 1000
print(f"expert load max/mean = {load.max() / load.mean():.2f}")


def report(name, p2l, cnt, num_gpus, gpus_per_node=8, groups=None):
    loads = gpu_loads(load[0], p2l[0], cnt[0], num_gpus)
    line = f"{name:<44} max/mean GPU load {loads.max() / loads.mean():.3f}"
    if groups:
        size = 256 // groups
        node_of_slot = np.repeat(np.arange(num_gpus) // gpus_per_node, p2l.shape[1] // num_gpus)
        spread = [len(set(node_of_slot[np.isin(p2l[0], np.arange(g * size, (g + 1) * size))]))
                  for g in range(groups)]
        line += f" | nodes per group {min(spread)}-{max(spread)}"
    print(line)


naive32 = np.tile(np.arange(256), (1, 1)), np.ones((1, 256), dtype=np.int64)
report("prefill EP32, no redundancy (8 per GPU)", *naive32, 32, groups=8)
report("prefill EP32 +32 redundant, hierarchical", *rebalance_experts(load, 288, 8, 4, 32), 32, groups=8)
report("prefill EP32 +32 redundant, global", *rebalance_experts(load, 288, 1, 1, 32), 32, groups=8)
report("decode EP144 +32 redundant, global (2/GPU)", *rebalance_experts(load, 288, 1, 1, 144), 144)
report("decode EP320 +64 redundant, global (1/GPU)", *rebalance_experts(load, 320, 1, 1, 320), 320)

# Try this:
# 1. Raise sigma to 1.5 (a hotter hot expert). How many redundant slots bring decode EP144 back under 1.1x?
# 2. Put 2 groups on 2 nodes in the README example (num_groups=2): hierarchical still applies. Does max load move?
# 3. Swap FP8 for BF16 dispatch in section 4 (disp_copy = HIDDEN * BF16): EP's edge over TP at n = 8 disappears.
