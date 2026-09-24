"""How GPUs talk: collectives and interconnects (course 6, card 1).

Run: python3 labs/collectives.py   (numpy only, CPU, under a second)

Nothing here is measured. It is a simulator plus a cost model:
  (1) The six NCCL collectives on 4 simulated GPUs, each checked against its definition,
      and the identities ReduceScatter + AllGather = AllReduce and Reduce + Broadcast = AllReduce.
  (2) Ring all-reduce, step by step, on 4 GPUs (the table on the card) and on 8 GPUs with
      random data: result checked against the plain sum, bytes counted on every link.
  (3) algbw vs busbw (nccl-tests PERFORMANCE.md factors).
  (4) Cost model T = 2(n-1)/n * S/W + 2(n-1) * alpha for NVLink (450 GB/s per direction) and
      InfiniBand (one 400 Gb/s NIC = 50 GB/s per GPU). alpha = 1 us per hop is ILLUSTRATIVE
      (the Scaling Book's figure for a TPU ICI hop; NVIDIA publishes no NVLink hop latency).
  (5) The measured reality (Scaling Book, "How to Think About GPUs") vs the model.
  (6) Worked example: Llama-3.1-70B TP8 decode, 160 all-reduces per step, B = 1 / 64 / 256.
  (7) The units trap and two naive alternatives to the ring.
"""
import numpy as np

rng = np.random.default_rng(0)
GB = 1e9
US = 1e-6

# ---------------------------------------------------------------- (1) the six collectives
print("(1) The six NCCL collectives on n = 4 GPUs (each GPU holds 4 chunks of N values)")
n, N = 4, 3
x = rng.standard_normal((n, n, N))          # x[gpu, chunk, values]; rank k's input


def allreduce(x):
    s = x.sum(axis=0)
    return np.stack([s] * len(x))


def reduce(x, root=0):
    out = np.zeros_like(x)                  # only the root's receive buffer is written
    out[root] = x.sum(axis=0)
    return out


def broadcast(x, root=0):
    return np.stack([x[root]] * len(x))


def reducescatter(x):                       # rank k gets the sum of chunk k
    s = x.sum(axis=0)
    return np.stack([s[k] for k in range(len(x))])


def allgather(y):                           # y[k] = rank k's one chunk -> everyone gets all, in rank order
    return np.stack([y] * len(y))


def alltoall(x):                            # chunk j of rank i goes to rank j, lands in slot i
    return x.transpose(1, 0, 2).copy()


s = x.sum(axis=0)
checks = {
    "AllReduce     every rank gets the sum":            np.allclose(allreduce(x), s),
    "Reduce        only the root gets the sum":         np.allclose(reduce(x)[0], s) and not reduce(x)[1:].any(),
    "Broadcast     every rank gets the root's buffer":  np.allclose(broadcast(x), x[0]),
    "ReduceScatter rank k gets chunk k of the sum":     all(np.allclose(reducescatter(x)[k], s[k]) for k in range(n)),
    "AllGather     every rank gets all ranks' chunks":  np.allclose(allgather(x[:, 0])[2], x[:, 0]),
    "AlltoAll      out[j][i] = in[i][j] (a transpose)": all(np.allclose(alltoall(x)[j, i], x[i, j]) for i in range(n) for j in range(n)),
    "ReduceScatter + AllGather == AllReduce":           np.allclose(allgather(reducescatter(x)), allreduce(x)),
    "Reduce + Broadcast        == AllReduce":           np.allclose(broadcast(reduce(x)), allreduce(x)),
}
for k, v in checks.items():
    print(f"  {k:52s} {'OK' if v else 'FAIL'}")
assert all(checks.values())

# ---------------------------------------------------------------- (2) ring all-reduce
print("\n(2) Ring all-reduce. Step s of phase 1 (reduce-scatter): GPU i sends chunk (i - s) mod n to GPU i+1,")
print("    which adds it to its own copy. Phase 2 (all-gather): GPU i forwards the finished chunk (i + 1 - s) mod n.")


def ring_allreduce(data, trace=False):
    """data: (n, total) array, one row per GPU. Returns (result, bytes sent per GPU, per-step states)."""
    n = data.shape[0]
    chunks = [np.array_split(row.copy(), n) for row in data]
    who = [[{i} for _ in range(n)] for i in range(n)]    # which GPUs' inputs each chunk contains
    sent = np.zeros(n)                                    # bytes each GPU put on its outgoing link
    states = [("start", [[set(c) for c in w] for w in who], [None] * n)]
    for phase in ("reduce-scatter", "all-gather"):
        for step in range(n - 1):
            idx = [((i - step) if phase == "reduce-scatter" else (i + 1 - step)) % n for i in range(n)]
            msgs = [(chunks[i][idx[i]].copy(), set(who[i][idx[i]])) for i in range(n)]  # all send at once
            for i in range(n):
                dst, c = (i + 1) % n, idx[i]
                buf, members = msgs[i]
                sent[i] += buf.nbytes
                if phase == "reduce-scatter":
                    chunks[dst][c] = chunks[dst][c] + buf
                    who[dst][c] = who[dst][c] | members
                else:
                    chunks[dst][c] = buf
                    who[dst][c] = set(members)
            states.append((f"{phase} {step + 1}", [[set(c) for c in w] for w in who], idx))
    return np.stack([np.concatenate(c) for c in chunks]), sent, states


letters = "abcdefgh"
_, _, states = ring_allreduce(rng.standard_normal((4, 4 * 16)))


def label(members, n):
    return "SUM" if len(members) == n else "".join(letters[m] for m in sorted(members))


print("    n = 4. Each cell lists which GPUs' data a chunk holds (a = GPU0 ... d = GPU3); [x] = chunk sent this step")
for name, st, idx in states:
    row = []
    for g in range(4):
        cells = [label(st[g][c], 4) for c in range(4)]
        if idx[g] is not None:
            cells[idx[g]] = "[" + cells[idx[g]] + "]"
        row.append(" ".join(f"{c:>5s}" for c in cells))
    print(f"    {name:17s} | " + " | ".join(row))

n8 = 8
S_elems = 8 * 1024 * 1024                              # 8M float32 = 32 MiB per GPU
data = rng.standard_normal((n8, S_elems)).astype(np.float32)
res, sent, _ = ring_allreduce(data)
S = data[0].nbytes
ok = np.allclose(res, np.broadcast_to(data.astype(np.float64).sum(0), res.shape), atol=1e-3)
print(f"\n    n = 8, S = {S / 2**20:.0f} MiB per GPU: every GPU ends with the exact sum: {'OK' if ok else 'FAIL'}")
print(f"    steps = 2(n-1) = {2 * (n8 - 1)}, each sends S/n = {S / n8 / 2**20:.0f} MiB per GPU")
print(f"    bytes sent per GPU = {sent[0] / 2**20:.0f} MiB = {sent[0] / S:.4f} x S   (formula 2(n-1)/n = {2 * (n8 - 1) / n8:.4f})")
print(f"    all 8 links busy every step; total bytes on the wire = {sent.sum() / S:.1f} x S")
for nn in (2, 4, 8, 16, 64):
    print(f"    n = {nn:3d}: 2(n-1)/n = {2 * (nn - 1) / nn:.4f}  -> approaches 2 as n grows")

# ---------------------------------------------------------------- (3) algbw vs busbw
print("\n(3) algbw vs busbw (nccl-tests): algbw = S / t, busbw = algbw x factor")
n = 8
factors = {"AllReduce": 2 * (n - 1) / n, "ReduceScatter": (n - 1) / n, "AllGather": (n - 1) / n,
           "AlltoAll": (n - 1) / n, "Broadcast": 1.0, "Reduce": 1.0}
for k, f in factors.items():
    print(f"    {k:13s} factor at n = 8: {f:.3f}")
W_NV = 450 * GB
S1 = 1 * GB
t = 2 * (n - 1) / n * S1 / W_NV
print(f"    1 GB AllReduce, 8 GPUs, links at 450 GB/s: t = 1.75 x 1 GB / 450 GB/s = {t * 1e3:.2f} ms")
print(f"    algbw = 1 GB / t = {S1 / t / GB:.0f} GB/s ; busbw = algbw x 1.75 = {S1 / t * 1.75 / GB:.0f} GB/s (= the link)")

# ---------------------------------------------------------------- (4) cost model
print("\n(4) Cost model, ring all-reduce on n = 8: T = 2(n-1)/n * S/W + 2(n-1) * alpha")
ALPHA = 1 * US                                          # ILLUSTRATIVE per-hop latency
W_IB = 400e9 / 8                                        # one 400 Gb/s ConnectX-7 per GPU = 50 GB/s
links = {"NVLink (450 GB/s)": W_NV, "InfiniBand (50 GB/s)": W_IB}


def t_ar(S, W, n=8, alpha=ALPHA):
    bw = 2 * (n - 1) / n * S / W
    lat = 2 * (n - 1) * alpha
    return bw + lat, bw, lat


print(f"    alpha = {ALPHA / US:.0f} us per hop (illustrative) -> latency floor 2(n-1) alpha = {14 * ALPHA / US:.0f} us per all-reduce")
for name, W in links.items():
    xover = 8 * ALPHA * W                                # 2(n-1)/n S/W = 2(n-1) alpha  ->  S = n alpha W
    print(f"    {name:22s} crossover S* = n * alpha * W = {xover / 1e6:.2f} MB  (below: latency-bound)")
print("    model busbw (= 1.75 S / T) vs message size:")
sizes = [1e3, 16384, 1e5, 1048576, 4194304, 1e7, 58720256, 1e8, 1e9, 1e10]
for S_ in sizes:
    row = []
    for name, W in links.items():
        T, bw, lat = t_ar(S_, W)
        row.append(f"{name.split()[0]:10s} T = {T / US:10.1f} us, busbw = {1.75 * S_ / T / GB:6.1f} GB/s")
    print(f"    S = {S_ / 1e6:10.4f} MB | " + " | ".join(row))

# ---------------------------------------------------------------- (5) measured reality
print("\n(5) Measured (Scaling Book, 8xH100, SHARP off) vs this model")
S58 = 8192 * 3584 * 2
for label_, S_, busbw in (("LLaMA-3-70B MLP shard bf16[8192,3584]", S58, 150 * GB),
                          ("~10 GB per device", 10e9, 370 * GB)):
    t_meas = 1.75 * S_ / busbw
    T, _, _ = t_ar(S_, W_NV)
    print(f"    {label_:38s} S = {S_ / 1e6:8.1f} MB: measured busbw ~{busbw / GB:.0f} GB/s -> t ~ {t_meas / US:8.0f} us;"
          f" model {T / US:8.0f} us ({t_meas / T:.1f}x slower than model)")
print(f"    peak measured / spec = 370 / 450 = {370 / 450:.0%}; at 58 MB 150 / 450 = {150 / 450:.0%}")

# ---------------------------------------------------------------- (6) worked example: 70B TP8 decode
print("\n(6) Llama-3.1-70B TP8 decode: 160 all-reduces per step (2 per layer x 80, our inference from Megatron)")
D, L, AR_PER_STEP, STEP_WEIGHTS_MS = 8192, 80, 160, 5.27
for B in (1, 64, 256):
    msg = B * D * 2
    parts = [f"B = {B:3d}: msg = B*D*2 = {msg:>9,d} B"]
    for name, W in links.items():
        T, bw, lat = t_ar(msg, W)
        parts.append(f"{name.split()[0]}: bw {bw / US:7.2f} us x160 = {bw * 160 * 1e3:6.2f} ms,"
                     f" +lat {lat * 160 * 1e3:.2f} ms = {T * 160 * 1e3:6.2f} ms/step")
    print("    " + " | ".join(parts))
print(f"    compare: memory-only decode step at TP8 = {STEP_WEIGHTS_MS} ms (course facts)")
lat_step = 160 * 14 * ALPHA
print(f"    latency floor alone: 160 x 14 us = {lat_step * 1e3:.2f} ms = {lat_step * 1e3 / STEP_WEIGHTS_MS:.0%} of the 5.27 ms step (illustrative alpha)")

# ---------------------------------------------------------------- (7) units trap + naive alternatives
print("\n(7) Units trap")
print(f"    '900 GB/s' vs '400 Gb/s' read naively: 900 / 400 = {900 / 400:.2f}x")
print(f"    bytes fixed, directions mixed: 900 GB/s / 50 GB/s = {900 / 50:.0f}x")
print(f"    apples to apples (per direction, bytes): 450 / 50 = {450 / 50:.0f}x")
print(f"    H800 (DeepSeek-V3 report): 160 / 50 = {160 / 50:.1f}x")
print(f"    V100 DGX-2H (Megatron): 300 GB/s NVSwitch vs 100 GB/s per server / 8 GPUs = {100 / 8:.1f} GB/s per GPU -> {300 / 12.5:.0f}x (our division)")
print("    Naive alternatives on 8 GPUs, bandwidth term only, in units of S/W:")
print(f"      ring:                              2(n-1)/n = {2 * 7 / 8:.2f}")
print(f"      star (all send to GPU0, it sends back): root link carries 7S each way >= {7:.2f}  ({7 / 1.75:.0f}x the ring)")
print(f"      everyone sends its whole array to all 7 others: {7:.2f}  ({7 / 1.75:.0f}x the ring)")

# Try this:
# 1. Set ALPHA = 5 * US: the NVLink crossover moves to 18 MB, and the latency floor becomes 11.2 ms per 70B step.
# 2. Change n8 to 16 or 64: bytes per GPU go to 1.875 S and 1.97 S, never past 2 S. The ring scales in bandwidth.
# 3. In (6), add a TP16 row across two nodes (n = 16, W = W_IB): the slowest link sets the ring's speed.
