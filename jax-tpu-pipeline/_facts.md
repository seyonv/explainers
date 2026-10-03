# Shared facts: pipeline parallelism across TPU hosts in JAX

Every card writer reads this file before writing. Reuse these values exactly. If a card needs a new shared number, add it in your return notes with its source.

## The reader
- Setup: a complete beginner to distributed training. They read a 2021-era JAX GitHub post by Sholto Douglas asking how to do pipeline parallelism across multiple TPU hosts using ICI, with send/recv like NCCL. They want to understand every concept in it.
- Assume no knowledge of GPUs/TPUs beyond "AI chip". Define every term the first time.

## The post (the source every card ties back to)
Paraphrase: "With torch/NVIDIA, NCCL gives multi-node GPU send/recv. Is there a JAX equivalent that uses ICI links between TPU hosts on a v3-32 or bigger, with send/recv in addition to collective mean/sum? mpi4jax has send/recv but no TPU support. Ray works (mesh-transformer-jax) but copies through CPU. xmap / global device array idea: mesh [time_shard, layer_shard, data_shard], put different layers on different time_shards, loop over micro-batches shifting data so n_time_shards micro-batches run at once. Works within one 8-device host (EleutherAI's 20B found PP=4, MP=2 best within a node). How do I do it across hosts over ICI?"

## The running example (use on every card that can)
- Hardware: a TPU **v3-32** slice = **32 TPU cores = 16 chips (2 cores per v3 chip) = 4 hosts × 8 cores** (each host is a CPU machine with 4 v3 chips).
- Model: a **24-layer transformer** (illustrative size).
- Mesh: **[pipeline (Sholto's "time_shard") = 4, model/tensor (his "layer_shard") = 2, data (his "data_shard") = 4]** → 4 × 2 × 4 = 32 cores.
- Pipeline stage s holds layers 6s+1 … 6s+6 (24 layers / 4 stages = 6 layers each).
- Micro-batches: compare m = 1, 4, 8, 16 with p = 4 stages.
- Pipeline bubble fraction (GPipe, Huang et al. 2019, arXiv:1811.06965): bubble = (p − 1) / (m + p − 1). For p=4: m=1 → 75%, m=4 → 42.9%, m=8 → 27.3%, m=16 → 15.8%.
- EleutherAI GPT-NeoX-20B: trained with tensor parallel 2 and pipeline parallel 4 (GPT-NeoX-20B paper, arXiv:2204.06745) — cite the paper; writer to confirm the exact wording.

## Sourced numbers
Writers: research and cite primary sources (Google Cloud TPU docs, JAX docs, papers). Write "not published" rather than guess. Any figure you cannot source must be labelled "illustrative" on the card.

| Fact | Value | Source (URL) | Measured / as of |
|---|---|---|---|
| v3 chip = 2 TensorCores; v3-8 = 1 host, 4 chips | | Google Cloud TPU docs (system architecture / v3 page) | writer to confirm |
| Bubble formula | (p−1)/(m+p−1) | GPipe arXiv:1811.06965 | 2019 |
| xmap status | experimental, later removed in favour of shard_map | JAX docs / changelog | writer to confirm |

## Local measurements
None — the reader has no TPU. Do not run benchmarks. Small `python3` arithmetic scripts only.

## Terms (name used on cards / aliases)
- **Host** (alias: node, VM, worker) — the CPU machine TPU chips are plugged into.
- **TPU core / chip** — v3 chip has 2 cores; JAX counts each core as one device on v3.
- **Pod slice** (alias: v3-32, "a v32") — the number after the dash counts cores for v2/v3.
- **ICI** — inter-chip interconnect, direct chip-to-chip links inside a pod, crossing hosts.
- **DCN** — data-center network, the regular network between hosts/pods (via CPU NICs).
- **Data parallelism**, **tensor parallelism** (alias: model parallelism, MP, Sholto's layer_shard), **pipeline parallelism** (alias: PP, Sholto's time_shard).
- **Micro-batch**, **bubble**.
- **Collective** (all-reduce = sum/mean everywhere), **point-to-point** (send/recv).
- **NCCL**, **MPI**, **mpi4jax**, **Ray**.
- **SPMD**, **mesh**, **pmap**, **xmap**, **jax.Array** (the "global device array"), **shard_map**, **ppermute** (XLA "collective permute").

## Colour meanings
- green `--accent`: the fast path / the answer / useful work (ICI, ppermute, busy pipeline stages)
- grey `--faint` / `--surface2`: idle or overhead (bubble time, CPU hops, waiting)
- red `--red`: ✗, the slow or unsupported option (CPU copy path, missing TPU support)

## Card list
| File | Title | Group | Owner |
|---|---|---|---|
| _overview.html | The post, decoded | overview | main agent (last) |
| sholto-bradbury.html | Why a GitHub question became a job at Google | overview / story | main agent |
| three-parallelisms.html | Data, tensor and pipeline parallelism | 1 basics | subagent |
| pipeline-bubble.html | Micro-batches and the pipeline bubble | 1 basics | subagent |
| collectives-vs-send-recv.html | Collectives vs send/recv (NCCL, MPI, mpi4jax, Ray) | 2 communication | subagent |
| tpu-pod-ici.html | TPU hosts, pod slices and ICI | 2 communication | subagent |
| spmd-ppermute.html | SPMD, meshes and ppermute: the JAX answer | 3 the answer | subagent |
