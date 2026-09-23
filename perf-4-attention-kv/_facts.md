# Shared facts: Attention & the KV cache (perf-4)

Every card writer reads this file before writing. Reuse these values exactly. If a card needs a new shared number, cite its source on the card.

## This course
The core systems topic: how attention is computed without materialising N×N, and how the KV cache is shrunk, paged, shared, tiered and scheduled. DEEP resources are walked **section by section, never condensed**: online softmax (1 card), FlashAttention (3 cards), FlashAttention-2 (2 cards), PagedAttention/vLLM (3 cards), SGLang RadixAttention (2 cards), Mooncake (2 cards), GQA (1 card), DeepSeek-V2 MLA (1 card). SKIM resources (FA3/FA4/FlashInfer, KIVI, ring attention, NSA) get one card each or share one.
- **Source maps with every number (read first):** ~/Desktop/repos/explainers/tasks/perf-kit/sources/p4-maps.md
- **Full source texts (exact wording, equations, tables; local only, never publish or paste long passages):** same folder: online-softmax.txt, flashattention.txt, fa2.txt, fa3.txt, fa4.txt, flashinfer.txt, pagedattention.txt, sglang.txt, mooncake.txt, gqa.txt, mla.txt, kivi.txt, ring.txt, nsa.txt
- Papers ran on V100/A100/A10G/A800/H100/B200. Show their numbers **as the paper gives them, naming the GPU and model**, then connect to the running example (Llama-3.1-8B on H100). Anything we recompute is labelled "our recomputation".
- Source quirks to state where relevant (see p4-maps.md): FA1 says 1.7× and 1.8× vs Megatron; Mooncake's FAST'25 paper and arXiv report are different papers with different numbers — always say which; PagedAttention's Fig 1 is "13B" on A100-40GB, the 800 KB/token example is OPT-13B; SGLang uses a "frontend hint" for fork (not hint-free); FA4's venue year is ambiguous (footer 2025, URL 2026).
- Courses that exist for links: ../perf-0-map/, ../perf-1-foundations/, ../perf-2-scaling-book-inference/, ../perf-3-kernels/ (cards: execution-model, first-kernel, coalescing, shared-memory-transpose, occupancy, reduction-divergence, reduction-unrolling, matmul-naive, matmul-coalescing-smem, matmul-blocktiling, matmul-warptiling, tensor-cores-low-precision, fusion-cuda-graphs, triton-programming-model, triton-matmul, profiling). Especially relevant: ../perf-2-scaling-book-inference/params-vs-kv.html, kv-tricks.html, attention-intensity.html, inference-engine-design.html, sharding-kv-cache.html; ../perf-3-kernels/reduction-divergence.html (softmax = two reductions), shared-memory-transpose.html, tensor-cores-low-precision.html, triton-programming-model.html (fused softmax). perf-5 (engines) and perf-6 (scaling out) are not built: mention them as plain text, no links.

## The reader
- Setup: Apple M3 (8-core CPU), 24 GB unified memory (~100 GB/s), Ollama 0.34.2 with qwen3.5:4b Q4_K_M (4.7B params, 3.4 GB file). Python 3 with numpy 2.2. **No NVIDIA GPU**: CUDA labs run on Colab T4 [C] or a rented H100 [H].
- Knows almost nothing about GPU/AI performance engineering. Is preparing to join a company that serves LLM inference fast and cheaply (wafer.ai-like). Wants connected, rigorous understanding: enough to follow the full training→serving flow and to talk shop, not PhD depth.
- This is part of an 8-course series: perf-0 map · perf-1 foundations · perf-2 Scaling Book inference · perf-3 kernels · perf-4 attention & KV cache · perf-5 inference engines · perf-6 scaling out · perf-7 training & economics. Link to sibling courses only as listed under "This course" above. The older consumer-side course `../llm-latency/` exists (cards: ttft, output-speed, memory-bandwidth, hardware, quantization, moe, speculative-decoding, prompt-caching, tail-latency, parallel-calls); link to it where relevant.

## The running example (use on every card that can)
**Llama-3.1-8B, BF16, on one H100 SXM.** Shapes: L=32 layers, D=4096, F=14336 (SwiGLU, 3 matrices), N=32 query heads, K=8 KV heads (GQA 4:1), H=128 head dim, V=128256 vocab, untied embeddings.
Computed (script, 2026-09-22):
- Params: attention 41.94M/layer, MLP 176.16M/layer, layer 218.11M; total **8.03B** = MLP 70.2% · attention 16.7% · embeddings+LM head 13.1%
- BF16 weights: **16.06 GB**
- KV cache per token: 2·L·K·H·2 bytes = **131,072 B = 128 KiB**; 8,192-token sequence = **1.07 GB**
- H100 SXM: **989 TFLOPS** dense BF16, **1,979** dense FP8, **3.35 TB/s** HBM3, **80 GB**, 132 SMs, 50 MB L2, 256 KB L1/SMEM per SM → ops:byte **≈ 295**
- Decode ceiling at batch 1: 3.35e12 / 16.06e9 = **≈ 209 tokens/s** (≈ 4.8 ms/step) — upper bound, real engines reach roughly 70–85% of this (label any specific % as illustrative unless sourced)
- Free HBM after weights: 80 − 16.06 = **63.9 GB** → ≈ 488k tokens of KV → ≈ **59 sequences of 8k tokens**
- Prefill of 1,000 tokens at 100% MFU: 2·8.03e9·1000 / 989e12 = **16.2 ms** (a floor)

**Second example (MoE, scale-out):** DeepSeek-V3: 671B total, 37B active, 61 layers, MLA, 256 routed experts + 1 shared, 8 routed active per token, FP8 training, MTP. (Source: DeepSeek-V3 tech report arXiv 2412.19437.)

## KV numbers for the running example (from perf-1/perf-2; reuse exactly)
- Llama-3.1-8B KV per token: 2·L·K·H·2 B = 2·32·8·128·2 = **131,072 B = 128 KiB** (GQA 4:1). MHA version (K=32): 512 KiB; MQA (K=1): 16 KiB.
- 8,192-token sequence: **1.07 GB**. H100 free HBM after BF16 weights: **63.9 GB** → ≈ 488k tokens → **59 sequences of 8k** (14 with MHA, 476 with MQA; 238 at 2k, 14 at 32k).
- FP8 KV halves it: 64 KiB/token → 119 seqs of 8k (perf-2 kv-tricks).
- Decode step at 8k with B = 59: KV read 18.9 ms + weights 4.79 ms = 23.7 ms → 2,489 tok/s; KV is 80% of the step (perf-2).
- Attention intensity in decode with GQA-4 ≈ 4 FLOPs/byte, far below H100's 295 → always memory-bound (perf-2 attention-intensity).
- DeepSeek-V3 (second example): 61 layers, MLA caches only c^KV (d_c = 512) and k^R (d_h^R = 64) → (512 + 64) × 61 = 35,136 elements/token; at 2 bytes (BF16, our assumption — V3 states no KV dtype) = 70,272 B ≈ 68.6 KiB/token (our recomputation). V2: 60 layers, same dims.

## Colour meanings (same on every card)
- green `--accent`: useful compute, the KV cache that is actually used/hit, the winner
- grey `--faint` / `--surface2`: waiting on memory, overhead, **wasted/reserved/fragmented KV memory**, free memory, evicted
- neutral `--text`/`--muted`: model weights and data at rest (Q, K, V, O tiles in HBM)
- red `--red`: ✗, limits (out of memory), the bottleneck, the N×N matrix that must not be materialised, the worst value
- `--amber` only for a third roofline region

## Terms (use these names; mention aliases once in the subtitle)
softmax, safe softmax, online softmax (running max m, running sum ℓ/d); logsumexp; tiling; SRAM (on-chip shared memory) vs HBM; IO complexity / IO-awareness; recomputation; FlashAttention (FA1/2/3/4); split-K vs split-Q (FA2 warp partitioning); warp specialization, pingpong; MHA/GQA/MQA/MLA; latent vector c^KV; KV block / page, block table, logical vs physical block; internal/external fragmentation, reservation; copy-on-write, reference count; preemption: swap vs recompute; prefix caching; radix tree; LRU; cache hit rate; cache-aware scheduling; KV tiers (HBM → CPU DRAM → SSD → remote); disaggregated KV store; KV quantization (FP8 KV, KIVI); sliding-window attention; ring attention; sparse attention (NSA).

## Labs (three tiers)
- **[L] local, runs on the Mac:** `labs/<slug>.py`, stdlib + numpy, CPU, < 30 s, prints the numbers the card uses, ends with 1–3 commented "try this" lines. Numpy reference implementations and simulators (online softmax, tiled attention with exact-match check and byte counting, allocators, radix trees, schedulers, calculators). Link text "Run it: labs/<slug>.py" → `https://github.com/seyonv/explainers/blob/main/perf-4-attention-kv/labs/<slug>.py`. Every number the lab computes must match the card.
- **[C] Colab T4 notebook:** only where a real GPU measurement matters. `labs/<slug>.ipynb`, nbformat 4, markdown intro with "Runtime → Change runtime type → T4 GPU", `!nvidia-smi`, PyTorch (preinstalled). Note: T4 is Turing (sm_75): torch SDPA's flash backend needs sm_80+, so on T4 compare the math backend vs the memory-efficient backend. Link twice: "Open in Colab: labs/<slug>.ipynb" → `https://colab.research.google.com/github/seyonv/explainers/blob/main/perf-4-attention-kv/labs/<slug>.ipynb`, and "view on GitHub". **Nobody here can run it**: the card says "Not yet run by the author" and never invents a T4 result.
- **[H] rented H100:** `labs/h100-vllm-kv.sh` only (owned by paged-results). Header states runtime and cost at $2–3.50/hr (budget cap $300–400).
- Use your own scratch subfolder named after your card for temp files; other writers run in parallel.

## Card list (course folder: perf-4-attention-kv)
| File | Title | Covers | Lab |
|---|---|---|---|
| _overview.html | KV memory is the currency of serving | written last by the main agent | – |
| online-softmax.html | Online softmax | Milakov & Gimelshein, all sections: Alg 1–4, Theorem 1, parallel ⊕, benchmarks | [L] online-softmax.py |
| flashattention-tiling.html | FlashAttention I: tiling and recomputation | FA §1, §2 (hardware, standard attention Alg 0), §3.1 (tiling, recomputation, fusion, Alg 1, Thm 1) | [L] flash-tiling.py (numpy tiled vs naive, exact match, step trace) |
| flashattention-io.html | FlashAttention II: the IO-complexity argument | FA §3.2 (Thm 2, Prop 3, Fig 2, block size), §3.3 block-sparse (Prop 4) | [L] flash-io.py (HBM bytes: standard vs tiled vs block size vs M) |
| flashattention-backward-results.html | FlashAttention III: backward pass and results | FA App B (memory-efficient backward, Alg 4, Thm 5, vs Rabe & Staats), §4 experiments, E tables, §5 limitations | [C] attention-t4.ipynb (naive vs SDPA math vs mem-efficient: time and peak memory vs N) |
| flashattention-2-algorithm.html | FlashAttention-2 I: fewer non-matmul FLOPs | FA2 §1–2, §3.1 (un-scaled O, logsumexp, Alg 1, causal masking, backward Alg 2, MQA/GQA) | [L] fa2-nonmatmul.py (count non-matmul ops FA1 vs FA2; cost at 16×) |
| flashattention-2-parallelism.html | FlashAttention-2 II: parallelism and work partitioning | FA2 §3.2, §3.3 (split-K vs split-Q), §4 benchmarks + Table 1, §5 | [L] fa2-occupancy.py (thread blocks vs 108/132 SMs for batch×heads vs +seq blocks) |
| flashattention-3-4-flashinfer.html | FlashAttention-3, -4 and FlashInfer | FA3 (warp specialization, pingpong, FP8), FA4 (Blackwell asymmetry, exp emulation, TMEM), FlashInfer (paged/ragged, cascade, JIT, scheduler) | – |
| gqa.html | Grouped-query attention | GQA paper, all sections: uptraining, groups, tables | [L] gqa.py (KV bytes and decode step vs G for Llama-3.1-8B; mean-pool uptraining demo) |
| mla.html | Multi-head latent attention (DeepSeek) | DeepSeek-V2 MLA section: compression, absorption, decoupled RoPE, KV table | [L] mla.py (numpy: absorbed vs explicit attention equal; KV bytes MHA/GQA/MQA/MLA) |
| kv-quantization.html | Quantizing the KV cache | FP8 KV (perf-2 kv-tricks) + KIVI (per-channel K, per-token V, residual window) | [L] kv-quant.py (per-token vs per-channel error on synthetic K with outlier channels) |
| paged-memory.html | PagedAttention I: KV memory as pages | PA §1–3, §4.1–4.3: waste (20.4–38.2%), blocks, block tables, Fig 6 walk-through | [L] paged-allocator.py (contiguous max-length vs paged: waste %, batch fit) |
| paged-sharing.html | PagedAttention II: sharing, copy-on-write and preemption | PA §4.4–4.6: parallel sampling, beam search, shared prefix, ref counts, COW, FCFS, swap vs recompute, distributed | [L] paged-cow.py (ref-count + COW simulator for n samples / beam) |
| paged-results.html | PagedAttention III: implementation and results | PA §5–8: kernels, fork/append/free, Table 1, evaluation, ablations (block size, recompute vs swap), discussion | [H] h100-vllm-kv.sh (vLLM Llama-3.1-8B: KV blocks at startup, max concurrency vs gpu_memory_utilization; prefix-caching TTFT on/off) |
| radixattention.html | RadixAttention I: the radix tree of KV caches | SGLang §3 core: radix tree, LRU leaves, ref counts, shared pool, the 9-step Fig 3, frontend hint, overhead | [L] radix-tree.py (radix tree + LRU replaying a Fig-3-like trace) |
| cache-aware-scheduling.html | RadixAttention II: cache-aware scheduling | SGLang Theorem 3.1 (DFS / longest-shared-prefix-first), Alg 1, starvation, data-parallel meta-tree router, eval (6.4×, hit rates, Chatbot Arena), compressed FSM in one line | [L] radix-schedule.py (FCFS vs LSPF hit rate on synthetic few-shot/multi-turn batches) |
| mooncake-store.html | Mooncake I: trading storage for computation | FAST'25 §1–3: KVCache-centric architecture, store-vs-recompute inequality (Eq 1–2), Mooncake Store, Transfer Engine, capacity analysis (3M vs 50M tokens) | [L] store-vs-recompute.py (min bandwidth for reuse to beat recompute: their LLaMA3-70B A800/H800 + Llama-3.1-8B H100) |
| mooncake-scheduling.html | Mooncake II: KV-centric scheduling | FAST'25 §3.3–5 + arXiv §7: Conductor, cache load balancing, chunked pipeline parallelism, early rejection, results | – |
| long-context.html | Long context: sliding windows, ring and sparse attention | Mistral sliding window + rolling buffer, Ring Attention (overlap condition), NSA (3 branches) | [L] long-context.py (KV bytes vs context for full / window / NSA-like; ring overlap condition) |

Siblings: all files above (relative hrefs).
