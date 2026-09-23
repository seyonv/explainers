# Shared facts: GPU programming & kernels (perf-3)

Every card writer reads this file before writing. Reuse these values exactly. If a card needs a new shared number, cite its source on the card.

## This course
Your first kernel and your first tuning. It walks the DEEP resources **section by section, never condensed**: Harris's reduction slides (2 cards, kernel by kernel), Harris's transpose blog (the coalescing and shared-memory cards), siboehm's matmul worklog (4 cards, kernel by kernel), the Triton tutorials 01–03 (2 cards) and the CUDA programming-model pages. Each card's footer names the exact section it covers.
- **Source maps with every number (read first):** ~/Desktop/repos/explainers/tasks/perf-kit/sources/p3-maps.md
- **Full source texts (for exact wording and code; local only, never publish or paste long passages):** the same folder: siboehm-mmm.txt, reduction.txt, transpose.txt, coalescing.txt, shared-memory.txt, triton-tutorials.txt, cuda-guide-intro.txt, even-easier-intro.txt, volkov.txt, lowprec.txt, fusion.txt, profiling.txt
- The sources ran on older GPUs (G80, M2050/K20c, A6000, GTX480). Show their numbers **as the source gives them, naming the GPU**, then connect to the reader's two GPUs: the **Colab T4** (labs) and the **H100** (running example). Anything we recompute is labelled "our recomputation".
- Source quirks to state where relevant (see p3-maps.md): siboehm 268 vs 278 MB, 30 vs 38.7 TFLOPs, text vs table rounding, "A100" in the kernel 10 text; the reduction slides' warp-synchronous `volatile` trick is unsafe on Volta and later GPUs (independent thread scheduling). Use `__syncwarp`/`__shfl_down_sync` today.

## The reader
- Setup: Apple M3 (8-core CPU), 24 GB unified memory (~100 GB/s), Ollama 0.34.2 with qwen3.5:4b Q4_K_M (4.7B params, 3.4 GB file). Python 3 with numpy 2.2. **No NVIDIA GPU**: CUDA labs run on Colab T4 [C] or a rented H100 [H].
- Knows almost nothing about GPU/AI performance engineering. Is preparing to join a company that serves LLM inference fast and cheaply (wafer.ai-like). Wants connected, rigorous understanding: enough to follow the full training→serving flow and to talk shop, not PhD depth.
- This is part of an 8-course series: perf-0 map · perf-1 foundations · perf-2 Scaling Book inference · perf-3 kernels · perf-4 attention & KV cache · perf-5 inference engines · perf-6 scaling out · perf-7 training & economics. Link to sibling courses as `../perf-N-slug/` only for courses that exist (perf-0-map, perf-1-foundations, perf-2-scaling-book-inference). The older consumer-side course `../llm-latency/` exists (cards: ttft, output-speed, memory-bandwidth, hardware, quantization, moe, speculative-decoding, prompt-caching, tail-latency, parallel-calls); link to it where relevant.

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

## Local measurements (Apple M3, Ollama 0.34.2, qwen3.5:4b Q4_K_M, think=false, temperature 0, 128 generated tokens, warm)
Command: `python3 measure.py qwen3.5:4b` (POST /api/generate, reads prompt_eval_count/duration, eval_count/duration), 2026-09-22.
| Prompt tokens | Prefill time | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| 27 | 0.29 s | 92 | 15.25 |
| 76 | 0.55 s | 137 | 15.14 |
| 244 | 1.30 s | 188 | 14.12 |
- Effective bandwidth implied by decode: 15 tok/s × ~3.4 GB ≈ **51 GB/s** (≈ half of the M3's ~100 GB/s peak). Mark the 3.4 GB-read-per-token assumption as approximate (the file includes a vision encoder not read per token).
- Prefill per token is ~12× faster than decode per token at 244 prompt tokens: prefill is parallel over tokens, decode is serial.

## Sourced numbers
| Fact | Value | Source | As of |
|---|---|---|---|
| H100 SXM specs | 989 TF BF16 dense, 1979 TF FP8 dense, 3.35 TB/s, 80 GB | NVIDIA H100 datasheet; Scaling Book Part 12 (GPUs) | 2025 |
| H200 / B200 | 141 GB 4.8 TB/s / 192 GB 8 TB/s, 2250 TF BF16 | Scaling Book Part 12 | 2025 |
| TPU v5e | 1.97e14 BF16 FLOP/s, 3.94e14 int8, 8.2e11 B/s HBM, 16 GB HBM, ICI 4.5e10 B/s one-way | Scaling Book Parts 1, 7 | 2025 |
| Critical batch size | v5e 240; int8 weights+bf16 compute 120; H100 ≈ 295 (book also says "about 280") ; B200 281 | Scaling Book Parts 1, 7, 12 | 2025 |
| M3 memory bandwidth | ~100 GB/s | Apple M3 spec | — |
| Kiely ops:byte for H100 FP16 | 989/3.35 ≈ 295 | Inference Engineering (Kiely, 2026) p.62 | 2026 |

## The lab GPU: NVIDIA T4 on free Colab (Turing, sm_75)
40 SMs, 2,560 CUDA cores, 320 tensor cores (2nd gen), 16 GB GDDR6, **320 GB/s** (Turing whitepaper Table 5; the T4 datasheet prints 300 GB/s — use 320, footnote 300 where it matters), boost 1,590 MHz, 70 W, **8.1 TFLOPS FP32**, **65 TFLOPS FP16 tensor** (mixed precision), 1,024 threads per SM max, 64 KB shared memory per SM max. No BF16 or FP8 tensor cores (Turing supports FP16/INT8/INT4 tensor math). Ops:byte FP32 = 8.1e12 / 320e9 ≈ **25**; FP16 tensor ≈ **203**. Sources: T4 datasheet https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf, Turing whitepaper https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf, CC table https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (CC 7.5 tensor inputs FP16/INT8/INT4 only) and Harris's "Even Easier Intro" (40 SMs, 320 GB/s). Colab's T4 availability and clocks vary; say so wherever a T4 number is shown.

## Colour meanings (same on every card)
- green `--accent`: compute / useful work, the useful bytes of a memory transaction, the winner or faster option
- grey `--faint` / `--surface2`: waiting on memory, overhead, idle threads, wasted bytes of a transaction, launch gaps
- neutral `--text`/`--muted`: data at rest (matrices, tiles, arrays) in memory diagrams
- red `--red`: ✗, limits, the bottleneck, bank conflicts, divergent/serialized work, the worst value
- `--amber` only for a third roofline region

## Terms (use these names; mention aliases once in the subtitle)
kernel; thread, warp (32 threads), thread block (CTA), grid, cluster; SM (streaming multiprocessor); SIMT; warp divergence; global memory (HBM/GDDR); shared memory (SMEM); registers; L1/L2; memory coalescing; memory transaction (32 B sector / 128 B); bank conflict (32 banks × 4 B); occupancy; latency hiding; ILP; tiling / blocking; arithmetic intensity; roofline; GFLOP/s, GB/s effective bandwidth; tensor core, MMA; FP8 E4M3/E5M2; MX (MXFP8/MXFP4), NVFP4; kernel fusion; launch overhead; CUDA graph; Triton program, BLOCK_SIZE, mask, autotune; Nsight Systems (nsys), Nsight Compute (ncu); Compute Sanitizer; TMA, warp specialization (Hopper vocabulary, overview only).

## Labs (three tiers)
- **[L] local, runs on the Mac:** `labs/<slug>.py`, stdlib + numpy, CPU, < 30 s, prints the numbers the card uses, ends with 1–3 commented "try this" lines. These are *simulators and calculators* (for example: count 32-byte transactions for an access pattern, count bank conflicts for a tile layout, count loads per result for a tiling, compute occupancy). Link text "Run it: labs/<slug>.py" → `https://github.com/seyonv/explainers/blob/main/perf-3-kernels/labs/<slug>.py`. Every number the lab computes must match the card.
- **[C] Colab T4 notebook:** `labs/<slug>.ipynb`, valid nbformat 4 JSON. First a markdown cell: title, "Runtime → Change runtime type → T4 GPU", and what you will measure. Then `!nvidia-smi`. CUDA code goes in `%%writefile name.cu` cells, built with `!nvcc -O3 -arch=sm_75 -o name name.cu && ./name`. Time with `cudaEvent` after warm-up runs, check every CUDA call, and check the result against a CPU or cuBLAS/torch reference. PyTorch and Triton are preinstalled on Colab. Print GB/s or GFLOP/s next to the T4 peak (320 GB/s, 8.1 TFLOPS FP32, 65 FP16 tensor). End with a markdown cell holding the source's reference numbers (naming the GPU) and "try this" ideas. Link it on the card twice: "Open in Colab: labs/<slug>.ipynb" → `https://colab.research.google.com/github/seyonv/explainers/blob/main/perf-3-kernels/labs/<slug>.ipynb`, and "view on GitHub" → the github.com blob URL.
  - **Nobody on this project can run [C] notebooks** (no NVIDIA GPU). The card must say plainly: "Not yet run by the author. Numbers on this card are from <source, GPU>; your T4 run is the exercise." Never invent a T4 result. Validate the notebook JSON with `python3 -c "import json;json.load(open(...))"`, and keep the CUDA simple and careful (compile errors are the main risk).
- **[H] rented H100:** only the profiling card has one: `labs/h100-ncu-roofline.sh`. Its header states the runtime and cost at $2–3.50/hr. The GPU budget cap is $300–400.

## Card list (course folder: perf-3-kernels)
| File | Title | Covers | Lab |
|---|---|---|---|
| _overview.html | The kernel optimization loop | measure → place on roofline → fix the bound → re-measure, plus Hopper/Blackwell vocabulary (main agent, written last) | – |
| execution-model.html | The CUDA execution model | CUDA guide 1.2: grid → block → warp → thread, SIMT, SMs, memory spaces, clusters | [L] occupancy-free index mapping: which thread handles which element, which warp, which SM round |
| first-kernel.html | Your first kernel: vector add | CUDA guide 2.1 + Even Easier Intro: launch config, grid-stride loop, error checking, unified memory and prefetch, the T4 1 → 45× → 1932× ladder | [C] first-kernel.ipynb |
| coalescing.html | Memory coalescing | Coalescing blog (offset/stride) + transpose blog's copy vs naive transpose: 32-byte sectors, 128 B per warp | [L] coalescing.py transaction counter · [C] coalescing.ipynb (offset/stride sweep + naive transpose) |
| shared-memory-transpose.html | Shared memory, tiling and bank conflicts | Transpose blog: transposeCoalesced, 32 banks, tile[32][33] padding, the M2050/K20c table | [L] bank-conflicts.py · [C] transpose.ipynb (all 5 kernels) |
| occupancy.html | Occupancy and latency hiding | Volkov GTC 2010 + siboehm's kernel 3 occupancy calculation: Little's law, TLP vs ILP, the occupancy calculator | [L] occupancy.py (calculator reproducing siboehm's 66% and a T4 version) |
| reduction-divergence.html | Parallel reduction I: divergence and bank conflicts | Harris slides 1–15: the metric, kernels 1–3 | [C] reduction.ipynb (shared with part II; owned by this card) |
| reduction-unrolling.html | Parallel reduction II: unrolling and cascading | Harris slides 16–38: kernels 4–7, Brent's theorem, algorithm cascading, modern warp shuffles | [L] reduction-cost.py (work, steps and cost for Brent) |
| matmul-naive.html | Matmul I: the naive kernel and the lower bound | siboehm intro + kernel 1: results table, 137 GFLOP / 268 MB lower bound, 548 GB naive traffic | [L] matmul-bounds.py · [C] matmul-ladder.ipynb (owned by this card: kernels 1, 2, 3 and cuBLAS via torch) |
| matmul-coalescing-smem.html | Matmul II: coalescing and shared-memory caching | siboehm kernels 2–3: warp layout, 15 → 110 GB/s, SMEM blocking, MIO throttle | [L] matmul-loads.py (owned by this card: loads per result for kernels 1–5) |
| matmul-blocktiling.html | Matmul III: 1D and 2D blocktiling | siboehm kernels 4–5: results per thread, loads per result, arithmetic intensity | reuses matmul-loads.py (don't edit it; ask for changes in your return) |
| matmul-warptiling.html | Matmul IV: vectorize, autotune, warptile | siboehm kernels 6, 7/8, 9, 10, 11 + conclusion; cuBLAS's kernel trace; the power law of effort | [C] matmul-siboehm-repo.ipynb (clone and build siboehm/SGEMM_CUDA for sm_75, run every kernel) |
| tensor-cores-low-precision.html | Tensor cores and low precision | MMA, FP8 E4M3/E5M2, MX block scaling, NVFP4; T4 vs H100 vs B200 tensor throughput | [L] lowprec.py (numpy FP8/MXFP4/NVFP4 quantize-dequantize error) |
| fusion-cuda-graphs.html | Kernel fusion, launch overhead and CUDA graphs | Horace He's three regimes, fusion, CUDA graphs blog, why engines capture decode steps | [C] fusion-graphs.ipynb (eager vs torch.compile vs CUDA graph replay on T4) |
| triton-programming-model.html | Triton: programming blocks, not threads | Triton intro + tutorials 01 vector add and 02 fused softmax | [C] triton-basics.ipynb |
| triton-matmul.html | Triton matmul: grouped ordering and autotuning | Tutorial 03 (+ where 06 fused attention leads, perf-4) | [L] grouped-ordering.py (90 vs 54 block loads) · [C] triton-matmul.ipynb |
| profiling.html | Profiling and benchmarking hygiene | Nsight Systems vs Nsight Compute, Compute Sanitizer, CUTLASS measurement methodology, do_bench | [C] profiling.ipynb (nsys + ncu on T4) · [H] h100-ncu-roofline.sh |

Siblings (relative hrefs): all files above. Previous courses: ../perf-1-foundations/gpu-machine.html, ../perf-1-foundations/roofline-intuition.html, ../perf-2-scaling-book-inference/roofline.html, ../perf-2-scaling-book-inference/critical-batch-size.html.
