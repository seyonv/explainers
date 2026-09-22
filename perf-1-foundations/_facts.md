# Shared facts: Foundations (perf-1)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
The prerequisites for reading the Scaling Book inference chapter (course perf-2): what a transformer computes, what attention is, what a GPU is, the roofline intuition, the metrics, and prefill vs decode measured on the reader's own Mac. Keep it beginner-friendly but exact; the rigorous treatment of roofline/critical batch/KV math lives in perf-2 — link forward to `../perf-2-scaling-book-inference/<card>.html` (cards: roofline, critical-batch-size, attention-intensity, step-time, params-vs-kv, kv-cache-sampling).
Primary reading for this course: Attention Is All You Need §3.1–3.2 (https://arxiv.org/abs/1706.03762); CUDA C++ basics (https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html); Roofline paper (https://users.cs.duke.edu/~lkw34/papers/roofline-cacm2008.pdf); Transformer Inference Arithmetic (https://kipply.github.io/blog/transformer-inference-arithmetic/); Etalon (https://arxiv.org/html/2407.07000); Scaling Book Part 4 transformer math (verbatim text: ~/Desktop/repos/explainers/tasks/perf-kit/sources/transformers.txt) and Part 12 GPUs (~/Desktop/repos/explainers/tasks/perf-kit/sources/gpus.txt); Inference Engineering (Kiely 2026) ch. 1.4, 2.2, 2.4, 3.1 — extracted text at ~/Desktop/repos/explainers/tasks/perf-kit/sources/ie.txt (book page = PDF page − 2).

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
- Prefill per token is ≈13× faster than decode per token at 244 prompt tokens (188 ÷ 14.12 = 13.3): prefill is parallel over tokens, decode is serial.

## Sourced numbers
| Fact | Value | Source | As of |
|---|---|---|---|
| H100 SXM specs | 989 TF BF16 dense, 1979 TF FP8 dense, 3.35 TB/s, 80 GB | NVIDIA H100 datasheet; Scaling Book Part 12 (GPUs) | 2025 |
| H200 / B200 | 141 GB 4.8 TB/s / 192 GB 8 TB/s, 2250 TF BF16 | Scaling Book Part 12 | 2025 |
| TPU v5e | 1.97e14 BF16 FLOP/s, 3.94e14 int8, 8.2e11 B/s HBM, 16 GB HBM, ICI 4.5e10 B/s one-way | Scaling Book Parts 1, 7 | 2025 |
| Critical batch size | v5e 240; int8 weights+bf16 compute 120; H100 ≈ 295 (book also says "about 280") ; B200 281 | Scaling Book Parts 1, 7, 12 | 2025 |
| M3 memory bandwidth | ~100 GB/s | Apple M3 spec | — |
| Kiely ops:byte for H100 FP16 | 989/3.35 ≈ 295 | Inference Engineering (Kiely, 2026) p.62 | 2026 |

## Colour meanings (same on every card)
- green `--accent`: compute / useful work, the KV cache in memory diagrams, the winner or faster option
- grey `--faint` / `--surface2`: time spent waiting on memory or communication, overhead, idle, free memory
- neutral `--text`/`--muted`: model weights/parameters in memory diagrams
- red `--red`: ✗, limits (out of memory, the ridge/critical cutoff line), the bottleneck, the worst value

## Terms (use these names; mention aliases once in the subtitle)
prefill; decode (a.k.a. generation); KV cache; arithmetic intensity (ops:byte); roofline; critical batch size B_crit; memory-bound / compute-bound; HBM; TTFT; TPOT / inter-token latency; throughput (tokens/s, total); MFU; GQA/MQA/MHA; tensor (model) parallelism; data parallelism; FSDP; continuous batching; prefix caching; disaggregated serving; speculative decoding.

## Labs
Each card with a ⚗️ lab ships `labs/<card-slug>.py`: runs with `python3 labs/<card-slug>.py`, stdlib + numpy only, CPU, < 30 s, prints the numbers the card uses (and ends with 1–3 "try this" suggestions as comments). The card links it as `https://github.com/seyonv/explainers/blob/main/<course-folder>/labs/<card-slug>.py` with text "Run it: labs/<card-slug>.py". Every number on the card that the lab computes must match the lab's output.

## Card list (course folder: perf-1-foundations)
| File | Title |
|---|---|
| _overview.html | Foundations on one card (main agent, written last) |
| transformer-forward.html | The transformer forward pass: where the parameters live |
| attention.html | Attention: Q, K, V, heads, and GQA |
| flops-per-token.html | FLOPs per token: 2N to serve, 6N to train |
| gpu-machine.html | The GPU as a machine |
| roofline-intuition.html | Compute-bound or memory-bound? The roofline intuition |
| serving-metrics.html | Serving metrics: TTFT, TPOT, throughput, goodput, percentiles |
| prefill-vs-decode-mac.html | Prefill vs decode, measured on your Mac |
