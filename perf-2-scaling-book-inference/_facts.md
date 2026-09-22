# Shared facts: Inference, by the Scaling Book (perf-2)

Every card writer reads this file before writing. Reuse these values exactly. If a card needs a new shared number, cite its source on the card.

## This course
A section-faithful walk through **How to Scale Your Model, Part 7: "All About Transformer Inference"** (https://jax-ml.github.io/scaling-book/inference/) plus **Part 8: "Serving LLaMA 3-70B"** (https://jax-ml.github.io/scaling-book/applied-inference/). The reader was told this chapter is the most important resource, so **do not condense**: each card covers its section's equations, numbers, figures and questions faithfully, in the book's own notation (B, T, S, D, F, N, K, H, L, V; B_crit; α; β; W_hbm; W_ici).
- Verbatim text of the chapters (use for exact wording, equations, tables and answers): ~/Desktop/repos/explainers/tasks/perf-kit/sources/inference.txt, ~/Desktop/repos/explainers/tasks/perf-kit/sources/applied-inference.txt, ~/Desktop/repos/explainers/tasks/perf-kit/sources/roofline.txt, ~/Desktop/repos/explainers/tasks/perf-kit/sources/transformers.txt, ~/Desktop/repos/explainers/tasks/perf-kit/sources/gpus.txt
- The book's numbers are **TPU v5e** unless stated. Show them as the book gives them, and put the **H100 equivalent** alongside (B_crit 295; H100 specs from the table below). Anything we recompute for H100 or for Llama-3.1-8B is labelled "our recomputation, not in the book".
- Book quirks to state where relevant: the LLaMA-2-13B tables mix GB and GiB; "batch" means tokens per step, not sequences.
- Each card's footer names the exact book section it covers ("Read: Part 7 §… — read all of it") so the reader can go read the original next.

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

## Card list (course folder: perf-2-scaling-book-inference)
| File | Title | Book section |
|---|---|---|
| _overview.html | The inference chapter on one card | whole chapter (main agent, written last) |
| kv-cache-sampling.html | Sampling and the KV cache | Basics intro: naive vs cached, prefill vs generation |
| what-we-optimize.html | What we optimize: TTFT, latency, throughput | "What do we actually want to optimize?" + granular view |
| roofline.html | The roofline, properly | Part 1 Rooflines (as needed by Part 7) |
| critical-batch-size.html | Linear layers and the critical batch size | "Linear operations: what bottlenecks us?" |
| attention-intensity.html | Attention's arithmetic intensity | "What about attention?" (+ Part 4 Q4 GQA intensity) |
| step-time.html | The step-time formula | "Theoretical estimates for LLM latency and throughput" + Pop Quiz |
| params-vs-kv.html | Memory: parameters vs KV cache | "What about memory?" |
| modeling-13b.html | Modelling LLaMA 2-13B on 8×v5e | "Modeling throughput and latency for LLaMA 2-13B" |
| kv-tricks.html | Tricks that shrink the KV cache | "Tricks for Improving Generation Throughput and Latency" |
| sharding-prefill-generation.html | Sharding prefill vs generation | "Distributing Inference" → Prefill, Generation |
| sharding-kv-cache.html | Sharding the KV cache | "Sharding the KV cache" + Appendix C latency-bound comms (+ Appendix B briefly) |
| inference-engine-design.html | Designing an inference engine | "Designing an Effective Inference Engine": batched/interleaved/disaggregated, continuous batching, prefix caching, JetStream |
| serving-llama-70b.html | Serving LLaMA 3-70B end to end | Part 8 |
| worked-problems.html | The chapter's worked problems | Part 7 Worked Problems Q1–Q7 |
| speculative-sampling.html | Speculative sampling and the batch-240 rule | Appendix D + Appendix A |
