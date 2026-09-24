# Shared facts: Training and the cost of a token (perf-7)

Every card writer reads this file before writing. Reuse these values exactly. If a card needs a new shared number, cite its source on the card.

## This course
Course 7 of the **Fast LLM inference** series, and its capstone. Two halves:
1. **Training breadth** (6 cards): enough to follow how a model is trained and to talk about it credibly — a training step and its memory, compute budgets (6ND, Chinchilla, MFU), ZeRO/FSDP, 4D parallelism, precision and recomputation, and post-training (SFT, RL, distillation) with what each hands to inference. Breadth, not PhD depth; the HF **Ultra-Scale Playbook** and **Scaling Book Part 5** are the spine; ZeRO, Chinchilla, Megatron PTD-P, Korthikanti and Llama 3 give the numbers.
2. **Production economics** (6 cards): $ per million tokens and the batch dial, choosing hardware, autoscaling and cold starts, KV-aware routing, reliability and quality, and AI-written kernels (KernelBench) as the frontier.
- **Source maps with every number (read first):** ~/Desktop/repos/explainers/tasks/perf-kit/sources/p7-maps.md (two sections: training; economics/production).
- **Full source texts (local only, never publish or paste long passages):** same folder: ultrascale.txt, training-scalingbook.txt (Scaling Book Parts 5+6), zero.txt, fsdp.txt, chinchilla.txt, palm.txt, llama3.txt, megatron-scaling.txt (Narayanan 2021), activations.txt (Korthikanti 2022 + Chen 2016), mixedprec.txt, posttrain.txt (DeepSeek-R1 + Hinton 2015), deepseek-v3.txt, transformers.txt (Scaling Book Part 4: 6ND), economics.txt, hardware.txt, serverlessllm.txt, routing.txt, reliability.txt, kernelbench.txt, deepseek-infer-overview.txt, dynamo.txt, inference.txt / applied-inference.txt (Scaling Book Parts 7/8), ie.txt (Kiely; book page = PDF page − 2).
- **Source quirks to state where relevant** (details in p7-maps.md):
  - Bytes per parameter of training state: **16** (ZeRO: BF16 weights 2 + BF16 grads 2 + FP32 master 4 + Adam m 4 + v 4), **20** with FP32 gradient accumulation (Playbook; Llama 3), 10 (Scaling Book, which omits master weights and grads), 15 (DeepSeek-V3). "18" appears in none of our sources — don't use it. Say which convention.
  - MFU vs HFU: PaLM 46.2% MFU (45.7% without attention FLOPs) vs 57.8% HFU. Megatron PTD-P's "52% of peak" counts recomputation FLOPs (HFU-like; ≈ 39% as MFU — map's arithmetic, label "our recomputation").
  - Chinchilla never states "20 tokens per parameter"; it's read off Table 3 (70B / 1.4T; 10B / 205B).
  - Llama 3 Table 5 percentages don't all reconcile (148/419 = 35.3% printed as 30.1%); cite the counts and flag.
  - Korthikanti's "5×" activation reduction is sequence parallelism + selective recompute together.
  - DeepSeek-R1 "distillation" is SFT on teacher samples, not Hinton soft targets.
  - NVIDIA headline FP8/BF16 TFLOPS are 2:4-**sparse**; dense is half (H100 dense BF16 989, FP8 1,979). MI300X is 192 GB, 5.3 TB/s (256 GB is MI325X).
  - DeepSeek Day 6: 545% is profit ÷ cost (84.5% as a share of revenue); $2/hr is an assumed lease price; revenue is theoretical at R1 prices.
  - Vendor $/M-token comparisons often change engines too (e.g. NVIDIA's B200 vs H100 GPT-OSS claim compares TensorRT-LLM vs vLLM).
  - Dynamo's "30×" is a launch-blog projection. KernelBench Table 1's column order was reconstructed from garbled PDF text — cite only the abstract's "less than 20%" and the R1 36%→72% Level-2 feedback result unless you verify the table.
- Courses that exist for links (titles: "0 · Start here", "1 · How a GPU runs a language model", "2 · The math of serving an LLM", "3 · Writing fast GPU kernels", "4 · Attention and the KV cache", "5 · Inside an inference engine", "6 · Serving across many GPUs"): ../perf-0-map/ … ../perf-6-scaling-out/. Especially: ../perf-1-foundations/flops-per-token.html, serving-metrics.html; ../perf-2-scaling-book-inference/step-time.html, serving-llama-70b.html, critical-batch-size.html; ../perf-3-kernels/tensor-cores-low-precision.html, profiling.html; ../perf-4-attention-kv/radixattention.html, cache-aware-scheduling.html, mooncake-scheduling.html; ../perf-5-engines/quantization-serving.html, benchmarking.html, spec-decoding-practice.html, eagle.html, scheduler.html; ../perf-6-scaling-out/collectives.html, megatron-mlp-attention.html, megatron-scaling.html, pipeline-data-parallel.html, moe-basics.html, expert-parallel.html, deepseek-v3-architecture.html, deepseek-v3-infrastructure.html, distserve-placement.html, disaggregation-in-production.html, sizing-deployment.html. Existing hub content: ../llm-latency/ (hardware.html, tail-latency.html, prompt-caching.html, parallel-calls.html). Refer to courses by their titles, not "perf-N", in visible text.

## The reader
- Setup: Apple M3 (8-core CPU), 24 GB unified memory (~100 GB/s), Python 3 with numpy 2.2 (no torch). No NVIDIA GPU: labs are simulators/calculators [L] plus one rented-H100 script [H].
- Knows almost nothing about GPU/AI performance engineering. Is preparing to join a company that serves LLM inference fast and cheaply (wafer.ai-like — it works on AI-written GPU kernels). Wants connected, rigorous understanding: enough to follow the full training→serving flow and to talk shop, not PhD depth.
- Has read courses 1–6: roofline, B_crit ≈ 295, step-time formula, KV cache, paging, continuous batching, speculative decoding, TP/PP/EP, MoE, disaggregation.

## Running examples
**1. Llama-3.1-8B, BF16, one H100 SXM (serving).** 8.03B params, 16.06 GB, KV 128 KiB/token, decode ceiling ≈ 209 tok/s at batch 1.
Step formula (course 2) at 2k context, step = B·KV_seq/BW + max(2·B·N/C, W/BW) — computed 2026-09-24:
| B | step | tok/s | $/M tokens at $3.99/GPU-hr |
|---|---|---|---|
| 1 | 4.87 ms | 205 | 5.40 |
| 8 | 5.44 ms | 1,472 | 0.75 |
| 16 | 6.08 ms | 2,633 | 0.42 |
| 32 | 7.36 ms | 4,349 | 0.25 |
| 64 | 9.92 ms | 6,450 | 0.17 |
| 128 | 15.05 ms | 8,505 | 0.13 |
| 238 | 23.86 ms | 9,973 | 0.11 |
These are ceilings (100% of bandwidth, output tokens only, no prefill cost); label them "our model". Market price for comparison (OpenRouter, 2026-09-24): Llama-3.1-8B-Instruct DeepInfra FP8 $0.02 in / $0.04 out per M; Groq $0.05/$0.08; CoreWeave BF16 $0.22/$0.22.

**2. Llama-3.1-8B, training (the training example).** 6ND with ~15T tokens (Llama 3 herd: "15T+"; label approx): 6 · 8.03e9 · 15e12 = **7.2e23 FLOPs** → at 40% MFU on H100 (989 TF dense) ≈ **0.51M H100-hours** (our recomputation; cite Meta's own GPU-hour figure only if you open its source). Training state at 16 bytes/param = **128.5 GB** (20 bytes → 160.6 GB) → doesn't fit one 80 GB H100 before activations → needs ZeRO/FSDP.

**3. Llama 3 405B (the big training example).** 15.6T tokens; 6·405e9·15.6e12 = 3.79e25 ≈ paper's 3.8e25 FLOPs; 16K H100s; Table 4: 8K GPUs TP8 PP16 DP64 → 430 TFLOPs/GPU 43% MFU; 16K GPUs DP128 → 400 TF 41%; CP16 at 131K seq → 380 TF 38%. 54-day snapshot: 466 interruptions, 47 planned, 419 unexpected, ~78% hardware; faulty GPU 148, HBM3 72.

**4. DeepSeek-V3 / R1 (the MoE and cost example).** 2.788M H800 GPU-hours = $5.576M at $2/hr (pretraining 2,664K, context 119K, post 5K; excludes prior research/ablations); 14.8T tokens; 180K GPU-hours per trillion tokens. R1 RL cost 147K H800 GPU-hours ($294K). Day 6 serving: 226.75 nodes avg × 8 H800 × $2 × 24 = **$87,072/day**; ~73.7k input tok/s (prefill) and ~14.8k output tok/s (decode) per node; theoretical revenue $562,027/day; margin 545% (profit ÷ cost). Derived: ≈ $0.30/M output tokens per decode node, ≈ $0.06/M input per prefill node (map's arithmetic; label ours).

## Hardware and prices (reuse exactly; prices seen 2026-09-24)
- H100 SXM: 989 dense BF16 TFLOPS, 1,979 dense FP8, 3.35 TB/s, 80 GB, ~700 W. H200: same compute, 141 GB, 4.8 TB/s. B200 (per GPU, DGX B200 ÷ 8): 4.5 PF dense FP8, 9 PF dense FP4, 180 GB, 8 TB/s (Kiely says 192 GB / ~5 PF — flag). MI300X: 192 GB, 5.3 TB/s. Full table with sources in p7-maps.md §2.
- H100 on-demand $/GPU-hr: Lambda $3.99 (8×) / $4.29 (1×); RunPod $3.49; Modal $3.95; AWS p5 $6.88 ($55.04/node). B200 ≈ $6.25–6.99. **Use $3.99 (Lambda 8×) as the default** in worked examples and say so.
- Kiely: ≈ 1 failure per 50,000 GPU-hours; one 8-GPU node for a year = 70,080 GPU-hours.

## Colour meanings (same on every card)
- green `--accent`: useful compute / tokens that are kept / the winner / cheapest
- grey `--faint` / `--surface2`: communication, waiting, idle, bubbles, overhead, recompute
- neutral `--text`/`--muted`: weights, optimizer state and data at rest
- red `--red`: ✗, SLO violations, the bottleneck, the worst/most expensive value, failures
- `--amber` only for a third roofline region

## Terms (use these names; mention aliases once in the subtitle)
forward / backward / optimizer step; gradients; Adam moments (m, v); master weights; activations; activation checkpointing (recomputation, remat), selective recompute; gradient accumulation; micro-batch, global batch; 6ND; compute-optimal (Chinchilla); MFU vs HFU; data parallelism (DP), ZeRO-1/2/3, FSDP, sharding; tensor (TP), sequence (SP), context (CP), pipeline (PP), expert (EP) parallelism; 1F1B, interleaved schedule, bubble; mixed precision, loss scaling, BF16, FP8, delayed vs fine-grained scaling; SFT, RLHF, PPO, GRPO, reward model, rollout, distillation; $/M tokens, blended price, GPU-hour, utilization; cold start, scale to zero, autoscaling; prefix-aware / KV-aware routing, endpoint picker; canary, blue-green, shadow traffic, quality regression, eval recovery; KernelBench, fast_p, speed of light.

## Labs (three tiers)
- **[L] local, on the Mac:** simulators/calculators only (`labs/<slug>.py`, stdlib + numpy, < 30 s, no torch). Link "Run it: labs/<slug>.py" → https://github.com/seyonv/explainers/blob/main/perf-7-training-economics/labs/<slug>.py.
- **[C] Colab:** not used in this course.
- **[H] rented H100:** `labs/h100-cost-sweep.sh` (owned by cost-per-token.html): vLLM Llama-3.1-8B BF16 and FP8, `vllm bench serve` at concurrency 1/8/32/64/128/256, prints output tok/s and computes $/M tokens at a price you pass in. Header states runtime (~30–45 min) and cost (~$2–3). "Not yet run by the author".
- Use your own scratch subfolder named after your card.

## Card list (course folder: perf-7-training-economics)
| File | Title | Covers | Lab |
|---|---|---|---|
| _overview.html | The whole stack, every lever ranked | capstone, written last by the main agent | – |
| training-step.html | A training step and where its memory goes | forward/backward/optimizer; 2NPM + 4NPM = 6 per param per token; ZeRO §3.1 16Ψ accounting (and the 20/10/15 conventions); Playbook memory table (7B 112/140 GB); activation memory sbh(34 + 5as/h) per layer (Korthikanti), gradient accumulation; Llama-3.1-8B worked example | [L] train-memory.py (weights/grads/optimizer/activations for 8B, 70B, 405B vs batch and sequence; what fits one H100) |
| compute-budgets.html | Compute budgets: 6ND, Chinchilla and MFU | 6ND derivation (Scaling Book Part 4; attention correction 12·L·h·s), Chinchilla approaches and Table 3, fitted loss (E = 1.69, exponents), "over-training" small models for cheap inference (Llama 3 8B on 15T ≫ 20 tokens/param — our framing, label), MFU vs HFU with PaLM's worked 45.7/46.2/57.8%, Llama 3 405B 3.8e25 and 38–43% MFU, DeepSeek-V3 GPU-hours | [L] compute-budget.py (6ND → GPU-hours → $ for 8B/70B/405B at MFU 30–50%; Chinchilla-optimal N, D for a budget) |
| zero-fsdp.html | Data parallelism, ZeRO and FSDP | DP all-reduce of gradients (overlap, bucketing), ZeRO-1/2/3 memory formulas and the 7.5B on 64 GPUs example (120 → 31.4 → 16.6 → 1.9 GB), communication 2Ψ vs 3Ψ, FSDP (PyTorch paper) as ZeRO-3, Scaling Book Part 5 DP/FSDP compute-bound conditions | [L] zero-memory.py (per-GPU memory by stage and DP size; comm volume per step and time at NVLink/IB bandwidth) |
| parallelism-4d.html | 4D parallelism: TP, PP, CP and EP in training | how training differs from serving (backward, activations, optimizer); TP + sequence parallel inside a node; PP schedules AFAB/1F1B/interleaved with bubble (p−1)/m and (p−1)/(v·m), zero-bubble/DualPipe; CP (ring attention) for long sequences; EP; Llama 3 Table 4 layouts; Playbook rules of thumb and its 5D summary; Scaling Book Part 5 TP/FSDP limits; Narayanan 502 PF / 3072 A100 (52% HFU caveat) | [L] pp-bubble.py (step-time sim for p stages, m micro-batches, v chunks: bubble fraction and memory) |
| precision-recompute.html | Mixed precision, FP8 and activation checkpointing | Micikevicius: FP32 master weights (why: ~5% of gradients < 2⁻²⁴), loss scaling (8…32K, FP16 max 65,504), BF16 removing the need; FP8 training: delayed (per-tensor) vs DeepSeek-V3 fine-grained scaling (1×128/128×128, < 0.25% loss error); recompute: full (~one extra forward, 6ND → 8ND), Chen √n, selective (Korthikanti: 70% less activation memory for 2.7% more FLOPs) | [L] loss-scaling.py (numpy float16: gradient underflow with and without loss scaling; bf16 emulation; checkpointing memory/compute trade-off curve) |
| post-training.html | From pretrained to served: SFT, RL and distillation | what each stage is and costs; SFT; RLHF with PPO (value model ≈ policy size) vs GRPO (group-normalized advantage), R1-Zero numbers; the inference inside RL (rollouts on vLLM with MTP speculative decoding, 8,192 outputs per step, 32k→65k lengths); R1 cost table (147K GPU-hours); distillation (R1 SFT on 800k samples into Qwen/Llama bases vs Hinton soft targets); what inference receives (checkpoint, precision, context length, reasoning-length output distribution → decode-heavy load) | – |
| cost-per-token.html | $ per million tokens and the batch dial | $/M = GPU $/hr ÷ (tok/s × 3600) × 1e6; the table above; input vs output pricing (prefill is cheap per token, decode is not); utilization and idle time; market prices (OpenRouter) vs our ceilings; DeepSeek Day 6 as a real P&L ($87,072/day, 545% with caveats); Scaling Book "~100× cheaper for 2× latency" (book code: 77× for 1.57×, course 2) | [L] cost-model.py (step formula → tok/s → $/M vs batch, context, GPU price, utilization, FP8) · [H] h100-cost-sweep.sh |
| choosing-hardware.html | Choosing hardware: FLOPS, bandwidth, memory and price | spec table (H100/H200/B200/GB200/L40S/A100/MI300X/MI325X) with sources; sparse vs dense trap; which spec matters for which regime (decode → TB/s and GB; prefill → FLOPS; big MoE → GB and interconnect); $ per TB/s-hour and $ per dense PFLOP-hour; MLPerf v5.0 Llama-2-70B 8×H200 offline 34,988 tok/s; vendor claims with engine caveats; link ../llm-latency/hardware.html | [L] hw-compare.py (decode ceiling and $/M for 8B/70B on each GPU at batch 1 and at B_crit; which GPU wins per regime) |
| autoscaling-cold-starts.html | Autoscaling, cold starts and scale to zero | Kiely §7.2 (four cold-start parts, autoscaling signals, scale to zero, min replicas), weight-load math (16 GB at 1 Gbps 128 s … PCIe 5 0.25 s; 70B 140 GB at 5 GB/s 28 s), ServerlessLLM (loading-optimized checkpoints 3.6–8.2×, multi-tier, live migration of tokens not KV, 10–200× with its caveat, 70B into 8 GPUs 84 s with PyTorch), CUDA graph capture / compile time; cost of idle vs latency of cold starts | [L] cold-start.py (time to first token after scale-up by component and storage tier; cost of keeping N warm replicas vs traffic) |
| kv-aware-routing.html | KV-aware routing: sending requests where their cache lives | why round-robin wastes prefix caches (link course 4's RadixAttention and cache-aware scheduling); Dynamo router cost function with its worked example (18/10/11 → less-loaded wins); llm-d EPP filters → scorers → pickers, approximate vs precise prefix modes; Gateway API Inference Extension; published numbers (llm-d 3× throughput / 2× TTFT on MI300X; Baseten 2× TTFT via Dynamo) with sources and caveats | [L] kv-router.py (sim: N replicas, shared-prefix workload; round-robin vs prefix-affinity vs cost-function routing: hit rate, load imbalance, TTFT) |
| reliability-quality.html | Reliability and quality: failures, canaries and evals | Llama 3 interruption table (with the % quirk), Kiely's 1 per 50k GPU-hours and 70,080 GPU-hours per node-year, node-level health (cordon, cycle), active-active vs active-passive, canary vs blue-green ("another 100 GPUs"), checking a speed change for quality (Red Hat quantized Llama 3.1 recovery ≥ 96–99%, speculative decoding is exact but MTP/Medusa typical acceptance is not, lm-eval), tail latency link | [L] failure-sim.py (expected failures per day for a fleet; availability of TP8 vs replicas; blast radius) |
| kernelbench.html | AI-written kernels: KernelBench and the frontier | KernelBench (250 tasks, levels 1/2/3 = 100/100/50, fast_p = correct AND ≥ p× faster, one-shot < 20%, R1 Level 2 36% → 72% with feedback, v0.1 fixes: reward hacking, 43 flawed tasks, speed-of-light check 0.138 ms for H100 BF16 4096² matmul), why this matters for an inference company (link course 3's kernel ladder and profiling) | [L] fast-p.py (compute fast_p from a table of (correct, speedup); speed-of-light time for a matmul/attention shape on H100 with dense vs sparse peak) |

Siblings: all files above (relative hrefs).
