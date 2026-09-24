# Shared facts: Inside an inference engine (perf-5)

Every card writer reads this file before writing. Reuse these values exactly. If a card needs a new shared number, cite its source on the card.

## This course
Course 5 of the **Fast LLM inference** series: how an engine like vLLM or SGLang turns one GPU into a service. DEEP resources are walked **section by section, never condensed**: Orca (2 cards), Sarathi-Serve (2 cards), Leviathan speculative decoding (2 cards, with our own Mac measurement), Medusa (1), EAGLE (1). SKIM resources (GPTQ/AWQ/SmoothQuant, XGrammar, S-LoRA/Punica, MLPerf/BurstGPT, vLLM/SGLang architecture) get one card each.
- **Source maps with every number (read first):** ~/Desktop/repos/explainers/tasks/perf-kit/sources/p5-maps.md
- **Full source texts (local only, never publish or paste long passages):** same folder: orca.txt, sarathi.txt, leviathan.txt, chen-spec.txt, medusa.txt, eagle.txt, quant.txt, xgrammar.txt, lora.txt, bench.txt, vllm-arch.txt, plus pagedattention.txt and sglang.txt (perf-4) and ie.txt (Kiely's book; book page = PDF page − 2).
- Papers ran on A100/A40/V100/TPUs/A10G. Show their numbers **as the paper gives them, naming the GPU and model**, then connect to the running example. Anything we recompute is labelled "our recomputation".
- Source quirks to state where relevant: Leviathan's notation is p = target, q = draft, accept with min(1, p/q); **Chen et al. swap the letters** (q target, p draft). Medusa's abstract changed between versions (v1 "2.3–3.6×", v3 "2.3–2.8×" — cite v3). Sarathi's 3.5× claim is attributed to Mistral-7B in the text and Yi-34B in the Fig 12 caption. Sarathi does NOT mention DeepSpeed-FastGen/SplitFuse. vLLM V1 detokenizes in the front-end process (not a separate process); SGLang runs a separate DetokenizerManager.
- Courses that exist for links (folder slugs unchanged; course titles are now "0 · Start here…", "1 · How a GPU runs a language model", "2 · The math of serving an LLM", "3 · Writing fast GPU kernels", "4 · Attention and the KV cache"): ../perf-0-map/, ../perf-1-foundations/, ../perf-2-scaling-book-inference/, ../perf-3-kernels/, ../perf-4-attention-kv/. Especially: ../perf-1-foundations/serving-metrics.html, ../perf-1-foundations/prefill-vs-decode-mac.html, ../perf-2-scaling-book-inference/step-time.html, inference-engine-design.html, speculative-sampling.html, critical-batch-size.html; ../perf-3-kernels/fusion-cuda-graphs.html, tensor-cores-low-precision.html; ../perf-4-attention-kv/paged-memory.html, paged-sharing.html, radixattention.html, cache-aware-scheduling.html, kv-quantization.html, mooncake-scheduling.html. Existing hub content to link rather than duplicate: ../llm-latency/ (speculative-decoding.html, quantization.html, grammar-compilation.html, tail-latency.html, output-speed.html, ttft.html, prompt-caching.html); ../speculative-decoding/speculative-decoding.html (a standalone card); and a whole course ../structured-outputs/ (index.html, token-masking.html, fast-masks.html, compile-vs-per-token.html, fast-forward.html, where-it-runs.html, …) — the structured-outputs card here must be a short engine-level view (XGrammar's overhead and where masking sits in the engine loop) that points to that course for the rest. perf-6 and perf-7 are not built: plain text only. Refer to other courses by their new titles ("course 4, Attention and the KV cache"), not "perf-N", in visible text.

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

## Serving numbers for the running example (from courses 1, 2 and 4; reuse exactly)
- Llama-3.1-8B BF16 on H100: weights 16.06 GB; decode ceiling at batch 1 ≈ 209 tok/s (4.8 ms/step); prefill floor 16.2 ms per 1,000 tokens at 100% MFU; KV 128 KiB/token; 63.9 GB free → 59 sequences of 8k.
- Step formula (perf-2): step ≈ B·KV_seq/BW + max(2·B·N/C, weights/BW). At 2k context: B=1 4.87 ms 205 tok/s → B=238 23.86 ms 9,973 tok/s. At 8k: B=59 23.70 ms 2,489 tok/s.
- B_crit (H100 BF16) ≈ 295 tokens/step.
- perf-2 speculative-sampling card: α=0.8, K=4, 1B draft → 2.24× (468 tok/s) at batch 1; fades to 1.26× at B=128 and 0.70× at B=256 (our model).
- perf-2 inference-engine-design sim (synthetic): static TTFT 7,314 ms → interleaved 46 → disaggregated 41; Orca 36.9×.

## Local measurements on this Mac (Apple M3, 24 GB, llama.cpp 0.4.1 Metal, 2026-09-22/23)
Models: Llama-3.2-3B-Instruct Q4_K_M (1.87 GiB, 3.21B params) = target; Llama-3.2-1B-Instruct Q8_0 (1.22 GiB, 1.24B) = draft (the GGUF files Ollama downloaded).
- `llama-bench -p 512 -n 128 -r 3`: 3B pp512 **407.9 tok/s**, tg128 **28.8 tok/s**; 1B pp512 **1,114 tok/s**, tg128 **43.3 tok/s**. So the draft costs ≈ 28.8/43.3 = **0.665** of a target step (cards use 0.665; predictions code 1.18/1.05/0.57×, chat 0.94/0.54/0.26×, repeat 0.90/0.55/0.23×) (c ≈ 0.66 — far above Leviathan's c < 0.05): a draft this close in size can't buy much.
- **Speculative decoding, acceptance** (`llama-speculative`, temp 0, 200 tokens, target 3B / draft 1B; accepted ÷ drafted — not load-sensitive): code 91.5% / 86.7% / 74.1%, chat 70.2% / 50.7% / 38.0%, repeat-a-list 66.1% / 52.3% / 32.6% for draft length n = 2 / 4 / 8.
- **Predicted speedup** with Leviathan's Thm 3.8, (1 − α^{γ+1}) / ((1 − α)(γc + 1)), using these acceptance rates as α and c = 0.66 (our calculation):
  - code: n=2: accept 91.5% → predicted 1.19×, n=4: accept 86.7% → predicted 1.05×, n=8: accept 74.1% → predicted 0.57×
  - chat: n=2: accept 70.2% → predicted 0.95×, n=4: accept 50.7% → predicted 0.54×, n=8: accept 38.0% → predicted 0.26×
  - repeat: n=2: accept 66.1% → predicted 0.90×, n=4: accept 52.3% → predicted 0.55×, n=8: accept 32.6% → predicted 0.24×
  → at best ~1.2× (code, n = 2); often < 1× (the draft is too expensive). Caveat: llama.cpp's accepted÷drafted is not exactly Leviathan's α (llama.cpp can stop drafting early when the draft is unsure, and acceptance isn't i.i.d.), so treat the prediction as a rough model — it's why n = 8 measured closer to 1× than the formula's 0.26–0.57×.
- **Measured speed** (llama-server, alternating target-only vs target+draft n = 4, 5 runs each, temp 0, 200 tokens): code 15.5 → 16.6 tok/s (1.08×), repeat 16.3 → 15.5 (0.95×); chat 16.4 → 36.4 (2.2×) is an outlier we don't trust. **Caveat, state it on the card:** these timings were taken while the Mac was busy (load average ≈ 33 on 8 cores from other processes); target-only speed fell from 28.8 (llama-bench, quieter) to ~16 tok/s. Treat all timings as rough and re-run the lab on an idle machine. The acceptance rates and the predicted ~0.9–1.2× are the reliable takeaways.
- **Throughput–latency sweep** (llama-server 3B, C concurrent requests each with C parallel slots, ~150-token prompts, 128 output tokens, temp 0, median of 3 runs; same busy-machine caveat):
  | C | total tok/s | per-user tok/s | median TTFT | median TPOT |
  |---|---|---|---|---|
  | 1 | 28.9 | 31.7 | 0.42 s | 31.5 ms |
  | 2 | 48.3 | 28.0 | 0.77 s | 35.7 ms |
  | 4 | 36.1 | 10.6 | 1.73 s | 94.0 ms |
  | 8 | 36.4 | 5.3 | 4.18 s | 188.4 ms |
  | 16 | 100.2 | 8.5 | 5.16 s | 118.2 ms |
  The shape (throughput rises with batch while per-user speed and TTFT get worse) is the textbook trade-off. The non-monotonic dip at 4–8 and jump at 16 repeated in all 3 runs (runs at 16: 100.2 / 92.7 / 102.0); our unverified guess is llama.cpp's Metal backend switching from matrix-vector to matrix-matrix kernels at larger batch sizes — label it "our guess", and don't generalise it to GPUs/vLLM.
Scripts and raw JSON: ~/Desktop/repos/explainers/tasks/perf-kit/p5-measure/ (measure_spec2.py, measure_sweep.py); the card labs reproduce these as `labs/*.py` that call llama-server (see Labs).

## Colour meanings (same on every card)
- green `--accent`: useful compute / tokens that are kept (accepted drafts, decode progress), the winner
- grey `--faint` / `--surface2`: waiting, idle, stalls, overhead, rejected draft tokens, padding
- neutral `--text`/`--muted`: weights and data at rest
- red `--red`: ✗, SLO violations, generation stalls, the bottleneck, the worst value
- `--amber` only for a third roofline region

## Terms (use these names; mention aliases once in the subtitle)
request-level (static) batching vs iteration-level scheduling (continuous batching); selective batching; initiation/increment phase = prefill/decode; generation stall; chunked prefill; stall-free batching; token budget (max_num_batched_tokens); max_num_seqs; admission, preemption (swap/recompute); TTFT, TBT/TPOT/ITL, E2E latency, goodput, capacity (max QPS under SLO); P50/P99; weight-only quantization (GPTQ, AWQ; W4A16), W8A8 (SmoothQuant), FP8, NVFP4; speculative decoding: draft/target, acceptance rate α, draft length γ (K), cost ratio c; tree attention; Medusa heads; EAGLE feature-level draft; MTP; n-gram / prompt-lookup drafts; structured outputs, token mask, context-independent tokens; LoRA adapters, multi-LoRA batching (SGMV); MLPerf scenarios (offline/server/single-stream); burstiness; throughput–latency curve.

## Labs (three tiers)
- **[L] local, on the Mac.** Two kinds: (a) pure simulators/calculators (`labs/<slug>.py`, stdlib + numpy, < 30 s); (b) **measured** labs that drive llama.cpp: `labs/<slug>.py` that starts `llama-server` (installed via `brew install llama.cpp`; the lab must print how to install it and how to point it at GGUF files — e.g. the Ollama blob paths via `ollama show --modelfile`), runs the measurement, and prints results. Writers must NOT run load-sensitive measurements themselves — reuse the measured numbers above; only verify the lab starts and runs a tiny smoke test (e.g. n_predict 8) if needed. Link "Run it: labs/<slug>.py" → https://github.com/seyonv/explainers/blob/main/perf-5-engines/labs/<slug>.py.
- **[C] Colab T4:** not used in this course unless a card says so.
- **[H] rented H100:** `labs/h100-vllm-bench.sh` (owned by the benchmarking card): vLLM Llama-3.1-8B — decode tok/s vs concurrency, BF16 vs FP8, and speculative decoding (EAGLE or ngram, per current vLLM docs) at concurrency 1 vs 64. Header states runtime and cost ($2–3.50/hr; budget cap $300–400). "Not yet run by the author".
- Use your own scratch subfolder named after your card.

## Card list (course folder: perf-5-engines)
| File | Title | Covers | Lab |
|---|---|---|---|
| _overview.html | The throughput–latency frontier | written last by the main agent | – |
| orca-iteration-scheduling.html | Orca I: iteration-level scheduling and selective batching | Orca §1–3: background, C1/S1, C2/S2 | [L] continuous-batching.py (static vs iteration-level sim: TTFT, wasted slots, throughput) |
| orca-system.html | Orca II: the distributed engine and its results | Orca §4–7: architecture, control/data plane, Alg 1 (FCFS, max_bs, KV slot reservation), pipelining, evaluation, 36.9× | – |
| sarathi-stalls.html | Sarathi-Serve I: why prefill stalls decode | Sarathi §1–3: metrics, prefill vs decode cost, piggybacking argument, throughput–latency trade-off, generation stalls, PP bubbles | [L] prefill-decode-cost.py (step-time model: decode batch + prefill chunk, H100 Llama-3.1-8B) |
| sarathi-chunked-prefill.html | Sarathi-Serve II: chunked prefill and stall-free batching | Sarathi §4–6: chunked prefill, Alg 3, token budget, tile quantization, evaluation, ablations, disaggregation | [L] chunked-prefill-sim.py (scheduler sim: vLLM-style vs hybrid vs stall-free, P99 TBT and TTFT) |
| scheduler.html | The scheduler: KV budget, admission and preemption | vLLM V1 unified token-budget scheduler, max_num_seqs, max_num_batched_tokens, gpu_memory_utilization, preemption (recompute default), priorities; ties to PagedAttention | [L] kv-admission.py (admission/preemption sim under a KV budget) |
| engine-anatomy.html | Anatomy of vLLM and SGLang | API server → EngineCore busy loop → scheduler → model runner (CUDA graphs) → sampler → detokenizer; ZMQ; SGLang overlap scheduling; where time goes | – |
| quantization-serving.html | Quantization for serving | GPTQ, AWQ, SmoothQuant, FP8, NVFP4; weight-only vs W8A8 vs FP8 by regime; why 30–50% not 2× (Kiely p.120) | [L] quant-regimes.py (step-time model: which quantization helps decode vs prefill; numpy toy of smoothing α) |
| spec-decoding-math.html | Speculative decoding I: exact sampling with a draft | Leviathan §1–3: algorithm, correctness, α, Eq 1, Thm 3.8, choosing γ, Table 1 | [L] spec-sim.py (numpy: verify exactness; expected tokens and speedup vs α, γ, c) |
| spec-decoding-practice.html | Speculative decoding II: drafts, batch size and our Mac | Leviathan §4 + Chen + our measurement (3B target / 1B draft, c ≈ 0.66) + why speculation fades at large batch | [L] spec-mac.py (measured: llama-server with and without draft; llama-speculative acceptance) |
| medusa.html | Medusa: extra heads and tree attention | Medusa all sections | – |
| eagle.html | EAGLE, MTP and n-gram drafts | EAGLE §1–5 + EAGLE-2/3 + DeepSeek-V3 MTP + n-gram/prompt lookup | – |
| structured-outputs.html | Structured outputs inside the engine | XGrammar only (token-mask cache, context-independent tokens, persistent stack, CPU/GPU overlap, overhead numbers); links the existing structured-outputs course for everything else | [L] token-mask.py (toy grammar: precompute context-independent mask; count per-step work) |
| multi-lora.html | Serving many LoRA adapters | S-LoRA + Punica: unified paging, SGMV, adapter batching | [L] lora-cost.py (bytes/FLOPs of LoRA vs base at rank r; adapters that fit) |
| benchmarking.html | Benchmarking a server | MLPerf scenarios, BurstGPT burstiness, vllm bench serve metrics, goodput, our Mac throughput–latency sweep | [L] server-sweep.py (measured llama-server sweep) · [H] h100-vllm-bench.sh |

Siblings: all files above (relative hrefs).
