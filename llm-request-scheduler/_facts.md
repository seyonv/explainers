# Shared facts: Build an LLM request scheduler

Every card writer reads this file before writing. Reuse these values exactly. If a card needs a new shared number, cite its source on the card.

## This course
**Title: "Build an LLM request scheduler".** A standalone course in the explainers hub (folder `llm-request-scheduler/`). It is problem-first: it takes the reader from "what is one forward step" to a working scheduler (code, data structures, complexity), and ends with the whole serving system drawn out.

**Framing rule (hard):** the course is framed as a practical engineering problem ("the build-it exercise"), never as preparation for any hiring process, and it never names a company as the place the problem comes from. Company names appear only as *builders of real systems* (vLLM, SGLang, NVIDIA, Character.AI, Moonshot/Kimi, DeepSeek, Google, Red Hat/llm-d, Databricks, Anyscale, Hugging Face…).

The three concept questions the course must answer crisply (on their own cards and in explain-it-back): (1) the main performance bottlenecks in LLM inference, (2) the role of the KV cache during generation, (3) ways to increase inference throughput. The coding core covers token batching strategies, handling concurrent requests, and combining requests to maximise GPU throughput, with an emphasis on data structures, scheduling logic and complexity.

### Research kit (read the parts relevant to your card; cite URLs from here)
- `~/Desktop/repos/explainers/tasks/scheduler-kit/research-companies.md`: 35 verified sources (ByteByteGo posts, Anyscale, the vLLM blog, Character.AI, DeepSeek, llm-d, SGLang, GKE, Dynamo, TensorRT-LLM, Together, Meta, Mooncake, Splitwise, DistServe, Llumnix, VTC, FastServe, Andes…) with exact numbers. It ends with a topic → sources map and caveats. ByteByteGo has **no** dedicated continuous-batching post and **no** full gateway→router→replica diagram; don't claim otherwise.
- `~/Desktop/repos/explainers/tasks/scheduler-kit/research-engines.md`: verified engine internals from source code at named commits: vLLM V1 (scheduler.py, the free-block queue as an O(1) doubly linked list, block hashing, defaults), SGLang (policies, new_token_ratio, retraction, radix tree, overlap scheduler), TGI (reserve-to-completion, waiting_served_ratio), TensorRT-LLM (GUARANTEED_NO_EVICT vs MAX_UTILIZATION), the Triton dynamic batcher, nano-vllm, and public practice exercises.
- Only state what these files say the source says. Mark inferences.

### Existing hub cards to link for depth (don't re-teach them at length; "go deeper →" links)
- Course 1, *How a GPU runs a language model*: ../perf-1-foundations/serving-metrics.html, prefill-vs-decode-mac.html, roofline-intuition.html, attention.html
- Course 2, *The math of serving an LLM*: ../perf-2-scaling-book-inference/step-time.html, critical-batch-size.html, inference-engine-design.html, params-vs-kv.html, kv-cache-sampling.html
- Course 4, *Attention and the KV cache*: ../perf-4-attention-kv/paged-memory.html, paged-sharing.html, paged-results.html, radixattention.html, cache-aware-scheduling.html, mooncake-scheduling.html, mooncake-store.html, gqa.html, kv-quantization.html
- Course 5, *Inside an inference engine*: ../perf-5-engines/orca-iteration-scheduling.html, orca-system.html, sarathi-stalls.html, sarathi-chunked-prefill.html, scheduler.html, engine-anatomy.html, benchmarking.html, quantization-serving.html, spec-decoding-practice.html
- User-side course: ../llm-latency/ttft.html, prompt-caching.html, tail-latency.html, parallel-calls.html
- Overview map: ../perf-0-map/stack.html
Refer to them by course title in visible text, never "perf-N".

## The reader
- Setup: Apple M3 (8-core CPU), 24 GB unified memory, Python 3.10, llama.cpp/Ollama installed. No NVIDIA GPU.
- Has read some of the perf series. Wants to be able to **build** a simplified scheduler in about 45 minutes from a blank file, state the complexity of each operation, and then **explain** bottlenecks, the KV cache and throughput levers with numbers. Wants trade-offs and alternatives, not a single answer.

## The running example
**Model/hardware (same as the perf series): Llama-3.1-8B, BF16, on one H100 SXM.**
- 8.03B params; weights **16.06 GB**; KV per token = 2·L·K·H·2 bytes = 2·32·8·128·2 = **131,072 B = 128 KiB**; an 8,192-token sequence = **1.07 GB** of KV
- H100 SXM: **989 TFLOPS** dense BF16, **3.35 TB/s** HBM3, **80 GB** → ops:byte ≈ **295** = critical batch size B_crit ≈ 295 tokens/step
- Free HBM after weights: 80 − 16.06 = **63.9 GB** → **487,504 tokens** of KV = **30,469 blocks** of 16 tokens → **238 sequences at 2k context**, **59 at 8k**
- Decode ceiling at batch 1: **4.9 ms/step ≈ 205 tok/s** (at 2k context; 209 tok/s ignoring the KV read)
- Prefill floor: **16.2 ms per 1,000 tokens** at 100% MFU; one 4,096-token prefill alone = **66.7 ms**

**The cost model** (`labs/scheduler.py`, `CostModel`): step = max(2·P·T / C, W / BW) + KV_bytes_read / BW, with T = tokens in the step, and the KV read covering every scheduled request's context. It is a **floor at 100% of peak**: real engines are slower by their kernel efficiency and CPU overhead. Say "our simulator" or "our model" on every card that uses it.

**The workload** (`poisson_trace`): Poisson arrivals; prompt lengths lognormal (median 512, σ 0.8, capped at 4,096); output lengths lognormal (median 200, σ 0.8, capped at 1,024); seeded. **SLO used throughout: TTFT ≤ 1 s and TPOT ≤ 50 ms.** Goodput = requests/s that met both.

**The toy trace** (8 requests, used for step-by-step traces; arrival ms, prompt, output):
0: 0, 12, 4 · 1: 0, 5, 9 · 2: 0, 20, 2 · 3: 1, 7, 6 · 4: 2, 30, 3 · 5: 4, 4, 8 · 6: 6, 16, 5 · 7: 10, 9, 7.
Toy config: token budget 16, max 4 sequences, 12 KV blocks of 4 tokens (48 tokens of KV), watermark 0.

## Measured / simulated numbers (all from `labs/experiments.py`, 2026-09-24; full tables in `labs/results.md`)
Every number below is **our simulation** unless marked otherwise. Link "Run it: labs/experiments.py" → https://github.com/seyonv/explainers/blob/main/llm-request-scheduler/labs/experiments.py and the code → .../labs/scheduler.py.

**E1 · step time vs batch (decode, 2k context each)**
| B | step ms | total tok/s | per-user tok/s |
|---|---|---|---|
| 1 | 4.9 | 205 | 205 |
| 8 | 5.4 | 1,472 | 184 |
| 32 | 7.4 | 4,349 | 136 |
| 64 | 9.9 | 6,450 | 101 |
| 128 | 15.1 | 8,505 | 66 |
| 238 (memory max at 2k) | 23.9 | 9,973 | 42 |
→ 49× the throughput from 1 to 238, while per-user speed falls 205 → 42 tok/s.

**E2 · batching strategies** (1,000 requests)
| rate | strategy | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | in SLO | padding waste |
|---|---|---|---|---|---|---|---|
| 2 req/s | static B=32 | 511 | 8,698 ms | 21,650 ms | 8.0 ms | 1% | 71% |
| 2 req/s | dynamic B=32, 50 ms window | 518 | 1,957 ms | 8,065 ms | 5.4 ms | 27% | 58% |
| 2 req/s | continuous + chunked (2,048) | 521 | 11.1 ms | 59.5 ms | 5.0 ms | 100% | 0% |
| 20 req/s | static B=32 | 913 | 118 s | 233 s | 8.0 ms | 0% | 71% |
| 20 req/s | dynamic | 930 | 113 s | 227 s | 8.0 ms | 0% | 70% |
| 20 req/s | continuous (whole-prompt prefill) | 4,756 | 16.2 ms | 75.4 ms | 7.2 ms | 100% | 0% |
| 20 req/s | continuous + chunked | 4,755 | 16.1 ms | 83.1 ms | 7.1 ms | 100% | 0% |
| 60 req/s | static | 916 | 134 s | 265 s | 8.0 ms | 0% | 71% |
| 60 req/s | continuous + chunked | 9,102 | 1,397 ms | 4,816 ms | 23.7 ms | 41% | 0% |
- Saturated throughput: static ≈ **916 tok/s** vs continuous ≈ **9,100 tok/s** → **≈ 10×** (our model; Anyscale measured 23× on OPT-13B/A100, 8× from continuous batching alone, see the kit).
- Capacity (highest rate with ≥ 99% of requests in SLO, **3,000-request traces, steady state**): continuous + chunked **40 req/s** (41 → 66% in SLO); under a tight SLO (TTFT ≤ 350 ms, TPOT ≤ 25 ms) **38 req/s**. Static and dynamic B=32 **fail even at 1 req/s** (0% and 54% in SLO: a request waits behind a whole batch that runs to its longest output). An earlier 600-request search gave 51 req/s, a short-trace artifact: the queue never reached steady state.
- "Padding waste" = the fraction of decode slot-steps where a row had already finished but still sat in the batch.

**E3 · chunked prefill** (long prompts: median 2,048, 8 req/s, 600 requests)
| config | out tok/s | TTFT p50 | TTFT p99 | ITL p99 | ITL max | TPOT p99 |
|---|---|---|---|---|---|---|
| whole prompt | 2,009 | 48.6 | 200.9 | 68.7 | 185.1 | 15.5 |
| budget 256 | 2,033 | 82.4 | 375.5 | 7.2 | 7.4 | 7.1 |
| budget 512 | 2,023 | 61.0 | 211.0 | 11.2 | 11.7 | 9.8 |
| budget 1,024 | 2,015 | 54.1 | 187.6 | 19.7 | 20.2 | 13.1 |
| budget 2,048 | 2,011 | 51.5 | 179.4 | 36.2 | 37.1 | 14.5 |
| budget 4,096 | 2,009 | 48.9 | 195.7 | 68.3 | 70.4 | 15.3 |
(all times in ms). A 4,096-token prefill joined to 64 decodes at 2k: **72.8 ms** vs **9.9 ms** for the decodes alone (a 7.4× stall). The token budget trades ITL (smaller is smoother) against TTFT (smaller is slower to first token). Note: our model has no per-chunk overhead, so real small budgets cost a bit more (the Sarathi-Serve paper measured up to ~25% chunking overhead at 512 and ~0 at 2,048, Fig 14, Yi-34B; Course 5's +4.8% is that course's own model, not a measurement).

**E4 · KV memory and preemption** (rate 30, 1,000 requests)
| KV blocks (tokens) | GB | out tok/s | TTFT p99 | TPOT p50 | preemptions | recomputed tokens |
|---|---|---|---|---|---|---|
| 30,469 (487,504) | 63.9 | 6,736 | 106.7 ms | 10.0 ms | 0 | 0 |
| 8,000 (128,000) | 16.8 | 6,736 | 106.7 ms | 10.0 ms | 0 | 0 |
| 4,000 (64,000) | 8.4 | 6,108 | 4,146 ms | 9.8 ms | 63 | 112,062 |
| 2,000 (32,000) | 4.2 | 4,063 | 25,883 ms | 7.5 ms | 270 | 625,438 |
| 1,000 (16,000) | 2.1 | 2,481 | 67,199 ms | 6.1 ms | 469 | 995,150 |
max_num_seqs at full memory, rate 60 (overload): 16 → 2,694 tok/s, TPOT 5.8 ms, TTFT p50 38.9 s · 64 → 6,407, 9.1 ms, 9.7 s · 128 → 8,056, 14.0 ms, 4.7 s · 256 → 8,784, 24.7 ms, 1.9 s · 512 → 8,980, 37.2 ms, 0.52 s.

**E5 · admission policy** (rate 55 = just past capacity, max_num_seqs 64)
| policy | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | worst E2E | out tok/s |
|---|---|---|---|---|---|---|
| FCFS | 7,490 ms | 16,611 ms | 10.16 s | 21.68 s | 22.40 s | 6,477 |
| SJF, oracle lengths | 97.2 ms | 27,650 ms | 2.01 s | 32.29 s | 37.34 s | 6,517 |
| SJF, 50% noisy predictor | 93.1 ms | 27,071 ms | 2.10 s | 32.44 s | 36.50 s | 6,583 |
| SJF, 100% noisy predictor | 109.1 ms | 26,970 ms | 2.24 s | 32.89 s | 37.74 s | 6,482 |
Priority classes (10% interactive = priority 0): FCFS gives interactive TTFT p50 7,395 ms / p99 16,449 ms; the priority policy gives interactive **36.1 ms / 121.1 ms**, while batch goes to 9,760 / 16,675 ms. SJF cuts the median ~77× (7,490 ÷ 97.2) but **starves long jobs** (worst E2E 22.4 → 37.3 s). The noise is lognormal σ on the predicted length.

**E6 · fairness** (tenant A 90% of traffic, B 10%, rate 60, max_num_seqs 64; "share" = share of output tokens in the first half of the run)
FCFS: A TTFT p50 9,279 ms, B 8,353 ms; share A 88% / B 12%. Fair (VTC-style counters, output tokens weighted 2×): A 10,974 ms, **B 46.0 ms (p99 181.4 ms)**; share A 79% / B 21%. The light tenant stops paying for the heavy tenant's burst.

**E7 · prefix caching** (4 shared system prompts of 2,048 tokens + a unique suffix, median 128; rate 20; 800 requests)
No cache: TTFT p50 3,344 ms, p99 9,482 ms, 3,733 tok/s, 1,779,902 prefill tokens computed. With block-hash prefix caching: **91.6%** hit rate, TTFT p50 **16.0 ms**, p99 35.2 ms, 4,830 tok/s, **149,694** prefill tokens computed (11.9× fewer).

**E8 · toy trace** (see labs/results.md for all 16 rows). Key moments: step 0 schedules `0:12P 1:4P` (request 2's 20-token prompt does not fit the 16-token budget alongside them); step 3 has 0 free blocks, so request 3 waits despite a free seat; **request 4 (prompt 30: its prompt alone takes 8 of the 12 blocks, and it peaks at 32 computed tokens = 8 blocks, since the last output token is never fed back) is admitted, then preempted at steps 6 and 8, then readmitted: thrashing**; request 6 is preempted once at step 12. 21 steps in total, 3 preemptions. TTFT/finish (ms): 0: 4.8/19.2 · 1: 9.6/48.0 · 2: 14.4/19.2 · 3: 23.0/48.0 · 4: 50.7/62.3 · 5: 53.5/91.1 · 6: 65.9/91.1 · 7: 61.9/100.7.

**E9 · scheduler CPU cost** (Python, M3, busy machine, ±50%): about **0.3–0.6 ms per step()** with 100 running requests and **0.85–1.2 ms** with 256 running, almost independent of waiting-queue size (100 vs 10,000). Against our model's GPU steps at those batch sizes (12.8–25.3 ms) that is 2.3–4.5%; against a 5 ms batch-1 step it would be up to ~20%: why vLLM V1 and SGLang overlap CPU scheduling with GPU work.

**Local real measurement (reused from Course 5, llama-server Llama-3.2-3B Q4_K_M on this M3, busy machine):** C=1 → 28.9 tok/s total, 31.7 per user, TTFT 0.42 s; C=2 → 48.3 / 28.0 / 0.77 s; C=16 → 100.2 / 8.5 / 5.16 s. llama-bench: prefill 407.9 tok/s vs decode 28.8 tok/s (14×).

## Engine facts (from research-engines.md; cite the source file/commit given there)
- vLLM V1 default: block size 16, gpu_memory_utilization 0.92, chunked prefill and prefix caching on, FCFS. max_num_batched_tokens/max_num_seqs depend on the GPU: an H100 gets 8192/1024 on the API server and 16384/1024 offline; an A100 gets 2048/256 on the API server and 8192/256 offline.
- vLLM V1 schedule(): running first, then waiting; no admission in a step that preempted; victim = running[-1] (FCFS) or the worst (priority, arrival); preemption = recompute (free blocks, num_computed_tokens = 0, push to the front of waiting); the scheduler output is {req_id: num_tokens}; no prefill/decode phase flag.
- The free block queue is a doubly linked list with sentinels → O(1) remove-from-middle on a cache hit; freed hashed blocks stay cached (the free queue doubles as the LRU); blocks are freed tail-first.
- SGLang: prefill-first loop; policies lpm/dfs-weight/fcfs/lof/random/…; new_token_ratio reservation (starts 0.7); retraction; radix tree with lock_ref; overlap scheduler on by default.
- TGI: FIFO, reserves prompt + max_new_tokens at admission → never preempts. TensorRT-LLM: GUARANTEED_NO_EVICT (default, reserve to completion) vs MAX_UTILIZATION (optimistic, pauses the last-started request).
- Triton dynamic batcher: preferred_batch_size, max_queue_delay_microseconds (default 0).

## Terms (use these names; mention aliases once)
prefill (prompt processing) · decode (generation) · forward step / iteration · token budget (max_num_batched_tokens) · max_num_seqs · static (request-level) batching · dynamic (time-window) batching · continuous batching (iteration-level scheduling, in-flight batching) · chunked prefill (stall-free batching) · generation stall · KV cache · paged KV / block · block table · free list · ref count · prefix caching (automatic prefix caching, RadixAttention) · admission · watermark · preemption (recompute vs swap) · thrashing · FCFS · SJF · priority · fairness (VTC) · starvation / aging · TTFT · TPOT (TBT, ITL) · E2E latency · throughput · goodput · SLO · P50/P99 · backpressure (HTTP 429) · disaggregated prefill/decode (P/D) · KV-aware routing · autoscaling.

## Colour meanings (same on every card)
- green `--accent`: useful compute / tokens that are kept / decode progress / the winner
- grey `--faint` / `--surface2`: waiting, idle, padding, stalls, overhead
- neutral `--text` / `--muted`: weights and data at rest
- red `--red`: ✗, SLO violations, stalls, preemption/thrash, the bottleneck, the worst value
- `--amber` only for prefill tokens when a diagram must distinguish prefill (amber) from decode (green). The shell has no amber variable, so add `--amber:#a8781c; --amber-bg:#f8efd9` (light) and `--amber:#d6a64a; --amber-bg:#2e2718` (dark), as in perf-5-engines/sarathi-stalls.html
- Worked request rule: a prefill pass emits the first token, so a 1,000-in / 200-out request = 1 prefill + 199 decode steps ≈ 16.3 + 199 × 4.84 = **979 ms** (token-loop.html)

## Card list (course folder: llm-request-scheduler)
| File | Title | Group |
|---|---|---|
| _overview.html | The scheduler on one page | 0 · Start here (main agent, last) |
| token-loop.html | The token loop: prefill, then decode | 1 · Why serving is hard |
| kv-cache.html | The KV cache: what it stores and why generation needs it | 1 |
| bottlenecks.html | The bottlenecks: bandwidth, compute, memory, and the CPU | 1 |
| metrics-slos.html | Measuring a server: TTFT, TPOT, throughput, goodput | 1 |
| why-batch.html | Why batching works: one weight read, many tokens | 2 · Batching strategies |
| static-batching.html | Static batching and its padding tax | 2 |
| dynamic-batching.html | Dynamic batching: wait a little, batch a lot | 2 |
| continuous-batching.html | Continuous batching: join and leave every step | 2 |
| chunked-prefill.html | Chunked prefill and the token budget | 2 |
| problem-spec.html | The build-it problem: interface and constraints | 3 · Building the scheduler |
| request-state.html | Request lifecycle and the data structures | 3 |
| step-algorithm.html | step(), line by line | 3 |
| block-manager.html | A paged KV block manager | 3 |
| admission-preemption.html | Admission, preemption and thrashing | 3 |
| policies.html | Who goes next: FCFS, SJF, priority, fairness | 3 |
| concurrency.html | Handling concurrent requests | 3 |
| prefix-caching.html | Prefix caching: never prefill the same tokens twice | 3 |
| reference-solution.html | The reference solution, end to end | 3 |
| throughput-levers.html | Every way to raise throughput, ranked | 4 · Increasing throughput |
| disaggregation.html | Splitting prefill and decode | 4 |
| routing-autoscaling.html | Many replicas: routing, admission and autoscaling | 5 · The full system |
| full-system.html | The whole system, drawn | 5 |
| real-systems.html | What real companies built | 5 |
| explain-it-back.html | Explain it back: the three big questions | 5 |
