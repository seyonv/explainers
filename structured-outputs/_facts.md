# Shared facts: structured outputs from language models

Every card writer reads this before writing. Reuse these values exactly. If a card needs a new shared number, it must have a source; add it to your returned numbers.

## Sources
- **The paper:** Geng, Cooper, Moskal, Jenkins, Berman, Ranchin, West, Horvitz, Nori. *Generating Structured Outputs from Language Models: Benchmark and Studies.* arXiv 2501.10868v1, 18 Jan 2025 (EPFL, Microsoft, JSON Schema org). https://arxiv.org/abs/2501.10868 · HTML: https://arxiv.org/html/2501.10868v1 · benchmark: https://github.com/guidance-ai/jsonschemabench
  - Full text saved at `/private/tmp/claude-501/-Users-seyonvasantharajan-Desktop-repos-explainers/659fc8d2-f29d-4f50-b1d7-a7ff20d0702a/scratchpad/paper.txt`.
- **The cookbook:** Nanonets, *Structured LLM Outputs* handbook. https://nanonets.com/cookbooks/structured-llm-outputs/ (subpages listed under "Cookbook pages" below).
  - Full text saved at `/private/tmp/claude-501/-Users-seyonvasantharajan-Desktop-repos-explainers/659fc8d2-f29d-4f50-b1d7-a7ff20d0702a/scratchpad/nanonets-cookbook.txt`, with one file per page in `/private/tmp/claude-501/-Users-seyonvasantharajan-Desktop-repos-explainers/659fc8d2-f29d-4f50-b1d7-a7ff20d0702a/scratchpad/nano/`.

## The reader
- Setup: Apple M3, 24 GB, Ollama 0.34.2, qwen3.5:4b (4.7B params, Q4_K_M, **248,320-token vocabulary**, from `/api/show`). Also uses hosted APIs (Claude, OpenAI, Gemini) and open models.
- Already knows: the llm-latency course (TTFT, output speed, output length). Knows JSON and Pydantic at a working level.
- Wants: to understand *how* structured output actually works under the hood, when to trust it, what it costs, and which tool to pick.

## The running example
The cookbook's store chatbot (adapted: the customer block is dropped).
- The message is *"Hi, I want to order three bottles of the citrus body wash and one lavender candle."*
- The price list has 10 SKUs: BW-CITRUS $12, BW-EUCALYPT $12, CANDLE-LAVENDER $18, CANDLE-SANDAL $20, LOTION-SHEA $15, SOAP-OATMEAL $8, DIFF-BERGAMOT $45, SCRUB-SUGAR $22, BALM-PEPPER $10, MIST-ROSE $24.
- The correct answer is `{"items":[{"sku":"BW-CITRUS","qty":3},{"sku":"CANDLE-LAVENDER","qty":1}],"total_usd":54}`, because 3 × $12 + 1 × $18 = $54.
- The schema is an object with `items` and `total_usd` (number), both required, `additionalProperties:false`. `items` is an array with `minItems:1` whose entries are objects with `sku` (enum of the 10 SKUs) and `qty` (integer, minimum 1), both required, `additionalProperties:false`.
- The unconstrained prompt is the cookbook's: listings, then "Extract the order details from the message and return a JSON object matching the schema below. Output only the valid JSON.", a template, and the message.

## Local measurements (M3, Ollama 0.34.2, qwen3.5:4b, think:false, 2026-09-21)
Command: `/private/tmp/claude-501/-Users-seyonvasantharajan-Desktop-repos-explainers/659fc8d2-f29d-4f50-b1d7-a7ff20d0702a/scratchpad/measure.py`, which posts to Ollama `/api/chat`. The unconstrained run sends the prompt only. The constrained run sends the same prompt plus `"format": <the JSON schema>`; Ollama turns the schema into a llama.cpp GBNF grammar. Runs used temperature 0.7 with seeds 1000–1029, 30 runs per arm. The validator is Python `jsonschema` 4.26.0.

| What | Unconstrained (prompt only) | Constrained (`format` = schema) |
|---|---|---|
| `json.loads` works on raw text | **7 / 30** (23%) | **30 / 30** |
| Wrapped in ```` ```json ```` fences | 23 / 30 | 0 / 30 |
| Parses after cleaning (first `{` to last `}`) | 30 / 30 | 30 / 30 |
| Valid against the schema (after cleaning) | 29 / 30 (one was a bare array with extra `unit_price`/`total_price` keys) | 30 / 30 |
| Values fully correct (right SKUs, qty, total = 54) | **15 / 30** (50%) | **22 / 30** (73%) |
| Wrong totals seen | 60 ×10, 66 ×2, 78 ×2 | 60 ×5, 58, 62, 66 |
| Output tokens, median | 80 (pretty-printed + fences) | 38 (compact JSON) |
| End-to-end time, median / p90 / max | 5.53 s / 7.10 s / 9.05 s | 2.72 s / 4.64 s / 6.03 s |
| Output speed, median | 14.86 tokens/s | 17.47 tokens/s |
| Prompt | 301 tokens, prompt eval median 0.22 s | 0.25 s |

Read these carefully:
- The constrained run was about 2× faster end to end mainly because its answers were **shorter** (38 vs 80 tokens). It is not mainly a faster per-token speed.
- n = 30 per arm, so 15 vs 22 correct is suggestive, not proof.
- The wrong totals are arithmetic errors (60 ≠ 54). The grammar cannot catch them, because they are schema-valid numbers.

**Grammar setup overhead** (prompt + 1 output token, temp 0, 5 reps, medians):
- no format: 0.164 s
- order schema: 0.206 s (+0.04 s)
- a 200-field schema with an 8-value enum per field: 0.237 s (+0.07 s)

(The first "no format" rep was 1.67 s cold, which is excluded as the warm-up.) Ollama/llama.cpp compiles grammars lazily, so the set-up cost here is tens of milliseconds, not seconds.

**Token cost of the same answer** (Qwen tokenizer, measured with `prompt_eval_count` delta):

| Format | Tokens | Characters |
|---|---|---|
| Pretty JSON (indent 2) | 72 | 152 |
| Minified JSON | 35 | 88 |
| Compact pipe format (`BW-CITRUS\|3` / `CANDLE-LAVENDER\|1` / `54`) | 18 | 32 |
| The JSON schema itself, pretty-printed | 286 | 911 |

**Reasoning-field order test:** the cookbook's lead-qualification reply ("impressive… big fan… However, budget freeze until Q4, cannot buy"). It ran 20 runs per variant at temp 0.7, all constrained. The results:
- `{status}` only: 20/20 not_interested, median 14 output tokens
- `{status, reasoning}`: 20/20, median 49 tokens
- `{reasoning, status}`: 20/20, median 54.5 tokens

On this model and this example, **field order made no difference to correctness**; it only cost tokens. The cookbook's story that the answer-first version would pick "interested" did not reproduce with qwen3.5:4b.

## Sourced numbers: the paper (always say "paper, A100, Llama-3.1-8B" or "paper, Llama-3.2-1B")
- **Dataset (Table 1):**

  | Dataset | Schemas | Category | Difficulty |
  |---|---|---|---|
  | GlaiveAI-2K | 1,707 | function call | easy |
  | Github-Trivial | 444 | misc | trivial |
  | Github-Easy | 1,943 | misc | easy |
  | Snowplow | 403 | operational API | medium |
  | Github-Medium | 1,976 | misc | medium |
  | Kubernetes | 1,064 | API config | medium-hard |
  | Washington Post | 125 | resource access API | medium-hard |
  | Github-Hard | 1,240 | misc | hard |
  | JSONSchemaStore | 492 | misc | hard |
  | Github-Ultra | 164 | misc | ultra-hard |

  The total is 9,558, called "10K" in the paper. Trivial and Ultra are excluded from the experiments.
- **Engines:** Guidance 0.2.0rc, Outlines 0.1.8, Llamacpp 0.3.2, XGrammar 0.1.6, OpenAI (version unknown), Gemini 0.8.3.
- **Efficiency setup:** Llama-3.1-8B-Instruct, Q8, 128K vocabulary, one A100-SXM4-80GB, batch 1, medians. Metrics are computed on the intersection of schemas every engine covers. Grammar caching doesn't help because every schema is unique; prefix caching is on.
- **Table 2 (llama.cpp backend)**, GCT s / TTFT s / TPOT ms:
  - GlaiveAI: LM-only NA / 0.10 / 15.40 · Guidance 0.00 / 0.24 / 6.37 · Llamacpp 0.05 / 0.20 / 29.98 · Outlines 3.48 / 3.65 / 30.33
  - GitHub Easy: LM 0.10 / 15.83 · Guidance 0.00 / 0.34 / 7.44 · Llamacpp 0.05 / 0.18 / 27.22 · Outlines 3.71 / 3.97 / 39.78
  - Snowplow: LM 0.11 / 16.23 · Guidance 0.00 / 0.28 / 6.55 · Llamacpp 0.05 / 0.20 / 28.90 · Outlines 3.91 / 4.14 / 42.66
  - GitHub Medium: LM 0.20 / 16.68 · Guidance 0.01 / 0.54 / 7.57 · Llamacpp 0.06 / 0.30 / 29.08 · Outlines 8.05 / 8.38 / 46.57
  - Kubernetes: LM 0.16 / 15.32 · Guidance 0.01 / 0.45 / 9.47 · Llamacpp 0.05 / 0.28 / 28.04 · Outlines 5.29 / 5.55 / 46.10
  - Table 13, Washington Post: LM 0.14 / 16.54 · Guidance 0.00 / 0.33 / 8.58 · Llamacpp 0.06 / 0.24 / 27.20 · Outlines 3.06 / 3.30 / 40.12
  - Table 13, GitHub Hard: LM 0.51 / 16.18 · Guidance 0.02 / 1.18 / 8.36 · Llamacpp 0.05 / 0.60 / 28.17 · Outlines 12.77 / 13.30 / 81.22
- **Table 3 (HF Transformers backend)**, Guidance vs XGrammar:
  - GlaiveAI: Guidance 0.01 / 0.36 / 36.92, XGrammar 0.12 / 0.30 / 66.78
  - GitHub Easy: Guidance 0.01 / 0.37 / 42.03, XGrammar 0.11 / 0.33 / 65.57
  - GitHub Medium: Guidance 0.01 / 0.55 / 44.21, XGrammar 0.20 / 0.48 / 65.51
  - GitHub Hard: Guidance 0.01 / 0.73 / 35.88, XGrammar 0.30 / 0.65 / 65.20

  XGrammar runs compilation concurrently with prefill. Guidance's TPOT beats LM-only because of "guidance acceleration": it skips generation steps when the next tokens are forced.
- **The "50% faster" headline** is Guidance's 6.37 ms vs LM-only 15.40 ms TPOT on GlaiveAI, i.e. 59% less time per token, which the paper calls "speed up … by 50%".
- **Coverage setup:** Llama-3.2-1B-Instruct, a 2-shot prompt, greedy decoding, one run per schema, a 40 s compile timeout plus a 40 s generation timeout, and `jsonschema` Draft 2020-12 with format checks.
- **Table 4 (declared / empirical / compliance):**
  - GlaiveAI: LM-only 1.00/0.90/0.90 · Guidance 0.98/0.96/0.98 · Llamacpp 0.98/0.95/0.97 · Outlines 0.99/0.95/0.96 · XGrammar 1.00/0.93/0.93 · OpenAI 0.89/0.89/1.00 · Gemini 0.86/0.86/1.00
  - GitHub Easy: LM 1.00/0.65/0.65 · Guidance 0.90/0.86/0.96 · Llamacpp 0.85/0.75/0.88 · Outlines 0.86/0.59/0.83 · XGrammar 0.91/0.79/0.87 · OpenAI 0.30/0.29/0.97 · Gemini 0.08/0.07/0.88
  - Snowplow: LM 1.00/0.46/0.46 · Guidance 0.87/0.82/0.94 · Llamacpp 0.92/0.74/0.81 · Outlines 0.95/0.36/0.61 · XGrammar NA · OpenAI 0.21/0.21/1.00
  - GitHub Medium: LM 1.00/0.38/0.38 · Guidance 0.79/0.69/0.87 · Llamacpp 0.77/0.57/0.74 · Outlines 0.72/0.29/0.40 · XGrammar 0.79/0.52/0.66 · OpenAI 0.13/0.12/0.92
  - Kubernetes: LM 1.00/0.56/0.56 · Guidance 0.98/0.91/0.92 · Llamacpp 0.98/0.76/0.78 · Outlines 0.98/0.57/0.58 · XGrammar 0.12/0.07/0.58 · OpenAI 0.21/0.21/1.00
  - Washington Post: LM 1.00/0.40/0.40 · Guidance 0.86/0.86/1.00 · Llamacpp 0.97/0.94/0.97 · Outlines 0.97/0.22/0.23 · XGrammar 0.85/0.64/0.75 · OpenAI 0.13/0.13/1.00
  - GitHub Hard: LM 1.00/0.13/0.13 · Guidance 0.60/0.41/0.69 · Llamacpp 0.61/0.39/0.63 · Outlines 0.47/0.03/0.06 · XGrammar 0.69/0.28/0.41 · OpenAI 0.09/0.09/1.00
  - JSONSchemaStore: LM 1.00/0.21/0.21 · Guidance 0.35/0.30/0.88 · Llamacpp 0.54/0.38/0.69 · Outlines 0.38/0.09/0.24 · XGrammar 0.76/0.33/0.43 · OpenAI 0.06/0.06/1.00

  Outlines' low compliance comes mostly from generation **timeouts**: `minItems`, `maxItems`, `enum` and arrays "often take 40 seconds to 10 minutes" to process. The closed APIs take a "conservative strategy" (they support fewer features, but those reliably). Closed APIs use their own models, so they are not directly comparable. LM-only benefits from Llama 3.1 being fine-tuned for JSON schemas.
- **The JSON Schema Test Suite** has 45 categories (one per keyword or keyword group). To pass, an engine must generate every valid instance and block every invalid one.
  - **Table 5** (number of categories), in the order Outlines / Llamacpp / XGrammar / Guidance:
    - >0%: 20 / 21 / 28 / 30
    - >25%: 11 / 11 / 16 / 25
    - >50%: 2 / 5 / 3 / 21
    - >75%: 0 / 2 / 1 / 17
    - 100%: 0 / 1 / 1 / 13
    - tied highest: 4 / 6 / 14 / 25
    - single highest: 1 / 0 / 10 / 19
  - **Table 6** (categories with at least one failure of each type):
    - compile error: 42 / 37 / 3 / 25
    - over-constrained: 16 / 18 / 5 / 7
    - under-constrained: 8 / 7 / 38 / 1
  - Definitions: **over-constrained** means it rejects valid instances; **under-constrained** means it allows invalid ones.
- **Quality (Table 8)**, Llama-3.1-8B-Instruct, output `{"reasoning":…, "answer":…}`, prompts from dottxt's "Say What You Mean" rebuttal. Values are LM-only / XGrammar / Llamacpp / Outlines / Guidance:
  - Last Letter: 50.7 / 51.2 / 52.0 / 53.3 / 54.0 %
  - Shuffle Objects: 52.6 / 52.7 / 52.6 / 53.0 / 55.9 %
  - GSM8K: 80.1 / 83.7 / 82.4 / 81.6 / 83.8 %

  The paper attributes Guidance's edge to token healing ("we believe"). Its toy example: the model wants `89,000`, the grammar blocks `,`, and the model drifts to `890000`. No engine wins on every instance (Figure 16).
- **Algorithm 1** (the constrained decoding loop): C.update(o) → m = C.mask() → v = f(x+o) logits → v′ = m ⊙ v → t = decode(v′) → stop if EOS, else append. The mask can run in parallel with the forward pass, and compilation can overlap prefill.
- **Limitations the paper states:**
  - empirical coverage is top-1 only
  - `jsonschema` validation is not exhaustive
  - open-source runs use Llama-3.2-1B while the closed APIs use their own models
  - the test suite excludes `format` and remote `$ref`
  - efficiency depends on the model and its vocabulary
  - quality tasks have simple output structures

## Sourced numbers: the cookbook (cite the page)
- **Why structure fails:** the chatbot "breaks for 20% of" production messages; the failure modes are a "Sure! Here is…" preamble and chatter. [basic-concepts/the-problem]
- **Token masking:** step 1 of the illustrative distribution is Sure 45%, Here 25%, `{` 5%, `{"` 4%, `{"customer"` 3%, `{"cust` 3%, `{'cust` 3%, `{"id":` 2%, `{"order":` 1%. After the mask, `{` is 33%, `{"` 27%, `{"customer"` 20% and `{"cust` 20%; the rest are 0. Vocabularies are ">32k tokens"; the mask runs on the CPU while the model runs on the GPU. [basic-concepts/constrained-method]
- **Regex → FSM:** the example is `\{"name":("John"|"Paul"),"age":(20|30)\}`, turned into a character FSM and then a token FSM. The first valid tokens are `{` and `{"`. Recursive schemas (Wikipedia first-link chains, social graphs, SQL subqueries, matching HTML tags) need infinite states. [constrained-decoding/schema-input/regex]
- **CFG → PDA:** `root ::= expr; expr ::= number | "(" expr op expr ")"; op ::= "+"|"-"|"*"|"/"; number ::= [0-9]+`, with `(3 + (5 * 2))` as the example output. A PDA is an FSM plus a stack, e.g. the `(())` push-push-pop-pop example. [schema-input/context-free-grammar]
- **Outlines-core:**
  - precomputes the whole FSM, so each step is an O(1) lookup
  - deduplicates deterministic paths ("7 paths from State 2 → State 6, all generate `name`"; the condensed FSM needs "only 2 LLM calls")
  - has no PDA support, so no recursion
  - pays TTFT on every new schema [backends/outlines-core]
- **llguidance:**
  - an Earley parser (simulates a non-deterministic PDA) that computes masks on the fly, so there is no precompute
  - a token trie; its example vocabulary is "a", "ab", "an", "and", "ant", "1", "10", "103", "108", "1e", "1e1", "1e2", and an integer constraint gives 1, 10, 103, 108
  - pruning invalid UTF-8 "discards 30-50% search space" in microseconds
  - a lexer (regex) / parser (CFG) split, so "the Earley parser needs to be invoked in under 0.5% of trie nodes"
  - handles ambiguous `anyOf` by keeping both paths alive [backends/llguidance]
- **XGrammar:**
  - converts schemas to a PDA and splits the vocabulary into context-independent and context-dependent tokens; "less than 1% of tokens are context-dependent"
  - precomputes a lookup table for the >99% and checks the stack at runtime for the rest
  - masks take "under ~20μs", but some edge schemas take "tens or hundreds of milliseconds" [backends/xgrammar]
- **LM Format Enforcer:**
  - a character-level parser plus a token trie
  - allows any field order and whitespace
  - `output_scores=True` shows the forced token vs the model's leading token
  - its Python per-token latency is high [backends/lm-format-enforcer]
- **Engines:**
  - vLLM and SGLang support XGrammar, llguidance and Outlines-core. SGLang is faster on most structured benchmarks, except at low cache-hit rates.
  - TGI uses Outlines-core.
  - llama.cpp uses GBNF; Ollama converts JSON schemas to GBNF.
  - MLC-LLM/WebLLM use XGrammar on devices.
  - TensorRT-LLM integrated XGrammar and claims 20–40% more throughput than vLLM/SGLang on a fixed model.
  - MAX uses llguidance. [constrained-decoding/inference-engines]
- **Commercial APIs** (OpenAI, Gemini, Anthropic) do constrained decoding server-side. You get no logits, schema support is limited, and they return a "Refusal" on safety triggers. [constrained-decoding/commercial-providers]
- **Unconstrained route** [basic-concepts/unconstrained-method]:
  - a system role, an explicit schema, few-shot examples, and priming with "JSON Output:"
  - `clean_text` (strip fences, keep the first `{` to the last `}`)
  - `json.loads`, falling back to `ast.literal_eval`
  - an allow-list of keys plus type casts
  - a repair loop that feeds the error back, with 3 repair attempts × 3 fresh retries
- **Libraries:**
  - BAML: type definitions use "60% less tokens than JSON schemas"; a JSON-schema prompt of ~420 tokens becomes ~140 in BAML; "beat[s] OpenAI's structured outputs … in … BFCL". [unconstrained-decoding/baml]
  - Instructor and Pydantic AI prompt, then parse and retry. [unconstrained-decoding/*]
- **Compact formats**, from the incident.io example [how-to-optimize/compact-prompts]:
  - drop the reasoning field: output 315 → 170 tokens, 11 s → 7 s
  - compact input format: 15,000 → 2,000 tokens, 7 s → 5.7 s
  - pipe-delimited output: 170 → ~50 tokens, 5.7 s → 2.3 s
  - overall: 11 s → 2.3 s
  - "each UUID costs a whopping 24 tokens"; replacing UUIDs with ints gives ">20% accuracy jump"
- **Sampling** [how-to-optimize/sampling-method]:
  - use temperature 0 for extraction, and 0.2–0.7 only for prose fields
  - disable the repetition penalty, because keys repeat
  - top-p/top-k can prune the only valid token (like `}`), stalling older backends
  - min-p is safer
  - beam search costs VRAM and latency linearly with width
  - best-of-N plus a validator
- **Chain of thought** [how-to-optimize/chain-of-thought]: put a reasoning field *before* the answer; don't constrain the reasoning field.
- **Handling exceptions:** for "I want a McFlurry" on a resume schema, you still get a JSON object, but an exception would be right. [how-to-optimize/handle-exceptions]
- **TTFT:** constrained decoding adds "milliseconds to seconds" once per schema, then the cost is amortised; it only matters for dynamic schemas. [basic-concepts/choosing-the-right-method]

## Cookbook pages (base https://nanonets.com/cookbooks/structured-llm-outputs/)
- basic-concepts/sampling/
- basic-concepts/the-problem/
- basic-concepts/constrained-method/
- basic-concepts/unconstrained-method/
- basic-concepts/choosing-the-right-method/
- constrained-decoding/schema-input/regex/
- constrained-decoding/schema-input/context-free-grammar/
- constrained-decoding/schema-input/templates/
- constrained-decoding/backends/outlines-core/
- constrained-decoding/backends/llguidance/
- constrained-decoding/backends/xgrammar/
- constrained-decoding/backends/lm-format-enforcer/
- constrained-decoding/the-full-stack/
- constrained-decoding/inference-engines/
- constrained-decoding/libraries/
- constrained-decoding/models/
- constrained-decoding/commercial-providers/
- constrained-decoding/choosing-the-right-method/
- unconstrained-decoding/baml/
- unconstrained-decoding/instructor/
- unconstrained-decoding/pydantic-ai/
- unconstrained-decoding/other-libraries/
- how-to-optimize/chain-of-thought/
- how-to-optimize/compact-prompts/
- how-to-optimize/few-shot-prompting/
- how-to-optimize/sampling-method/
- how-to-optimize/handle-exceptions/

## Terms (the name each card uses)
- structured output
- constrained decoding (alias: grammar-constrained decoding, guided decoding, structured generation)
- unconstrained / prompt-only (alias: LM-only in the paper)
- token mask
- JSON Schema
- grammar engine (the paper uses "grammar engine" and "constrained decoding framework" interchangeably)
- backend (Outlines-core, llguidance, XGrammar, llama.cpp GBNF)
- inference engine (vLLM, SGLang, Ollama)
- regex / FSM (finite state machine)
- CFG (context-free grammar) / PDA (pushdown automaton)
- grammar compilation time (GCT)
- time to first token (TTFT)
- time per output token (TPOT)
- declared coverage, empirical coverage, true coverage, compliance rate
- over-constrained / under-constrained
- token healing
- schema-valid vs correct

## Colour meanings
- green `--accent`: valid tokens or output, the schema-compliant or correct result, the winner, the faster option
- grey `--faint` / `--surface2`: masked-out tokens, overhead, compile time, waiting, forced or skipped tokens
- red `--red`: ✗, invalid or broken output, schema violations, wrong values, timeouts, the worst value

The same thing always gets the same colour: masked tokens are always grey, allowed tokens always green, and a broken parse or wrong value always red.

## Card list
| File | Title | Group | Owner |
|---|---|---|---|
| `_overview.html` | Structured outputs: the whole picture | Start here | main agent |
| `_overview-choosing.html` | Choosing a method and a tool | Start here | main agent |
| `why-json-breaks.html` | Why asking for JSON isn't enough | 1 · The problem | subagent |
| `json-schema.html` | JSON Schema, the contract | 1 · The problem | subagent |
| `valid-not-correct.html` | Valid is not correct | 1 · The problem | subagent |
| `token-masking.html` | Token masking | 2 · How constraining works | subagent |
| `regex-fsm.html` | Regex and finite state machines | 2 · How constraining works | subagent |
| `cfg-pda.html` | Grammars and pushdown automata | 2 · How constraining works | subagent |
| `fast-masks.html` | Computing masks fast | 2 · How constraining works | subagent |
| `token-alignment.html` | Token boundaries vs grammar boundaries | 2 · How constraining works | subagent |
| `compile-vs-per-token.html` | Compile time vs per-token time | 3 · What it costs in time | subagent |
| `fast-forward.html` | Why constraining can be faster | 3 · What it costs in time | subagent |
| `quality-shift.html` | Does constraining hurt quality? | 4 · What it does to quality | subagent |
| `sampling-with-grammar.html` | Sampling settings under a grammar | 4 · What it does to quality | subagent |
| `coverage-compliance.html` | Coverage and compliance | 5 · Judging an engine | subagent |
| `over-under-constrained.html` | Over- and under-constrained | 5 · Judging an engine | subagent |
| `repair-loop.html` | The prompt-and-repair loop | 6 · In production | subagent |
| `where-it-runs.html` | Where constrained decoding runs | 6 · In production | subagent |
| `compact-formats.html` | Compact formats | 6 · In production | subagent |

## Added after the writers ran (from their light local checks, 2026-09-21)
- **qwen3.5:4b Modelfile defaults** are temperature 1, top_k 20, top_p 0.95 and presence_penalty 1.5. The 30-run tests overrode only the temperature (0.7), so the penalties were on. With presence_penalty 0, the constrained answer at temperature 0 came out pretty-printed at 61 tokens instead of 38. [sampling-with-grammar]
- **Last Letter field order** (Ian Peter Bernard Stephen → nrdn, 10 seeds, temp 0.7): `{answer}` 0/10 · `{answer, reasoning}` 0/10 · `{reasoning, answer}` 8/10. [quality-shift, qs_lastletter.py]
- **JSON mode** (`format:"json"`): with the schema template in the prompt, 10/10 parse, 10/10 valid and 5/10 correct. Without the template, 10/10 parse and 0/10 schema-valid, with 7 different key sets. [why-json-breaks, where-it-runs]
- **Ollama accepts but doesn't enforce** `maximum` on numbers, `format:email` and `multipleOf`. It does enforce `enum`, `minItems`, integer `minimum` (which warps the answer: qty 3 → 30), and the schema's key order. [json-schema, coverage-compliance, over-under-constrained]
- **Repair loop on the 15 bad prompt-only seeds**, with a value check: 11/15 fixed within 3 calls and 4/15 stuck. [repair-loop]
- **Token-alignment probe** (integer schema vs "89,000"): at temp 0.7, 9/10 gave 89000 and 1/10 drifted to 64500. The model wanted `,` at 85.6% and was forced to `0` at 11.8%. [token-alignment]
