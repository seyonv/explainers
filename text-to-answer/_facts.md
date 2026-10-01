# From text to answer: inside an LLM. Course facts

The reader is a software engineer who is new to machine learning. The course follows one input from text to generated answer in **data-flow order** (not build order like Zero to Hero), and names the paper behind each stop. Every card fills one stop on the route. Attention itself, MQA/GQA and FlashAttention are already well covered by other hub cards, so the course links to them instead of repeating them.

## The running example (use it on every card)

- Input text: `What is the capital of India?`
- Model: **Qwen2.5-0.5B** (base model, no chat tuning), float32, Hugging Face transformers, measured on an M3 Mac.
- Config: 24 layers, hidden size 896, 14 query heads, 2 KV heads (GQA), head dim 64, FFN 4,864 (SwiGLU), vocab 151,936 rows (151,665 real tokens), tied input/output embeddings, RoPE, 494,032,768 parameters.
- Token IDs: `[3838, 374, 279, 6722, 315, 6747, 30]` = `What`, ` is`, ` the`, ` capital`, ` of`, ` India`, `?` (29 characters → 7 tokens).
- Embedding table: 151,936 × 896 = 136,134,656 weights (27.6% of the model).
- FFN weights: 24 × 3 × 896 × 4,864 = 313,786,368 (63.5% of the model).
- Cosine similarity of raw embedding rows: India·China 0.449, India·Pakistan 0.443, Delhi·Mumbai 0.483, India·capital 0.009, India·banana 0.013, Delhi·banana 0.097, king·queen 0.635.
- First next-token step after the question: ` The` 0.277 (logit 16.52), ` India` 0.134, ` What` 0.119, ` How` 0.043.
- Greedy answer: ` The capital of India is New Delhi.<|endoftext|>`. The probability of each chosen token: 0.277, 0.964, 0.919, 0.994, 0.969, 0.698, 0.994, 0.680. Their product is ≈ 0.11.
- The step after `…? The capital of India is`: ` New` 0.698 (logit 17.28), ` Delhi` 0.164, ` known` 0.018, ` Mumbai` 0.015, ` not` 0.013, ` __` 0.0095, ` the` 0.0094, ` called` 0.0087. At T = 0.7: ` New` 0.869, ` Delhi` 0.110. Top-p 0.8 keeps 2 tokens, 0.9 keeps 5 and 0.95 keeps 12. Beam search with 4 beams gives the same answer as greedy.
- Pure sampling (T = 1, no cut) with seeds 0 to 3 gave: ` Which is the capital of Kenya?…`, `:\nIndia\n\nThis question is about…`, ` The capital of India is not Delhi but…`, ` Options: - (A) Rajasthan…`.
- KV cache per token (bf16): 24 × 2 KV heads × 64 × 2 (K and V) × 2 bytes = 12,288 B = 12 KB.
- Attention sink: layers 6, 12 and 18 each send 35–63% of the last token's attention to the first token, `What`.

Scripts that produced these numbers: `tasks/tta-kit/facts/trace-run1.py` and `trace-run2.py`. New measurements go in `text-to-answer/labs/<slug>.py`, which imports `labs/common.py`. Run them with `HF_HUB_OFFLINE=1 ~/repos/nanoRL/.venv/bin/python` (torch and transformers are installed there, and the model is cached).

## Cards in this course (`text-to-answer/<slug>`)

| Slug | Title (claim) | Stop | One line |
|---|---|---|---|
| `_overview` | Ten stops from text to answer | all | the map of the route and the paper per stop |
| `the-whole-route` | How an LLM turns text into an answer | 1–10 | the spine: our prompt traced through every stop with measured shapes and numbers |
| `embedding-space` | Similar words end up close together | 3 | token ID → a row of a table; word2vec's idea; measured similarities and analogies in Qwen's table |
| `positions` | Attention can't tell first from last | 4 | order-blindness shown by measurement; sinusoids, learned positions, RoPE, ALiBi |
| `ffn-and-experts` | Most of the model is the feed-forward part | 6 | what the FFN does per token, where the weights live, dense → mixture of experts (Switch) |
| `decoding` | Choosing the next token is a choice | 8–9 | softmax, greedy, temperature, top-k, top-p, beam search, degeneration |
| `the-loop` | One token at a time | 10 | autoregressive generation, the probability of a whole answer, the KV cache, chat templates |
| `checkpoint-1` | Checkpoint | — | trace the route from memory |

## Existing hub cards to link (as `course/slug`)

- Tokens: `z2h-9-tokenizer/why-tokenization`, `z2h-9-tokenizer/bpe-by-hand`, `z2h-9-tokenizer/sentencepiece-llama`, `z2h-9-tokenizer/tiktokenizer-tour`, `z2h-9-tokenizer/quirks-explained`
- Embeddings: `z2h-8-gpt/embeddings-and-positions`, `z2h-4-mlp/bengio-embeddings`, `z2h-4-mlp/embedding-lookup`
- Attention: `z2h-8-gpt/attention-notes`, `z2h-8-gpt/self-attention-head`, `z2h-8-gpt/scaled-attention`, `perf-1-foundations/attention`, `perf-4-attention-kv/gqa`, `perf-4-attention-kv/flashattention-io`, `reasoning-models/modern-block` (RoPE, RMSNorm, SwiGLU, GQA)
- Block: `z2h-8-gpt/feedforward-residual`, `perf-1-foundations/transformer-forward`, `perf-6-scaling-out/moe-basics`, `llm-latency/moe`
- Output: `z2h-10-gpt2/load-and-forward`, `z2h-1-toolkit/softmax-and-cross-entropy`, `nucleus-sampling/nucleus-sampling`, `z2h-10-gpt2/sampling`, `paper-2107-03374/sampling-temperature`
- Speed: `perf-1-foundations/prefill-vs-decode-mac`, `llm-latency/_overview`

## Papers (the arXiv IDs come from the user's source list; use only these URLs)

BPE 1508.07909 · SentencePiece 1808.06226 · word2vec 1301.3781 · Distributed representations 1310.4546 · Attention Is All You Need 1706.03762 · RoFormer 2104.09864 · ALiBi 2108.12409 · MQA 1911.02150 · GQA 2305.13245 · FlashAttention 2205.14135 · FlashAttention-2 2307.08691 · Switch Transformers 2101.03961 · GPT-3 2005.14165 · Holtzman et al. (top-p) 1904.09751 · The Illustrated Transformer https://jalammar.github.io/illustrated-transformer/

## Sketch colour roles (use the same colours in the prose: `--blue`, `--green`, `--red`, `--yellow`, `--violet`)

- yellow: the input text and tokens
- blue: vectors and numbers (embeddings, hidden states, logits)
- violet: the model's machinery (attention, the FFN, the block)
- green: the chosen token, or "this one wins"
- red: failure, cut-off, or wrong output

## Naming

Never call the course or its cards "prompt …". The user wants that word avoided in titles, because it reads as prompt engineering. Writing "input text" or "the question" is fine; "prompt" inside the prose is fine when it's needed.
