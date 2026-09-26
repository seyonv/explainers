# Labs: Let's build GPT (z2h-8-gpt)

Runnable companions to the cards of this course, for Lecture 7 of Karpathy's Neural Networks: Zero to Hero ([video](https://www.youtube.com/watch?v=kCc8FmEb1nY), [ng-video-lecture](https://github.com/karpathy/ng-video-lecture)). Each lab prints the numbers its card shows.

## Setup

```bash
uv venv
uv pip install torch tiktoken
```

`tiktoken` is optional; two labs use it to count GPT-2 tokens and skip that line without it. The first lab you run downloads tiny shakespeare (`input.txt`, 1.1 MB) into `labs/data/`. `common.py` holds the shared pieces: the dataset and char tokenizer, `get_batch`, `estimate_loss`, the training loop, and gpt.py's modules (`Head`, `MultiHeadAttention`, `FeedFoward`, `Block`, `GPTLanguageModel`).

Small models run on the CPU, which is faster than MPS at this size. Only `layernorm-dropout-scale.py --gpt` uses the GPU (`cuda` if present, else `mps`).

## Run order

Run each lab from the course folder, e.g. `python labs/shakespeare-lm.py`. There is one lab per card, in card order. Times were measured on the M3 while other jobs were running, so treat them as upper bounds.

| # | Lab | What it prints | Time on the M3 |
|---|---|---|---|
| 1 | `shakespeare-lm.py` | 1,115,394 characters; the 65-character vocabulary | ~3 s |
| 2 | `char-tokenizer-split.py` | `encode("hii there")`; the 1,003,854 / 111,540 split; GPT-2 token count | ~5 s |
| 3 | `batches-of-chunks.py` | the "when input is … the target: …" table; a 4×8 batch = 32 examples | ~3 s |
| 4 | `bigram-baseline.py` | untrained loss 4.8786 vs −ln(1/65) = 4.1744; 100 chars of garbage | ~2 s |
| 5 | `train-and-script.py` | the notebook loop (4.6563 after 100 steps, ~2.38 after 10k); then bigram.py (val 2.49) | ~35 s |
| 6 | `adam-optimizer.py` | 3 Adam steps by hand next to `torch.optim.Adam`, SGD and AdamW | ~5 s |
| 7 | `average-the-past.py` | `x[0]` and `xbow[0]` from the double loop | ~2 s |
| 8 | `tril-matmul-trick.py` | the 3×3 `a @ b` demo; `xbow2 = wei @ x`, `allclose` True | ~1 s |
| 9 | `masked-softmax.py` | `-inf` mask → softmax gives the same `wei`; `allclose` True | ~1 s |
| 10 | `embeddings-and-positions.py` | tok_emb + pos_emb shapes, param counts, the index error past block_size | ~4 s |
| 11 | `self-attention-head.py` | one head, head_size 16: `wei[0]` (last row 0.0210 … 0.2391) | ~1 s |
| 12 | `attention-notes.py` | encoder vs decoder `wei`; sets, no cross-batch talk, cross-attention shapes | ~1 s |
| 13 | `scaled-attention.py` | variances 1.0449 / 1.0700 / 1.0918; softmax vs softmax×8 (0.80 peak) | ~1 s |
| 14 | `heads-in-the-model.py` | trains one head (val ~2.41), then 4 heads × 8 (val ~2.29) | ~45 s |
| 15 | `feedforward-residual.py` | trains + feed-forward (~2.27), 3 blocks without residuals (~2.33), with residuals (~2.08) | ~3 min |
| 16 | `layernorm-dropout-scale.py` | LayerNorm by hand, dropout, param counts (209,729 and 10,788,929), + LayerNorm model (~2.06) | ~1 min |
| 16b | `layernorm-dropout-scale.py --colab` | the Colab's small config, 5000 iters on CPU (the laptop-friendly run): val ~1.83 | ~2.7 min |
| 16c | `layernorm-dropout-scale.py --gpt --quick` | gpt.py, 250 iters on MPS (1/20 of the run, cheaper evals): val ~2.39 and the time per step | ~9 min |
| 16d | `layernorm-dropout-scale.py --gpt` | gpt.py in full, 5000 iters on MPS (val ~1.48 on the A100 in the video) | about 2–2.5 h (not yet timed on an idle machine) |
| 17 | `decoder-nanogpt-chatgpt.py` | EX1: nanoGPT's batched `CausalSelfAttention` equals our 6 heads; GPT-3 scale ratios | ~2 s |
