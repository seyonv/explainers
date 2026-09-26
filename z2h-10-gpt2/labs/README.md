# Labs: Reproducing GPT-2 (124M) (z2h-10-gpt2)

Runnable companions to Lecture 9 of Karpathy's Neural Networks: Zero to Hero
([video](https://www.youtube.com/watch?v=l8pRSuU81PU),
[code](https://github.com/karpathy/build-nanogpt)). There is one lab per card. They run on `mps` when it is
available (the device is picked cuda → mps → cpu, as in chapter 8), and each lab rebuilds everything it needs.

## Setup

```bash
uv venv
uv pip install torch numpy matplotlib tiktoken transformers datasets
```

Run every lab from the course folder (`z2h-10-gpt2/`), e.g. `python labs/the-module.py`.

- `common.py` holds build-nanogpt's `GPT` module and `from_pretrained`, the device pick, tiny shakespeare with
  `DataLoaderLite`, and the HellaSwag helpers.
- The first run downloads OpenAI's `gpt2` weights from Hugging Face (~550 MB, cached in `~/.cache/huggingface`)
  and `input.txt` into `labs/data/`, which is gitignored.
- HellaSwag's GitHub source is DMCA-blocked (HTTP 451), so the labs read the same validation split from
  Hugging Face (`Rowan/hellaswag`).
- Anything that needs an NVIDIA GPU (the A100 speed ladder, fused AdamW, DDP) prints a `[skipped]` line instead
  of crashing.

## Run order (times measured on an M3, 24 GB, while other jobs were running)

| # | Lab | What it prints | Time |
|---|---|---|---|
| 1 | `labs/the-target.py` | The 149 checkpoint tensors (wte 50257×768, wpe 1024×768), 124,439,808 params, the wpe stats, and 5 HF pipeline samples. `--plot` saves the wpe image; `--all` lists every tensor. | ~5 s (first run downloads gpt2) |
| 2 | `labs/the-module.py` | `GPT(GPTConfig())` = 124,439,808 params (50304 vocab: 124,475,904), where they live, the module tree, and GELU tanh vs exact. | ~8 s |
| 3 | `labs/load-and-forward.py` | The Conv1D transpose, the shapes (1,8) → (1,8,768) → (1,8,50257), the top-5 next tokens, and a max diff vs HF logits of 4.6e-05. | ~11 s |
| 4 | `labs/sampling.py` | 5 top-k-50 samples from GPT-2 on mps (seed 42), then 5 from a random init. | ~10-19 s |
| 5 | `labs/first-loss.py` | The 4×6 play.ipynb batch, the (B·T, V) flatten, and the init loss 10.96 vs −ln(1/50257) = 10.8249. | ~4 s |
| 6 | `labs/overfit-then-load.py` | 50 AdamW steps at lr 3e-4, B=4, T=32: one batch (→ 0.36), then DataLoaderLite (→ 6.80). | ~40 s |
| 7 | `labs/weight-tying.py` | The same `data_ptr()` in the checkpoint, 38,597,376 params saved (31.0%), and the nearest rows in wte. | ~11 s |
| 8 | `labs/init.py` | 1/√d for GPT-2 widths, the std-growth demo (10.09 vs 1.00), the scaled c_proj std 0.00408, and the residual-stream std per block. | ~10 s |
| 9 | `labs/precision.py` | fp32/fp16/bf16 ranges, what autocast does on MPS, and fp32 vs bf16 step time (default B=4, T=256; `--T 1024` for the bigger case). | ~26 s (`--T 1024`: ~75 s) |
| 10 | `labs/compile-flash.py` | Manual attention vs SDPA (same output, 2.7-6.3× faster op), whole-model fwd+bwd, and `torch.compile` on MPS. | ~38 s |
| 11 | `labs/nice-numbers.py` | The factors of 50257 and 50304, and the lm_head matmul timed at both vocab sizes on MPS. `--cpu` also times the CPU (slow). | ~20 s |
| 12 | `labs/optimizer-recipe.py` | `get_lr` at key steps, 50 decayed / 98 non-decayed tensors, fused-AdamW availability, and clipping on real gradients. `--plot` saves the LR curve. | ~9 s |
| 13 | `labs/big-batch.py` | play.ipynb's accumulation demo (the /4 fix matches), grad_accum_steps for several setups, and a tiny accumulation run. DDP is skipped without CUDA. | ~23 s |
| 14 | `labs/data-and-evals.py` | Streams FineWeb-Edu into two 1M-token uint16 mini shards in `labs/data/edu_fineweb10B/`, then runs HellaSwag with HF gpt2 on 20 examples. `--hella 10042` runs the full eval. | ~11 s (full eval ~14-20 min) |
| 15 | `labs/train_gpt2.py --quick` | build-nanogpt's final training script, run for 50 steps at B=4, T=256 with 2,048 tokens per step. It includes val loss, HellaSwag (20 examples) and samples, and writes `labs/data/log/log.txt`. It needs lab 14's shards. | ~1.8 min |
| 16 | `labs/results.py` | Parses the log like play.ipynb and compares against GPT-2 (3.2924 val loss, 0.2945 HellaSwag) and GPT-3 (0.337). It also prints the 8×A100 vs M3 time for 10B tokens. `--plot` saves the two panels. | ~1 s |

The full-length runs are listed below. Don't run them on the laptop by accident.

- `python labs/data-and-evals.py --hella 10042` is the full HellaSwag eval of GPT-2 124M on MPS, about 14-20 minutes. hellaswag.py's reference result is acc_norm 0.2955.
- `torchrun --standalone --nproc_per_node=8 labs/train_gpt2.py` is the lecture's run: 19,073 steps of 2^19 tokens. It needs 8 CUDA GPUs and all 100 FineWeb-Edu shards from build-nanogpt's `fineweb.py`. It takes about 1-2 h on 8×A100. On an M3 it would take about 88 days.

Sampled text on this machine differs from the video's (different device and library versions). Where the lecture's numbers depend on an A100, the labs print the M3's own numbers.
