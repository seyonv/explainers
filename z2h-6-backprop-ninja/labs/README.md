# Labs: makemore 4, becoming a backprop ninja (z2h-6-backprop-ninja)

Runnable companions to Lecture 5 of Karpathy's Neural Networks: Zero to Hero ([video](https://www.youtube.com/watch?v=q8SA3rM6ckI), [notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb)). Each lab rebuilds what it needs from scratch and prints the numbers its card shows.

## Setup

```bash
uv venv
uv pip install torch numpy matplotlib
```

Run every lab from the course folder (`z2h-6-backprop-ninja/`), for example `python labs/chunked-forward-and-cmp.py`. The first run downloads `names.txt` into `labs/data/`. Everything runs on the CPU.

`labs/common.py` holds the shared pieces: the dataset and the 80/10/10 split, `cmp()`, the notebook's init (4,137 parameters) and the chunked forward pass with `retain_grad()` on every intermediate. Karpathy draws `bngain`/`bnbias` from the unseeded global RNG, so `common.py` seeds it with `torch.manual_seed(42)`. That's why the forward loss is 3.3444 here, not the notebook's 3.3377.

## Run order

Times are for an Apple M3 while other jobs were running, so an idle machine is faster.

| # | Lab | What it prints | Time |
|---|---|---|---|
| 1 | `labs/leaky-abstraction.py` | the "clip the loss" bug (outliers get zero gradient) vs Huber; a saturated tanh; a dead ReLU | 2 s |
| 2 | `labs/chunked-forward-and-cmp.py [--all]` | the split, 4,137 params, loss, the intermediate shapes, `cmp` on right, nearly right and wrong gradients; `--all` runs the whole Exercise 1 answer (26 of 26 exact) | 1 s |
| 3 | `labs/softmax-chain-backward.py` | dlogprobs to dnorm_logits, `cmp` after each; dcounts False with one branch and True with two; `**-1` vs `1.0/` | 2 s |
| 4 | `labs/sum-broadcast-duality.py` | the duality on a toy; dlogit_maxes of about 1e-9; the one_hot second branch of dlogits | 3 s |
| 5 | `labs/matmul-backward-by-shapes.py` | the 2×2 paper example against autograd; dh, dW2, db2 from shapes | 3 s |
| 6 | `labs/tanh-bn-atomic.py` | dhpreact to dhprebn; bndiff and hprebn False then True after their second branch | 3 s |
| 7 | `labs/bessel-correction.py` | biased vs unbiased variance of batches of 32 against the full-data variance; what `nn.BatchNorm1d` uses | 3 s |
| 8 | `labs/embedding-backward.py` | linear 1, the view, the dC loop and `index_add_`; the repeated characters | 3 s |
| 9 | `labs/cross-entropy-analytic.py [--plot]` | loss_fast, dlogits in 3 lines (maxdiff about 1e-8), row 0 × n, the row sums; `--plot` saves `dlogits.png` | 4 s |
| 10 | `labs/batchnorm-analytic-and-train.py [--quick]` | the dhprebn one-liner (maxdiff 9.31e-10); a gradient check of all 7 parameters; training with no autograd, then train/val and 20 samples | about 3 min idle (200k steps); `--quick` (20k steps) 20–75 s |

No lab is pencil-only in this course.
