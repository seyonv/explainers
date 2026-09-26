# z2h-1-toolkit labs

Runnable companions to the Math & code toolkit cards of "Zero to hero: neural networks, built". Each lab is self-contained and prints exactly the numbers its card shows.

## Setup (once)

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install torch numpy matplotlib jupyter tiktoken
```

The labs that need data (`setup-m3.py`, `exp-and-log.py`, `probability-distributions-and-sampling.py`) download `names.txt` and tiny shakespeare `input.txt` into `labs/data/` on first run.

## Run order

Run from the course folder (`z2h-1-toolkit/`), for example `python labs/setup-m3.py`. Times are on an Apple M3 (most of each is the `import torch`).

| # | Lab | What it prints | Time |
|---|---|---|---|
| 1 | `labs/python-you-need.py` | a `Money` class with `+`, `*`, `__rmul__`, `sum()`; a closure; `build_topo` with `set()` and recursion | < 1 s |
| 2 | `labs/setup-m3.py` | Python/torch/numpy versions, MPS available, data files, a CPU vs MPS matmul | ~2 s (+ download on first run) |
| 3 | `labs/derivative-rules.py` | f(x)=3x²−4x+5 slope at x=3 (14), and a numeric check of each of the 7 rules | < 1 s |
| 4 | `labs/chain-rule.py` | z = tanh(2x+1) at x=0.5: local derivatives, their product, numeric and autograd checks; gradients add over two paths | ~1 s |
| 5 | `labs/exp-and-log.py` | log rules, log(1/27), "emma" under the bigram model as product vs sum of logs, float32 underflow and overflow | ~2 s |
| 6 | `labs/vectors-dot-product.py` | the L1 neuron: x·w = −6, n = 0.8814, tanh = 0.7071; cosine alignment | ~1 s |
| 7 | `labs/matmul-by-shapes.py` | a 2×3 @ 3×2 by hand and in torch, a batch through a layer, (4,5,80)@(80,200), Linear param counts | ~1 s |
| 8 | `labs/tensors-dims-and-indexing.py` | ndim/shape, sum over dim 0 vs 1, keepdim, view and storage, C[X] → (32,3,2), tensor vs Tensor dtype | ~1 s |
| 9 | `labs/broadcasting-rules.py` | shape pairs → result or error; the keepdim bug on a 3×3 count matrix | ~1 s |
| 10 | `labs/probability-distributions-and-sampling.py` | p = [0.6064, 0.3033, 0.0903], cumsum sampling, multinomial, replacement pitfall, law of large numbers, first-letter distribution of names.txt | ~1 s |
| 11 | `labs/mean-variance-gaussian.py` | mean/var/std, randn stats, std of sums grows as √n, x@w std 3.16 → 1 after /√10, standardising. `--plot` saves a histogram PNG | ~1 s |
| 12 | `labs/softmax-and-cross-entropy.py` | softmax of [2.0, 1.0, 0.1], scaling, subtract-max, cross-entropy per target, log 27 = 3.2958, likelihood naming | ~1 s |

`common.py` holds the data-download helper.
