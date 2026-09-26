# Labs: makemore part 2, the MLP (z2h-4-mlp)

Runnable build-alongs for the cards of this course, following Karpathy's
[makemore part 2 lecture](https://www.youtube.com/watch?v=TCH_1BHY58I) and its
[notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb).

## Setup

```bash
uv venv
uv pip install torch numpy matplotlib
```

Run every lab from the course folder (`z2h-4-mlp/`), e.g. `python labs/embedding-lookup.py`.
The first run downloads `names.txt` into `labs/data/` (gitignored). Everything runs on the CPU.
`labs/common.py` holds the shared pieces (vocabulary, `build_dataset`, the seed-42 split,
parameter init with `torch.Generator().manual_seed(2147483647)`, the minibatch training loop).
Minibatch indices come from the same generator, so every run prints the same numbers.

## Run order (times measured on an M3 while other jobs were running)

| # | Lab | What it prints | Time |
|---|---|---|---|
| 1 | `python labs/why-counts-explode.py` | 27ⁿ possible contexts vs how many actually occur in names.txt, n = 1..4 | 2 s |
| 2 | (none) `bengio-embeddings` | pencil and paper: read Fig. 1 of the paper and sketch it | – |
| 3 | `python labs/rolling-window-dataset.py` | emma's 5 (context → next char) rows; X.shape (32, 3) | 1 s |
| 4 | `python labs/embedding-lookup.py` | `C[5] == one_hot(5) @ C`; the two type errors; `C[X]` is (32, 3, 2) | 1 s |
| 5 | `python labs/view-storage.py` | strides of `arange(18).view(...)`; `cat` == `unbind` == `view`, and only `view` shares memory | 1 s |
| 6 | `python labs/hidden-and-output.py` | shapes through the hidden and output layers; 3,481 parameters | 1 s |
| 7 | `python labs/cross-entropy.py` | manual loss vs `F.cross_entropy` (17.7697); the logit-100 `nan` | 1 s |
| 8 | `python labs/overfit-one-batch.py` | 1000 full-batch steps on 32 examples; predictions vs labels | 1 s |
| 9 | `python labs/minibatches.py` | full-batch vs minibatch step time; full-set loss while training | 4–8 s |
| 10 | `python labs/learning-rate-finder.py [--plot]` | loss per lr-exponent bin; then lr 0.1 → decay to 0.01 | 7–9 s |
| 11 | `python labs/splits-and-fit.py` | 80/10/10 split sizes; train/dev loss at 100 and 300 hidden units | 16–21 s |
| 12 | `python labs/embeddings-scale-sample.py [--quick] [--plot]` | 2-D embedding coordinates; the 11,897-param model's train/dev loss; 20 samples | `--quick` 19 s; full about 1–1.5 min (estimated from 0.22 ms/step, not yet timed) |

`--plot` saves a PNG next to the script (`learning-rate-finder.png`, `embeddings-2d.png`) instead of opening a window.
`--quick` trains the final model for 20,000 steps instead of the lecture's 200,000.
