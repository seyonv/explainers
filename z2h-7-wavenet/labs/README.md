# Labs: makemore 5, building a WaveNet (z2h-7-wavenet)

Runnable companions to Lecture 6 of Karpathy's Neural Networks: Zero to Hero ([video](https://www.youtube.com/watch?v=t3YJ5hKiMQ0), [notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb)). Each lab rebuilds what it needs from scratch and prints the numbers its card shows.

## Setup

```bash
uv venv
uv pip install torch numpy matplotlib
```

Run every lab from the course folder (`z2h-7-wavenet/`), for example `python labs/flatten-consecutive.py`. The first run downloads `names.txt` into `labs/data/`. Everything runs on the CPU.

`labs/common.py` holds the shared pieces: the dataset and the 80/10/10 split, the lecture's layer classes (`Linear`, `BatchNorm1d`, `Tanh`, `Embedding`, `Flatten`, `FlattenConsecutive`, `Sequential`), the flat and hierarchical models, and the training loop (SGD, batch 32, lr 0.1 dropping to 0.01 for the last quarter of the steps). As in the notebook, the layers draw from the global RNG seeded with `torch.manual_seed(42)`. The scale-up run follows the notebook's random stream: its step-0 loss is 3.3167 and its step-10,000 loss is 2.0576, both the same as the notebook.

`--quick` runs 20,000 steps instead of 200,000, with the lr drop at step 15,000.

## Run order

Times are for an Apple M3 while other jobs were running, so an idle machine is faster.

| # | Lab | What it prints | Time |
|---|---|---|---|
| 1 | `labs/smooth-loss-plot.py [--quick] [--plot]` | the 3-character baseline (12,097 params): lossi averaged per 1,000 steps, the drop at the lr decay, train/val; `--plot` saves `loss-raw.png` and `loss-smooth.png` | 75 s (200k steps); `--quick` 9 s |
| 2 | `labs/torch-nn-containers.py` | the `Sequential` model and each layer's output shape; samples in eval mode; the batch-of-1 nan when `training=False` is forgotten | 10 s |
| 3 | `labs/more-context-baseline.py [--quick]` | the `........ --> y` rows; the flat block_size-8 model (22,097 params), train/val | 2 min (200k steps); `--quick` 9 s |
| 4 | `labs/flatten-consecutive.py` | batched `@` shapes, `view` = explicit `cat` of pairs, the 3-level shape walk-through, parameter counts for n_hidden 200 and 68 | 2 s |
| 5 | `labs/batchnorm-3d-bug.py [--quick]` | `running_mean` shape (1,4,68) vs (1,1,68); train/val of the 22K hierarchical net with the bug, then fixed | 7–8 min of CPU (2 × 200k steps; 15 min wall under heavy load); `--quick` 40 s |
| 6 | `labs/scale-up.py [--quick]` | the final model (76,579 params): training log, train/val, 20 samples | about 2.5–4 min (200k steps); `--quick` 15–25 s |
| 7 | `labs/convolutions-and-workflow.py` | the 8 rows of one name, forwarded in a loop (logits (8, 27)), and how many tree nodes the loop recomputes | 2 s |

Pencil-only card (no lab): `wavenet-tree-idea` (sketch the WaveNet tree and label the receptive fields).
