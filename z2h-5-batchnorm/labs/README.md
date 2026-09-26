# Labs: makemore 3, activations, gradients and BatchNorm (z2h-5-batchnorm)

Runnable companions to Lecture 4 of Karpathy's Neural Networks: Zero to Hero
([video](https://www.youtube.com/watch?v=P6sfmUTpUmc),
[notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb)).
Each card's "▶ Build it" box shows the output of the lab with the same name.

## Setup

```bash
uv venv
uv pip install torch numpy matplotlib
```

Run every lab from the course folder (`z2h-5-batchnorm/`), for example `python labs/kaiming-init.py`.
The first run downloads `names.txt` into `labs/data/`. Everything runs on the CPU with one thread,
so the numbers are deterministic and match the notebook's printed outputs where the lecture printed them.

Flags:
- `--quick` runs 20k training steps instead of the lecture's 200k (for the labs that train the one-hidden-layer MLP).
- `--plot` saves the lecture's figure as a PNG next to the script. Without it, the labs print the numbers the plot shows.

`common.py` holds the shared pieces: the dataset and the 80/10/10 split (`random.seed(42)`), the
`torch.Generator().manual_seed(2147483647)` generator, the one-hidden-layer MLP at each stage of the
lecture's loss log, and the lecture's `Linear` / `BatchNorm1d` / `Tanh` classes.

## Run order

Times were measured on an M3 (8 cores) while other jobs were running, so an idle machine is faster. With a light load, each 200k-step run of the one-hidden-layer MLP takes about 25 s.

| # | Lab | What it prints | Time |
|---|---|---|---|
| 1 | `expected-initial-loss.py` | −ln(1/27) = 3.2958, the 4-class toy, and the starter MLP's step-0 loss (27.88) | 3 s |
| 2 | `squash-the-logits.py` | step-0 loss for W2 × 1 / 0.1 / 0.01 / 0; full runs of the original and fixed init (val 2.1698 → 2.1311) | 1–8 min (`--quick` 15 s) |
| 3 | `saturated-tanh.py` | hpreact range, h histogram counts, % of \|h\| > 0.99, dead columns; W1 × 0.2 then a full run (val 2.1027) | 1 min (`--quick` 5 s) |
| 4 | `dead-neurons.py` | tanh / sigmoid / ReLU / Leaky ReLU values and local gradients; ReLUs killed by a single step at a huge lr | 3 s |
| 5 | `kaiming-init.py` | the `x @ w` std toy (3.16 → 1.00), torch's gains, (5/3)/√30 = 0.3043, then a full run (val 2.1070) | 1.3 min (`--quick` 16 s) |
| 6 | `batchnorm-forward.py` | column stats before and after BN, b1.grad ≈ 0, a full BN run (val 2.1057) | 1.7 min (`--quick` 17 s) |
| 7 | `batchnorm-inference.py` | calibrated vs running stats, batch jitter, single-example eval, momentum 0.1 vs 0.001 | 1.7 min (`--quick` 11 s) |
| 8 | `resnet-and-torch-nn.py` | `nn.BatchNorm1d` / `nn.Linear` signatures, parameters vs buffers, `nn.Linear`'s init range, our BN vs torch's | 2 s |
| 9 | `pytorchify-layers.py` | the 17-layer list, 47,024 parameters (46,497 without BN), 1001 steps, eval losses, samples | 4 s |
| 10 | `activation-gradient-plots.py` | per-Tanh-layer std, % saturated and grad std for gains 0.5 / 1 / 5/3 / 3, the linear case, BN | 5 s |
| 11 | `update-to-data-ratio.py` | grad:data ratios, then the log10 update:data ratios for seven settings (lr, gain, BN, fan_in) | 13 s |

Card `resnet-and-torch-nn` is mostly reading: the torchvision ResNet `Bottleneck` block is at
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py (torchvision is not needed to run the lab).
