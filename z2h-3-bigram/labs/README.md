# Labs: makemore 1, a bigram language model (z2h-3-bigram)

Runnable companions to Lecture 2 of Karpathy's Neural Networks: Zero to Hero
([video](https://www.youtube.com/watch?v=PaCmpygFfXo),
[notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb)).
One lab per card, all on the CPU, all deterministic (seed `2147483647`).

## Setup

```bash
uv venv
uv pip install torch numpy matplotlib
```

Run every lab from the course folder (`z2h-3-bigram/`), e.g. `python labs/names-dataset.py`.
The first run downloads `names.txt` into `labs/data/` (gitignored). `common.py` holds the shared
loader, `stoi`/`itos` and the 27x27 count matrix `N`; each lab rebuilds everything it needs.

## Run order (times measured on an M3, CPU, while other jobs were running)

| # | Lab | What it prints | Time |
|---|---|---|---|
| 1 | `labs/names-dataset.py` | 32,033 names, lengths 2 to 15, the bigrams of 'emma', 228,146 bigrams in total | < 1 s |
| 2 | `labs/bigram-counts.py` | dict top-5 counts, the 27x27 int32 `N`, N[0,:5], m. = 516, .m = 2538, sums (`--plot` saves the imshow) | ~2 s |
| 3 | `labs/count-matrix-visualised.py` | the 'e' row, first/last letters, top-5 bigrams, zero cells (`--plot` saves the labelled imshow) | ~2 s |
| 4 | `labs/sampling-multinomial.py` | row 0 as probabilities, the rand(3) demo, 5 bigram vs 5 uniform samples | ~2 s |
| 5 | `labs/broadcasting-keepdim-trap.py` | sum shapes with and without keepdim; the bug makes row 0 sum to 7.0225 | ~2 s |
| 6 | `labs/likelihood-and-nll.py` | per-bigram probs of 'emma', NLL 2.4241 (3 words), 2.4540 (all), andrej 3.0391, andrejq inf | ~2 s |
| 7 | `labs/smoothing.py` | +1 fake counts: P(jq) 0.000342, andrejq 3.4834, loss 2.4546; bigger fake counts drift to log 27 | ~4 s |
| 8 | `labs/one-hot-and-linear-layer.py` | xs/ys of 'emma', dtypes, one-hot shapes, xenc @ W for 1 and 27 neurons | < 1 s |
| 9 | `labs/softmax-as-exp-normalise.py` | logits, counts, probs; per-example NLL on 'emma'; loss 3.7693 three ways | < 1 s |
| 10 | `labs/train-neural-bigram.py` | 3 steps on 'emma', 100 steps lr 50 on all 228,146 bigrams (with and without L2), row lookup, net vs counting probs, samples | ~10 s (`--iters 200`: ~20 s) |

Samples from `torch.multinomial` on current PyTorch differ from the ones in the video; the losses match.
