# Labs: micrograd, backprop from scratch (z2h-2-micrograd)

Runnable companions to the cards of this course, for Lecture 1 of Karpathy's Neural Networks: Zero to Hero ([video](https://www.youtube.com/watch?v=VMj-3S1tku0), [micrograd](https://github.com/karpathy/micrograd)). Each lab prints the numbers its card shows.

## Setup

```bash
uv venv
uv pip install torch numpy matplotlib graphviz
```

Only `more-ops-and-pytorch.py` needs torch. The labs don't need a dataset. The `--plot` flag saves PNGs next to the script. Graph pictures also need the graphviz `dot` binary (`brew install graphviz`); without it the labs save the `.gv` source instead. By default every lab prints a text version of the graph.

## Run order

Run each lab from the course folder, e.g. `python labs/derivative-as-nudge.py`. They run in this order, one per card:

| # | Lab | What it prints | Time on the M3 |
|---|---|---|---|
| 1 | `derivative-as-nudge.py` | f(3)=20; slopes ≈14, −22, 0; what happens when h gets too small | < 0.1 s |
| 2 | `partial-derivatives.py` | d1, d2 and the slope for bumping a, b, c (−3, 2, 1) | < 0.1 s |
| 3 | `value-object.py` | a minimal `Value`; L = −8 and the graph as text | < 0.1 s |
| 4 | `chain-rule-by-hand.py` | grads set by hand (a 6, b −4, c/e/d −2, f 4) checked with `lol()` | < 0.1 s |
| 5 | `one-step-along-gradient.py` | one step of +0.01·grad: L goes from −8 to −7.286496 | < 0.1 s |
| 6 | `a-neuron.py` | o = 0.7071, 1−o² = 0.5, grads w1 1, x1 −1.5, w2 0, x2 0.5 | < 0.1 s |
| 7 | `automatic-backward.py` | `_backward` closures, the topological order, `o.backward()` | < 0.1 s |
| 8 | `accumulate-gradients.py` | the `=` bug next to the `+=` fix (a.grad 1 vs 2; −3, −8) | < 0.1 s |
| 9 | `more-ops-and-pytorch.py` | the new ops, tanh rebuilt from exp, and the same numbers from PyTorch | about 1.3 s (importing torch) |
| 10 | `mlp-and-training-loop.py` | MLP(3,[4,4,1]) has 41 params; a 20-step loop, with and without zero_grad | < 0.1 s |

`common.py` holds the lecture's finished `Value` engine (with every op from lab 9), the `draw_dot` helper, and the text graph printer. Labs 3, 7 and 8 define their own `Value` at that card's stage.
