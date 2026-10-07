# The math of an LLM: course facts

**The reader.** They're new to ML and working through Raschka's *Build a Large Language Model (From Scratch)* as their spine. The LLM basics path felt hard to them, and it wasn't clear what it led to. They want every equation in order, with real numbers, plus the *what* and the *why* behind each one.

**The output of this course.** By the end, the reader can do one full training step of a tiny transformer on paper (forward, loss, backward, update) and match PyTorch.

Every card uses the **same tiny model** and the **same numbers**. Never re-derive or round them differently.

## Running example 1: the tiny network (card `tiny-net-by-hand` only)

`labs/tiny_net.py` has 6 weights, 2 inputs, 2 hidden neurons with ReLU, 1 output, and squared-error loss. The full output is in `tasks/llm-math-kit/facts/tiny_net.out`.

- Setup: x = [1, 2], y = 1, W1 = [[0.5, −0.5], [0.3, 0.2]], w2 = [0.4, 0.6], lr 0.1.
- Forward: p = W1 x = [−0.5, 0.7]. ReLU gives h = [0, 0.7]. Then o = 0.42 and L = (0.42 − 1)² = 0.3364.
- Backward:
  - dL/do = −1.16
  - dL/dw2 = [0, −0.812]
  - dL/dh = [−0.464, −0.696]
  - dL/dp = [0, −0.696] (ReLU blocks neuron 1)
  - dL/dW1 = [[0, 0], [−0.696, −1.392]]
- Update: W1 row 2 becomes [0.3696, 0.3392] and w2 becomes [0.4, 0.6812]. Then o = 0.7139 and **L = 0.0819** (was 0.3364).
- PyTorch agrees exactly (difference 0.0).

## Running example 2: the tiny transformer (every other card)

`labs/tiny.py` is one GPT block built the way Raschka builds it in chapter 4, at hand size:
- vocab 5 (`the cat sat on mat`), 3 tokens of context, d_model 4, 1 head (d_head 4), feed-forward hidden 8, 1 block
- learned position embeddings, pre-LayerNorm (eps 1e-5, divides by D), GELU (tanh form), no qkv bias, a separate output head
- **220 weights** in total

Matrices are random and rounded to 1 decimal. Biases start at 0, and LayerNorm starts at scale 1, shift 0.
- **Convention:** rows are tokens and a layer is `x @ W` with W shaped (in, out). PyTorch's `nn.Linear` stores W.T. Say this once where it matters (`grad-linear-layer`, `vectors-and-matmul`).
- **Input** `the cat sat` = ids [0, 1, 2]. **Targets** `cat sat on` = ids [1, 2, 3]. Every position predicts the next token, and the loss is the mean over 3 positions.
- Full printed values: `tasks/llm-math-kit/facts/tiny.out`. Every array as JSON: `tasks/llm-math-kit/facts/tiny.json` (keys `W`, `values`, `grads`, `d`, …). **Take every number from these files.** For one entry of a matrix product, compute it in a scratch script and check it against the file.
- **The check:** numpy loss 2.206682 equals torch 2.206682. The largest gradient difference over all 220 weights is 5.6e-16.

Headline numbers (from tiny.out):
- **Embedding:** E row 0 (`the`) = [0.3, 0.8, 0.6, −0.5] and P row 0 = [−0.6, −0.7, 0.2, −0.9], so x0 row 0 = [−0.3, 0.1, 0.8, −1.4].
- **LayerNorm 1:** row means [−0.2, −0.075, −0.125], σ [0.7969, 1.1541, 0.6572]. Row 0 of a = [−0.1255, 0.3765, 1.2549, −1.5059].
- **Scores:** S_raw row 2 = [0.9141, 4.0013, −4.2507]. Divided by √4 = 2 that's [0.4570, 2.0006, −2.1253]. The mask puts −inf above the diagonal.
- **Attention weights A:** [[1, 0, 0], [0.4979, 0.5021, 0], [0.1737, 0.8132, 0.0131]].
- **Logits** z, last row: [0.4963, −0.5355, −0.3458, 0.0269, 2.4248]. p, last row: [0.1076, 0.0384, 0.0464, 0.0673, 0.7404]. The target `on` gets p = 0.0673.
- **Loss** per position: [1.9390, 1.9825, 2.6985], mean **2.2067**. ln 5 = 1.6094, which is what uniform guessing would score. The random model is *worse* than guessing, because it's confidently wrong: it puts 0.74 on `mat`.
- **dz = (p − y)/3**, last row [0.0359, 0.0128, 0.0155, −0.3109, 0.2468].
- **Biggest gradients:** dW1[0,0] = −0.4500 and dWout[1,3] = −0.6103. Gradients on Wq are small (largest |0.232|) because attention row 0 is fixed at 1 and row 1 is nearly 50/50.
- **dE:** rows 3–4 (`on`, `mat`) are all 0, because those tokens were never looked up.
- **One SGD step at lr 1.0:** loss 2.2067 → **1.3544**. One Adam step at lr 0.1: 2.2067 → **1.0965**.

## The "why" measurements (`labs/why.py`; full output in `tasks/llm-math-kit/facts/why.out`)

- **Nudge test:** add 0.001 to Wout[0,1] and the loss changes by +0.000288, so change/nudge = 0.2881, equal to the gradient 0.2881. For W1[0,0]: nudge 0.1 gives −0.4781, 0.001 gives −0.4503, and the gradient is −0.4500. The smaller the nudge, the closer the match.
- **Softmax:**
  - exp(1000) overflows to inf and the naive softmax gives nan. Subtracting the max gives [0, −1, −2], so softmax = [0.6652, 0.2447, 0.0900]. float32 exp overflows above 88.72.
  - Dividing logits by their sum instead of using exp gives negative "probabilities" ([0.2401, −0.2591, …]).
  - Temperature on the last logits row: T 0.5 puts 0.9652 on `mat`, T 1 puts 0.7404, T 2 puts 0.4628.
- **√d:** the std of q·k for random unit-variance vectors is 2.007 at d = 4, 8.005 at d = 64 and 29.898 at d = 896. That tracks √d (2, 8, 29.93), and after dividing by √d it's about 1.00.
  - d = 64 with 10 keys: the average top attention weight is 0.868 unscaled and 0.324 scaled.
  - In one unscaled row, the max weight is 0.9519 and the softmax slope A(1−A) is at most 0.0458, so the gradients are nearly dead.
- **Nonlinearity:** two linear layers W1 then W2 equal one 4×4 matrix W1 @ W2 (difference 4.4e-16). With GELU between them, the result differs from the merged matrix by up to 1.1481.
  - GELU(−1) = −0.1588 with slope −0.0830. GELU(0) = 0 with slope 0.5. GELU(1) = 0.8412 with slope 1.0830. GELU(3) = 2.9964.
- **Init / why normalise** (d = 896, 24 random layers): with W ~ N(0,1) the std goes 29.3, 859, 2.63e4, … then 3.22e35 at layer 24. With W ~ N(0, 1/d) it goes 0.993, 1.01, 1.01, then 1.08 at layer 24.
- **Residual:** the gradient size at the input of a stack of tanh layers (d = 64):

  | Depth | Without residual | With residual |
  |---|---|---|
  | 4 | 1.79e-05 | 0.125 |
  | 24 | 3.52e-16 | 0.371 |

  Raschka §4.4 shows the same effect with his own 5-layer example; cite the section, don't copy its numbers.
- **RoPE** (2-d pair, angle 0.5 rad per position; q = [1, 0], k = [0.6, 0.8]): the rotated q·k depends only on distance. Distance 1 gives 0.9101 whether the positions are (1,0), (3,2) or (7,6). Distance 2 gives 0.9974. Unrotated it's 0.6000.
  - Qwen2.5: rope_theta 1,000,000, head dim 64, so 32 pairs. The angle per position is 1 rad for pair 0, 0.649 for pair 1, 0.001 for pair 16 and 1.54e-6 for pair 31.
- **Learning rate, SGD on the tiny model:**

  | lr | Loss at steps 0 / 1 / 5 / 10 / 30 |
  |---|---|
  | 0.01 | 2.2067 / 2.0127 / 1.4982 / 1.2461 / 0.8542 |
  | 0.1 | 2.2067 / 1.2269 / 0.6927 / 0.3913 / 0.0871 |
  | 1.0 | 2.2067 / 1.3544 / 0.5302 / 0.5209 / **1.2475** (bounces back up) |
  | 10 | 20.6348 after step 1, then inf |
  | 100 | inf after step 1 |

  Adam lr 0.01: 2.2067 / 1.7487 / 1.0724 / 0.8404 / 0.2699. Adam lr 0.1: 2.2067 / 1.0965 / 0.4870 / 0.0358 / 0.0002.
- **Adam's first step** moves every weight by exactly lr = 0.1, whatever the gradient's size:
  - Wout[1,3]: gradient −0.61026, SGD step +0.61026, Adam step +0.1
  - Wq[1,0]: gradient −0.00385, SGD step +0.00385, Adam step +0.1
- **The book's recipe:** AdamW with lr 0.0004 and weight decay 0.1 (Raschka §5.2).

## At real scale (reuse; don't re-measure)

Qwen2.5-0.5B (from `text-to-answer/_facts.md`):
- 24 layers, hidden 896, 14 query heads, 2 KV heads, head dim 64
- FFN 4,864 (SwiGLU), vocab 151,936, tied embeddings, RoPE, RMSNorm, 494,032,768 parameters
- The prompt "What is the capital of India?" is 7 tokens, giving a 7 × 896 input.

GPT-2 124M (the book's model): 12 layers, hidden 768, 12 heads, vocab 50,257, context 1,024, learned positions, LayerNorm, GELU. Each card's "At real scale" strip states the same equation's shapes in GPT-2 and Qwen.

## Raschka sections (verified against the PDF's headings; cite chapter and section only, never copy text)

| Card | Raschka |
|---|---|
| `vectors-and-matmul` | A.2 (tensors), A.2.3, §3.3.1 (dot products as similarity) |
| `derivatives-and-chain-rule` | A.3 (computation graphs), A.4 (autograd) |
| `tiny-net-by-hand` | A.5, A.7 |
| `embedding-lookup` | §2.7 |
| `positions-rope` | §2.8 (learned/absolute positions) |
| `softmax` | §3.3.1 (normalising with softmax), §5.3.1 (temperature) |
| `attention-scores` | §3.4.1, §3.5.1 |
| `attention-output` | §3.4.1 (context vectors), §3.6 |
| `norm-and-residual` | §4.2, §4.4 |
| `ffn-and-gelu` | §4.3 |
| `logits-and-loss` | §4.6, §5.1.2 |
| `grad-softmax-ce` | §5.1.2, A.4 |
| `grad-linear-layer` | A.4, A.5 |
| `grad-ffn-norm-residual` | §4.4 |
| `grad-attention` | none (the book leaves this to autograd); go deeper with z2h-6 |
| `gradient-descent-and-adam` | §5.2, A.7, appendix D |
| `one-training-step` | §5.2, A.7 |

Code links: `https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02` … `/ch05`, `/appendix-A`, `/appendix-D` (all verified).

## Cards (`llm-math/<slug>`), in reading order

| # | Slug | Part | Title (claim) | One line |
|---|---|---|---|---|
| 0 | `_overview` | — | Every equation, in the order the data moves | the whole chain on one page (written last) |
| 1 | `vectors-and-matmul` | 0 Tools | One operation does almost all the work | dot product, matmul, shapes, (rows=tokens) |
| 2 | `derivatives-and-chain-rule` | 0 Tools | A derivative is a nudge ratio | slope, partials, chain rule, nudge test |
| 3 | `tiny-net-by-hand` | 0 Tools | Train a 6-weight network on paper | forward, loss, backward, update: the whole loop small |
| 4 | `embedding-lookup` | 1 Forward | A token becomes a row of a table | x = E[id] = one-hot · E; + positions |
| 5 | `positions-rope` | 1 Forward | Position is added, or rotated in | learned P vs RoPE rotation |
| 6 | `softmax` | 1 Forward | Softmax turns scores into shares | exp/Σexp, max trick, temperature |
| 7 | `attention-scores` | 1 Forward | Every token scores every earlier token | Q, K, QKᵀ/√d, causal mask |
| 8 | `attention-output` | 1 Forward | Attention is a weighted average of values | A·V, heads, output projection |
| 9 | `norm-and-residual` | 1 Forward | Normalise, then add, don't replace | LayerNorm (vs RMSNorm), x + f(x) |
| 10 | `ffn-and-gelu` | 1 Forward | The bend that makes depth worth having | W2·GELU(W1x); SwiGLU noted |
| 11 | `logits-and-loss` | 1 Forward | The loss is surprise at the right answer | h·Wout, softmax, −log p, 2.2067 vs ln 5 |
| 12 | `grad-softmax-ce` | 2 Backward | The gradient at the top is p − y | why it's that clean |
| 13 | `grad-linear-layer` | 2 Backward | Let the shapes write the gradient | dW = xᵀ·dy, dx = dy·Wᵀ |
| 14 | `grad-ffn-norm-residual` | 2 Backward | The residual is a gradient highway | GELU', LayerNorm backward, + copies |
| 15 | `grad-attention` | 2 Backward | Backprop through attention, three tokens | dA, softmax backward, dQ dK dV |
| 16 | `gradient-descent-and-adam` | 3 Learning | Step downhill, at the right size | SGD, lr sweep, momentum, Adam, AdamW |
| 17 | `one-training-step` | 3 Learning | One training step, by hand | the capstone; links the worksheet |
| — | `worksheet` | — | The one-step worksheet | printable grid with blanks (written last) |

## Existing hub cards to link (as `course/slug`; go deeper, don't repeat)

- Tools: `z2h-1-toolkit/vectors-dot-product`, `z2h-1-toolkit/matmul-by-shapes`, `z2h-1-toolkit/derivative-rules`, `z2h-1-toolkit/chain-rule`, `z2h-1-toolkit/exp-and-log`, `z2h-1-toolkit/softmax-and-cross-entropy`, `z2h-1-toolkit/mean-variance-gaussian`
- Backprop: `z2h-2-micrograd/derivative-as-nudge`, `z2h-2-micrograd/chain-rule-by-hand`, `z2h-2-micrograd/one-step-along-gradient`, `z2h-2-micrograd/mlp-and-training-loop`, `z2h-6-backprop-ninja/cross-entropy-analytic`, `z2h-6-backprop-ninja/matmul-backward-by-shapes`, `z2h-6-backprop-ninja/softmax-chain-backward`, `z2h-6-backprop-ninja/embedding-backward`
- Init and norms: `z2h-5-batchnorm/kaiming-init`, `z2h-5-batchnorm/expected-initial-loss`, `z2h-5-batchnorm/saturated-tanh`
- GPT: `z2h-8-gpt/scaled-attention`, `z2h-8-gpt/masked-softmax`, `z2h-8-gpt/self-attention-head`, `z2h-8-gpt/heads-in-the-model`, `z2h-8-gpt/feedforward-residual`, `z2h-8-gpt/layernorm-dropout-scale`, `z2h-8-gpt/adam-optimizer`, `z2h-8-gpt/embeddings-and-positions`, `z2h-10-gpt2/init`, `z2h-10-gpt2/optimizer-recipe`, `z2h-10-gpt2/first-loss`
- Route at Qwen scale: `text-to-answer/the-whole-route`, `text-to-answer/embedding-space`, `text-to-answer/positions`, `text-to-answer/ffn-and-experts`, `text-to-answer/decoding`
- Modern block: `reasoning-models/modern-block` (RoPE, RMSNorm, SwiGLU, GQA)
- Build: `build-an-llm/training-pairs`

## Colour roles (sketches and highlighter marks; the same on every card)

| Colour | Means | Mark class |
|---|---|---|
| yellow | tokens and input text | `m-tok` |
| blue | activations: vectors flowing forward (x, Q, K, V, h, logits, probabilities) | `m-act` |
| violet | weights, the things that get learned (E, Wq, W1, Wout, …) | `m-w` |
| green | the loss and gradients, the learning signal | `m-grad` |
| red | masked, overflowing, broken, or the failure case | `m-bad` |

## Notation (the same on every card)

- `x0` is the embeddings and `a` is after LayerNorm 1. `Q K V`, then `S` (scores), `A` (attention weights), `C = A V`, `o` (after Wo), `x1 = x0 + o`.
- `u = LN2(x1) W1 + c1`, `f = GELU(u) W2 + c2`, `x2 = x1 + f`, `h = LNf(x2)`, `z = h Wout`, `p = softmax(z)`, `L`.
- A gradient is written `dX`, meaning ∂L/∂X, with the same shape as X. Say this on every Part 2 card.
