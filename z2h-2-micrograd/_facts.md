# Shared facts: micrograd: backprop from scratch (z2h-2-micrograd)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
Lecture 1 of Zero to Hero, card by card: build a 100-line autograd engine that turns any expression of +, ×, tanh into a graph and computes every gradient in one backward pass, then train a tiny neural net with it.
- Lecture: L1. 10 concept cards plus `big-ideas.html` and `_overview.html` (written after the concept cards).
- **Running example / conventions for this course:** Karpathy's own numbers throughout: f(x)=3x²−4x+5; the expression a=2, b=−3, c=10, e=a·b, d=e+c, f=−2, L=d·f = −8 (grads a 6, b −4, c/e/d −2, f 4, L 1); the neuron x1=2, x2=0, w1=−3, w2=1, b=6.8813735870195432 → o=0.7071; the 4-example dataset xs=[[2,3,-1],[3,-1,0.5],[0.5,1,1],[1,1,-1]], ys=[1,-1,-1,1] with MLP(3,[4,4,1]) (41 params).

## Series facts (shared by every z2h-* course)

### The series: "Zero to hero: neural networks, built"
A build-along companion to Andrej Karpathy's **Neural Networks: Zero to Hero** (https://karpathy.ai/zero-to-hero.html, playlist https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). One course per lecture, following the lecture's own chapter order and notation, plus a prerequisites toolkit (z2h-1) and a start-here map (z2h-0).
Courses: z2h-0-start · z2h-1-toolkit · z2h-2-micrograd · z2h-3-bigram · z2h-4-mlp · z2h-5-batchnorm · z2h-6-backprop-ninja · z2h-7-wavenet · z2h-8-gpt · z2h-9-tokenizer · z2h-10-gpt2

### Sources (read the section for your card before writing)
- The four lecture maps in `~/Desktop/repos/explainers/tasks/z2h-kit/`: `map-A-micrograd-bigram.md` (L1, L2), `map-B-makemore-2-5.md` (L3–L6), `map-C-gpt-tokenizer.md` (L7, L8), `map-D-gpt2-repos.md` (L9, plus the **repo/link inventory in Part 2** and the **prerequisite links in Part 3**). They hold verbatim chapter lists with `&t=Ns` deep links, the code per chapter, and numbers tagged by where they came from ([NB] notebook, [VID] said in the video, [DERIVED], [?] uncertain). **Re-derive anything you put on a card**; don't trust a map claim you can't check (the maps flag their own doubts).
- Karpathy's code, cloned at `~/Desktop/repos/explainers/tasks/z2h-kit/src/`: micrograd, makemore, nn-zero-to-hero (lecture notebooks), ng-video-lecture, minbpe, nanoGPT, build-nanogpt. Link to the GitHub versions on cards (`https://github.com/karpathy/<repo>/blob/master/<path>`; build-nanogpt and minbpe also use `master`: check with the local clone's `git branch`), with `#L<n>-L<m>` line anchors where it helps.
- Only use URLs that appear in the maps, the local clones or this file. Never guess a URL.

### The reader
- Comfortable programmer, **no deep-learning background**. Wants to build every piece themselves and understand the code, the concepts and the math. Explain every symbol the first time it appears on a card; never assume a term from a later card.
- Machine: Apple M3 (8 cores), 24 GB, macOS. `python3` is 3.10.20 with **torch 2.14.0** (MPS available), numpy. Default device: `cpu` for makemore (L1–L6: small, deterministic, matches the video), `mps` for GPT work (L7, L9).
- Seeded results on torch 2.14 sometimes match the notebooks exactly (e.g. the L2 multinomial draws, the micrograd gradients, the WaveNet losses at step 0 and step 10,000) and sometimes don't (e.g. the bigram name samples). Always show what the course's lab printed. Where it differs from the video, add one muted line saying so.
- The labs run in the course venv (`tasks/z2h-kit/venv`, Python 3.10.14, torch 2.14.0); the reader builds the same venv on the setup-m3 card.
- The `serial` rows in a course's measurements were timed one job at a time, with the reader's usual background apps still running. Every other time was measured "under load" while other jobs ran in parallel, so treat those as upper bounds. For the `.where` badge, prefer the serial time; otherwise round an under-load time and say "about".

### Card structure for this series (concept-explainer + three additions)
1. Title + subtitle. The subtitle gives aliases, then **"Lecture N · <chapter title> · <a href=…&t=…s>h:mm:ss</a>"**.
2. In one breath (1–2 sentences, plain words).
3. **Predict first** (new): one `<details class="predict">` question the reader answers before the worked example; the answer (with the number) is inside. Make it a question whose answer is surprising or checks understanding (e.g. "a is used twice; what should a.grad be?").
4. Worked example with Karpathy's own numbers (see the course's running example). Show every computation. Use **MathML** (patterns.md §1) for any equation.
5. **▶ Build it** box (new, required on every concept card): see markup below.
6. The 2×2 grid ("Why it beats the alternatives", or "How it differs from related ideas", or the glossary variant).
7. Clarification `.note` (when this is not the right tool / the common misconception).
8. Footer: three or four `<p>` lines with bold labels: **Watch:** the chapter deep links; **Code:** Karpathy's files; **Lab:** this card's lab on GitHub; **Related:** 2–4 sibling cards (same course `slug.html`, other course `../z2h-N-name/slug.html`; only slugs from the card lists; OK if not written yet).

### ▶ Build it box (verbatim CSS; put it in the card's `<style>`)
```css
.build{border:1.5px solid var(--accent-border);background:var(--accent-bg);border-radius:10px;padding:12px 16px 10px;margin:18px 0}
.build .hd{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:6px}
.build .hd b{font-size:15px;color:var(--accent)}
.build .where{font-size:12px;color:var(--muted);border:1px solid var(--border);background:var(--bg);border-radius:999px;padding:2px 9px;white-space:nowrap}
.build p{font-size:14px;margin:6px 0}
.build .code{background:var(--bg)}
.build .out{color:var(--muted)}
.build .done{border-top:1px dashed var(--accent-border);padding-top:7px;margin-top:8px}
.predict{border:1px solid var(--border);border-radius:10px;padding:10px 14px;margin:14px 0;background:var(--surface)}
.predict summary{cursor:pointer;font-size:15px}
.predict summary b{color:var(--accent)}
.predict[open] summary{margin-bottom:6px}
.predict p{font-size:14px;margin:4px 0 0}
```
```html
<div class="build">
  <div class="hd"><b>▶ Build it</b><span class="where">runs on your M3 · under 1 s</span></div>
  <p><b>When:</b> pause the video at <a href="https://www.youtube.com/watch?v=VMj-3S1tku0&t=4948s">1:22:28</a> and type this before watching on.</p>
  <pre class="code">…≤ 14 lines, Karpathy's variable names…</pre>
  <p><b>You should see</b> (measured on your M3):</p>
  <pre class="code out">…real output from labs/…</pre>
  <p class="done"><b>Done when:</b> one checkable condition. · <b>Karpathy's version:</b> <a href="…">engine.py L24–L31</a> · <b>Lab:</b> <a href="https://github.com/seyonv/explainers/blob/main/COURSE/labs/SLUG.py">labs/SLUG.py</a></p>
</div>
```
- `.where` is one of: `runs on your M3 · <time>` · `your M3, slow · <time>` · `needs a rented GPU` (z2h-10 only) · `pencil and paper` (no code). Times come from the course's measurements.
- **Every "You should see" value must come from the course's Local measurements table** (labs run on the M3). If you need an output that isn't there, run the lab yourself (≤ 60 s, CPU) and report it; never write an output you didn't see.
- Code block CSS (the same `.code` class every series uses, verbatim):
```css
.code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;line-height:1.5;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin:10px 0;white-space:pre;overflow-x:auto;tab-size:4;color:var(--text)}
.code .k{color:var(--accent);font-weight:600}.code .c{color:var(--muted)}.code .o{color:var(--muted)}
```
  `<pre class="code">` with HTML-escaped content. `.k` = Python keywords, `.c` = comments. Lines ≤ 72 characters.

### Hand-drawn sketches (from the software-factory guide's style)
- Generator: `~/Desktop/repos/explainers/tasks/z2h-kit/sketch.mjs` (rough.js, seeded, Virgil font inlined). Read its header comment for the spec format. Use it from a small node script in your scratch folder:
  `import { sketch, sketchCss } from '~/Desktop/repos/explainers/tasks/z2h-kit/sketch.mjs'` (use the absolute path) → paste `sketchCss()` into the card's `<style>` **once** and the returned `<svg>` inside `<figure class="sketch">…<figcaption><span>what it shows</span><b>short takeaway</b></figcaption></figure>`. Example: `tasks/z2h-kit/proto/demo.mjs`.
- **Use sketches for** compute graphs, data flow, architectures, pipelines, "what goes where" pictures, and the big-ideas panels. **Don't use them for** anything with precise quantities (loss curves, bar charts, histograms, tables): those stay in the flat house style with CSS variables.
- Sketch palette (fixed across the series): **blue** = inputs, data and leaf values; **yellow** = operations (ellipse) and decisions (diamond); **green** = outputs, what the model learned, the correct/good path; **red** = bugs, the wrong/bad path; **grey** = hidden, internal or overhead; **dashed** outline = not built yet, optional, or can't be seen; gradient text in `color:'title'` (blue ink); notes in `color:'note'`. Keep text ≥ 15px in a 1000-wide viewBox; keep the viewBox ≤ 1000×640.
- Max one or two sketches per concept card.

### Colour meanings (flat style, same on every card)
- green `--accent`: the correct answer, what's learned/kept, the winner, the current step
- grey `--faint` / `--surface2`: discarded, masked, saturated/dead, overhead
- neutral `--text` / `--muted`: data at rest
- red `--red`: ✗, a bug, a loss that's too high, a cutoff, the worst value
- Gradients are always written `.grad` / `d<name>` in text; in sketches, blue ink.

### Terms (use these names; mention aliases once in the subtitle)
gradient (not "derivative vector"); loss; logits; forward pass / backward pass; backpropagation (backprop); parameters (weights and biases); learning rate (lr); minibatch; embedding; hidden layer; activation; preactivation (hpreact); train / dev (validation) / test split; negative log likelihood (NLL) = cross-entropy for these cards; token; context length (block_size); self-attention; head; residual connection; LayerNorm; BatchNorm; tokenizer; BPE.

### Writing rules
- Write in your own words. Paraphrase the lectures, papers and docs; quote at most a short phrase of Karpathy with its timestamp. Never reproduce figures or long code verbatim from notebooks (≤ 14 lines of code per block, and cite the file).
- Only state what a source says; mark inferences. Numbers: measured (labs), computed (show the arithmetic), sourced (link), or labelled "illustrative".
- Card length: about one tall screen; a build-heavy card may run longer. Cut prose before cutting the worked example or the Build-it box.

### Every card in the series (use these exact paths for Related links)
- **z2h-1-toolkit**: python-you-need, setup-m3, derivative-rules, chain-rule, exp-and-log, vectors-dot-product, matmul-by-shapes, tensors-dims-and-indexing, broadcasting-rules, probability-distributions-and-sampling, mean-variance-gaussian, softmax-and-cross-entropy, big-ideas
- **z2h-2-micrograd**: derivative-as-nudge, partial-derivatives, value-object, chain-rule-by-hand, one-step-along-gradient, a-neuron, automatic-backward, accumulate-gradients, more-ops-and-pytorch, mlp-and-training-loop, big-ideas
- **z2h-3-bigram**: names-dataset, bigram-counts, count-matrix-visualised, sampling-multinomial, broadcasting-keepdim-trap, likelihood-and-nll, smoothing, one-hot-and-linear-layer, softmax-as-exp-normalise, train-neural-bigram, big-ideas
- **z2h-4-mlp**: why-counts-explode, bengio-embeddings, rolling-window-dataset, embedding-lookup, view-storage, hidden-and-output, cross-entropy, overfit-one-batch, minibatches, learning-rate-finder, splits-and-fit, embeddings-scale-sample, big-ideas
- **z2h-5-batchnorm**: expected-initial-loss, squash-the-logits, saturated-tanh, dead-neurons, kaiming-init, batchnorm-forward, batchnorm-inference, resnet-and-torch-nn, pytorchify-layers, activation-gradient-plots, update-to-data-ratio, big-ideas
- **z2h-6-backprop-ninja**: leaky-abstraction, chunked-forward-and-cmp, softmax-chain-backward, sum-broadcast-duality, matmul-backward-by-shapes, tanh-bn-atomic, bessel-correction, embedding-backward, cross-entropy-analytic, batchnorm-analytic-and-train, big-ideas
- **z2h-7-wavenet**: smooth-loss-plot, torch-nn-containers, more-context-baseline, wavenet-tree-idea, flatten-consecutive, batchnorm-3d-bug, scale-up, convolutions-and-workflow, big-ideas
- **z2h-8-gpt**: shakespeare-lm, char-tokenizer-split, batches-of-chunks, bigram-baseline, train-and-script, adam-optimizer, average-the-past, tril-matmul-trick, masked-softmax, embeddings-and-positions, self-attention-head, attention-notes, scaled-attention, heads-in-the-model, feedforward-residual, layernorm-dropout-scale, decoder-nanogpt-chatgpt, big-ideas
- **z2h-9-tokenizer**: why-tokenization, tiktokenizer-tour, unicode-utf8, bpe-by-hand, stats-and-merge, train-loop, decode, encode, regex-presplit, encoder-py-special-tokens, build-gpt4-tokenizer, sentencepiece-llama, vocab-size-new-tokens, quirks-explained, big-ideas
- **z2h-10-gpt2**: the-target, the-module, load-and-forward, sampling, first-loss, overfit-then-load, weight-tying, init, precision, compile-flash, nice-numbers, optimizer-recipe, big-batch, data-and-evals, results, big-ideas


## Local measurements for this course (labs/ run on the reader's M3)
| card | what | command | result (verbatim, trimmed) | wall time | notes |
|---|---|---|---|---|---|
| derivative-as-nudge | f(3) and numerical slopes, h=0.001 | `python labs/derivative-as-nudge.py` | `f(3.0) = 20.0` · `x = 3.0000   slope (h=0.001) = 14.003000` · `x = -3.0000   slope (h=0.001) = -21.997000` · `x = 0.6667   slope (h=0.001) = 0.003000` | 0.03 s | (under load) |
| derivative-as-nudge | Karpathy's cell, x=2/3, h=1e-6 | same | `x = 2/3, h = 1e-6: 2.999378523327323e-06` | 0.03 s | matches the notebook exactly. (under load) |
| derivative-as-nudge | h too small (slope at x=3) | same | `h = 1e-06   slope = 14.000003002223593` · `h = 1e-09   slope = 14.000001158365194` · `h = 1e-12   slope = 14.001244608152774` · `h = 1e-15   slope = 10.658141036401503` | 0.03 s | the floating-point pitfall made concrete. (under load) |
| partial-derivatives | bump a, b, c by h=0.0001 | `python labs/partial-derivatives.py` | `bump a:  d1 4.0  d2 3.999699999999999  slope -3.000000000010772` · `bump b:  d1 4.0  d2 4.0002  slope 2.0000000000042206` · `bump c:  d1 4.0  d2 4.0001  slope 0.9999999999976694` | 0.02 s | map A says bumping a takes d "4 to 3.9996"; the arithmetic (2.0001·−3+10) and torch 2.14 run give 3.9997. The c row matches the notebook exactly. (under load) |
| value-object | L and the graph | `python labs/value-object.py` | `L = Value(data=-8.0)` · `d._prev (a set) holds: [('c', 10.0), ('e', -6.0)]   d._op = '+'` · text graph: a 2, b −3, c 10, e −6 = *(a, b), d 4 = +(c, e), f −2, L −8 = *(d, f), every grad 0.0000 | 0.02 s | no `dot` binary on this machine: the text graph stands in for `draw_dot`; `--plot` saves `value-object.gv`. (under load) |
| chain-rule-by-hand | grads by hand plus the `lol()` check | `python labs/chain-rule-by-hand.py` | `dL/da: by hand   6.0   lol() 6.000000000000227` · `dL/db: by hand  -4.0   lol() -3.9999999999995595` · `dL/dc: by hand  -2.0   lol() -1.9999999999988916` · `dL/de: ... -2.000000000000668` · `dL/dd: ... -2.000000000000668` · `dL/df: by hand   4.0   lol() 3.9999999999995595` | 0.02 s | b row matches the notebook's `lol()` output exactly. (under load) |
| one-step-along-gradient | +0.01·grad on a, b, c, f | `python labs/one-step-along-gradient.py` | `before:  L = -8.0` · `leaves now: a 2.06  b -3.04  c 9.98  f -1.96` · `after one step (+0.01*grad):  L = -7.286496` · `after one step (-0.01*grad):  L = -8.726303999999999` | 0.02 s | the minus-step row is extra (not in the video): stepping against the gradient lowers L. (under load) |
| a-neuron | forward pass | `python labs/a-neuron.py` | `n = 0.8813735870195432   o = tanh(n) = 0.7071067811865476   1/sqrt(2) = 0.7071067811865475` · `with b = 8 instead: tanh( 2.0 ) = 0.9640275800758169` | 0.02 s | video says 0.96 for b=8. (under load) |
| a-neuron | grads by hand | same | `1 - o**2 = 0.4999999999999999` · grads: b 0.5000, w1 1.0000, x1 -1.5000, x1*w1 0.5000, w2 0.0000, x2 0.5000, x2*w2 0.5000, x1*w1 + x2*w2 0.5000, n 0.5000, o 1.0000 | 0.02 s | (under load) |
| automatic-backward | forgot `o.grad = 1.0` | `python labs/automatic-backward.py` | `without o.grad = 1.0, every grad: [0.0]` | 0.02 s | (under load) |
| automatic-backward | topological order | same | `x1=2.0000, w1=-3.0000, x1*w1=-6.0000, x2=0.0000, w2=1.0000, x2*w2=0.0000, x1*w1 + x2*w2=-6.0000, b=6.8814, n=0.8814, o=0.7071` (another run: `b=6.8814, x2=0.0000, w2=1.0000, x2*w2=0.0000, x1=2.0000, ...`) | 0.02 s | `_prev` is a set, so the order changes from run to run; it always ends `n=0.8814, o=0.7071`, as in notebook cell 16, and children always come before parents. The notebook shows yet another order (starting with b). (under load) |
| automatic-backward | `o.backward()` grads | same | b 0.5000, w1 1.0000, x1 -1.5000, w2 0.0000, x2 0.5000, n 0.5000, o 1.0000 (same as by hand) | 0.02 s | (under load) |
| automatic-backward | the `_backward()` bug | same | `bug demo: 'NoneType' object is not callable` | 0.02 s | same error text as the video (about 1:15:05). (under load) |
| accumulate-gradients | `=` bug | `python labs/accumulate-gradients.py` | `--- Buggy ---` · `b = a + a = 6.0   a.grad = 1.0   (should be 2)` · `f = (a*b)*(a+b) = -6.0   a.grad = -6.0   b.grad = -6.0   (should be -3, -8)` | 0.02 s | (under load) |
| accumulate-gradients | `+=` fix | same | `--- Value ---` · `b = a + a = 6.0   a.grad = 2.0` · `f = (a*b)*(a+b) = -6.0   a.grad = -3.0   b.grad = -8.0` · `numerical: df/da = -3.0   df/db = -8.0` | 0.02 s | (under load) |
| more-ops-and-pytorch | new ops on Value | `python labs/more-ops-and-pytorch.py` | `a + 1 = Value(data=3.0)    2 * a = Value(data=4.0)    a / b = Value(data=0.5)    a - b = Value(data=-2.0)    a.exp() = Value(data=7.38905609893065)` | 1.28 s | a=Value(2.0), b=Value(4.0). The time is mostly the torch import. (under load) |
| more-ops-and-pytorch | tanh as one op vs rebuilt from exp | same | `tanh as one op        : o 0.7071  x2 0.5000  w2 0.0000  x1 -1.5000  w1 1.0000   nodes 10` · `tanh from exp, -, +, /: o 0.7071  x2 0.5000  w2 0.0000  x1 -1.5000  w1 1.0000   nodes 18` | 1.28 s | node count includes the hidden constant Values (2, 1, −1). (under load) |
| more-ops-and-pytorch | PyTorch cell | same | `0.7071066904050358` · `x2 0.5000001283844369` · `w2 0.0` · `x1 -1.5000003851533106` · `w1 1.0000002567688737` | 1.28 s | torch 2.14 matches the notebook digit for digit. (under load) |
| more-ops-and-pytorch | why the tiny deviations | same | `float32 b, then .double(): 6.881373405456543    float64 from the start: 6.881373587019543` · `all-float64 o 0.7071067811865476   x1.grad -1.4999999999999998` · `x2.requires_grad default for a new tensor: False` | 1.28 s | confirms map A's inference: `torch.Tensor([...])` rounds b to float32 before `.double()`. (under load) |
| mlp-and-training-loop | parameter count and init | `python labs/mlp-and-training-loop.py` | `number of parameters: 41 = (3*4+4) + (4*4+4) + (4*1+1) = 41` · `n([2.0, 3.0, -1.0]) = Value(data=0.42767811248845455)    (random init, random.seed(1337))` · `initial ypred: [0.4277, 0.5571, 0.7185, 0.452]   loss: 6.005402783252284` · `layer0 neuron0 w[0]: data 0.2355  grad -0.4537` | 0.07 s | the notebook has no seed (0.1658 there; the video's loss was ~7.12). The lab uses `random.seed(1337)` from micrograd's demo.ipynb. (under load) |
| mlp-and-training-loop | 20 steps, lr 0.1, with zero_grad | same | `0 6.005402783252284` · `1 3.202149646920501` · `2 2.4621709281541335` · `5 0.7000976207724656` · `6 0.2539413462114517` · `10 0.05335489910674919` · `19 0.01778187468615766` · `final ypred: [0.9259, -0.9872, -0.9272, 0.9255]` | 0.07 s | a from-scratch run. The notebook's 0.002056 → 0.001793 came from an already-trained net. (under load) |
| mlp-and-training-loop | forgot zero_grad (same seed) | same | `step  2: zero_grad 2.462171   forgot 2.432486` · `step  5: zero_grad 0.700098   forgot 0.001606` · `step 10: zero_grad 0.053355   forgot 0.009928` · `step 19: zero_grad 0.017782   forgot 0.000000` | 0.07 s | the buggy loop reaches a lower loss here: the piled-up grads act like a growing step size, and this 4-example problem is easy enough to survive it. That is Karpathy's point ("bugs hide"). The buggy loss is not monotonic (0.0016 at step 5, 0.0099 at step 10). (under load) |

Long runs (> 3 min)
- None. Every lab in this course finishes in under 2 s on the M3.


## Card list (z2h-2-micrograd)
| # | File | Title | Group | Brief |
|---|---|---|---|---|
| 1 | derivative-as-nudge.html | A derivative is a nudge ratio | Derivatives | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L1 00:00:25, 00:08:08). Gist: The slope says how much the output moves when you nudge the input. At x=3, f=3x²-4x+5 has slope 14. Build step: Define `f`, compute `(f(3+h)-f(3))/h` for h=0.001 and get ≈14; try x=-3 (≈-22) and x=2/3 (≈0). |
| 2 | partial-derivatives.html | Partial derivatives: one knob at a time | Derivatives | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L1 00:14:12). Gist: For d=a·b+c at (2,-3,10), bumping a gives slope -3 (=b), b gives 2 (=a), c gives 1. Build step: Bump each of a, b, c by h=0.0001 and print d1, d2, slope. |
| 3 | value-object.html | The Value object: numbers that remember | The engine | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L1 00:19:09). Gist: Wrap scalars and overload + and ×, so each result remembers its parents and op, forming a graph. L = -8. Build step: Write `Value` with `data, _prev, _op, label, grad`; build a,b,c,e,d,f,L; `draw_dot(L)`. This is the REFERENCE card the main agent writes first. |
| 4 | chain-rule-by-hand.html | Backprop by hand with the chain rule | The engine | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L1 00:32:10). Gist: Walk back from L: plus copies the gradient, times swaps in the other input. Grads: a=6, b=-4, c=e=d=-2, f=4. Build step: Set each `.grad` by hand; verify one with the `lol()` numerical check. |
| 5 | one-step-along-gradient.html | One step along the gradient | The engine | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L1 00:51:10). Gist: Nudging the leaves by 0.01·grad moves L from -8 to -7.2865: gradients let you steer the output. Build step: `x.data += 0.01*x.grad` for a,b,c,f; recompute L. |
| 6 | a-neuron.html | A neuron, backpropagated | The engine | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L1 00:52:52). Gist: tanh(x1w1+x2w2+b) = 0.7071. The local tanh gradient 1-o² = 0.5 flows back, giving w1=1, x1=-1.5, w2=0, x2=0.5. Build step: Add `tanh()`; build the neuron with b=6.88137...; fill grads by hand. |
| 7 | automatic-backward.html | Automatic backward: closures and topological order | Automating it | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L1 01:09:02, 01:17:32). Gist: Each op stores its own `_backward`; a topological sort guarantees the right order; `o.backward()` does it all. Build step: Add `_backward` closures to +, *, tanh; add `backward()` with `build_topo`; run `o.backward()`. |
| 8 | accumulate-gradients.html | The += bug: reused nodes add their gradients | Automating it | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L1 01:22:28). Gist: `b=a+a` should give a.grad=2, not 1. Gradients from multiple paths add up. Build step: Reproduce the bug with `=`; switch to `+=`; check a=-2,b=3 gives a.grad=-3, b.grad=-8. |
| 9 | more-ops-and-pytorch.html | More operations, and PyTorch agrees | Automating it | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L1 01:27:05, 01:39:31). Gist: Add exp, pow, div, sub, radd/rmul, and rebuild tanh from exp: same grads. PyTorch gives identical 0.7071 / 0.5 / 0 / -1.5 / 1. Build step: Implement `exp`, `__pow__`, `__truediv__`, etc.; run the torch cell with `.double()` and `requires_grad=True`. |
| 10 | mlp-and-training-loop.html | From neuron to MLP to a training loop | A neural net | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L1 01:43:55 - 02:14:03). Gist: Neuron, Layer, MLP(3,[4,4,1]) has 41 params. MSE loss on 4 examples. Loop: forward, zero_grad, backward, update, with the forgot-zero_grad bug. Build step: Write the classes and `parameters()`; run the 20-step loop with `p.grad = 0.0` and lr 0.1; watch the loss fall. Cover the zero_grad bug explicitly. The notebook's 20-step printout came from an already-trained net: don't present it as from scratch; use the labs run. |
