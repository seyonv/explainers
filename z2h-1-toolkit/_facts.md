# Shared facts: Math & code toolkit (z2h-1-toolkit)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
Everything Zero to Hero assumes and never stops to teach: the Python features micrograd leans on, your M3 setup, derivatives and the chain rule, logs, vectors and matrix shapes, tensors and broadcasting, probability, the Gaussian and softmax. Each card ends in code you run.
- Lecture: none (supplemented prerequisites). 12 concept cards plus `big-ideas.html` and `_overview.html` (written after the concept cards).
- **Running example / conventions for this course:** Every card works with tiny, hand-checkable numbers and ends in a runnable `labs/<slug>.py` (Python 3.10, torch 2.14). Where possible, use the same numbers the lectures later use (f(x)=3x²−4x+5 at x=3; a=2, b=−3, c=10; the 27-character alphabet '.abcdefghijklmnopqrstuvwxyz') so the lectures feel familiar. All cards here are [S] (supplemented): cite the reference you used (3Blue1Brown, the PyTorch docs pages listed in map-D Part 3, CS231n numpy tutorial).

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
| python-you-need | operator overloading, __rmul__ | `python labs/python-you-need.py` | `m1 + m2      = Money(cents=350)` / `3 * m1       = Money(cents=750)` / `3 * m1 without __rmul__ -> TypeError: unsupported operand type(s) for *: 'int' and 'Money'` / `sum([m1, m2, m1]) = Money(cents=600)` | 0.02 s | no torch import (under load) |
| python-you-need | closure + build_topo | `python labs/python-you-need.py` | `closure calls: ['backward of c', 'backward of b', 'backward of a']` / `topo order    : ['a', 'b', 'e', 'c', 'd', 'f', 'L']` / `reversed topo : ['L', 'f', 'd', 'c', 'e', 'b', 'a']` | 0.02 s | graph is micrograd's L = d*f, d = e+c, e = a*b (under load) |
| setup-m3 | versions, MPS | `python labs/setup-m3.py` | `python  : 3.10.14 \| arm64` / `torch   : 2.14.0` / `numpy   : 2.2.6` / `mps built    : True` / `mps available: True` / `cuda available: False` | 2.4 s | kit venv is Python 3.10.14; _facts says system python3 is 3.10.20 (under load) |
| setup-m3 | data files | `python labs/setup-m3.py` | `names.txt: 228,145 chars, 32,033 lines` / `input.txt: 1,115,394 chars, 40,000 lines` | (same run) | downloads to labs/data/ if missing (under load) |
| setup-m3 | CPU vs MPS matmul | `python labs/setup-m3.py` | `cpu: 2048x2048 matmul 27.4 ms  (628 GFLOP/s)` / `mps: 2048x2048 matmul 10.9 ms  (1576 GFLOP/s)` | (same run) | timing varies run to run; unseeded; illustrative only (under load) |
| setup-m3 | seeded rand | `python labs/setup-m3.py` | `torch.rand(3, generator=g): tensor([0.7081, 0.3542, 0.1054])` | (same run) | normalised = [0.6064, 0.3033, 0.0903], same as the L2 notebook (under load) |
| derivative-rules | f'(3) for 3x²−4x+5 | `python labs/derivative-rules.py` | `f(3) = 20.0` / `(f(3+h) - f(3))/h, h=0.0001: 14.000300` / `f'(3) = 6*3 - 4 = 14.0` | 0.02 s | f and x=3 are the L1 example; the notebook cells here are not saved with this output (under load) |
| derivative-rules | h too small | `python labs/derivative-rules.py` | `h=0.1     slope = 14.3000000000` / `h=0.001   slope = 14.0030000000` / `h=1e-05   slope = 14.0000300000` / `h=1e-12   slope = 14.0012446082` | (same run) | float64 round-off at h=1e-12 (under load) |
| derivative-rules | 7 rules, numeric check | `python labs/derivative-rules.py` | `x^3 -> 3x^2  2  12.000000  12.000006` / `e^x -> e^x  1  2.718282  2.718283` / `ln x -> 1/x  2  0.500000  0.500000` / `tanh -> 1-tanh^2  0.8814  0.500000  0.500000` / `a*b, d/da -> b  2  -3.000000  -3.000000` / `a/b, d/db -> -a/b^2  4  -0.125000  -0.125000` / `a+b, d/da -> 1  2  1.000000  1.000000` | (same run) | columns: rule, point, exact, numeric h=1e-6; a*b uses b=−3, a/b uses a=2, a+b uses b=10 (under load) |
| chain-rule | tanh(2x+1) at x=0.5 | `python labs/chain-rule.py` | `y = 2x+1 = 2.0` / `z = tanh(y) = 0.964028` / `dy/dx = 2.0` / `dz/dy = 1 - tanh(y)^2 = 0.070651` / `dz/dx = dz/dy * dy/dx = 0.141302` / `numeric (h=1e-6)      = 0.141301` / `torch autograd x.grad = 0.141302` | 1.1 s | (under load) |
| chain-rule | multi-path adds | `python labs/chain-rule.py` | `car vs walker = 2 * 4 = 8` / `q = x*x at x=3: x.grad = 6.0 (= 3 + 3, one per path)` / `b = a + a: a.grad = 2.0  (overwriting instead of += would give 1.0)` | (same run) | a+a matches micrograd's L1 bug example at 1:22:28 (under load) |
| exp-and-log | log rules | `python labs/exp-and-log.py` | `log(a*b) = -4.6052;  log a + log b = -4.6052` / `log(0.5) = -0.6931` / `log(0.1) = -2.3026` / `log(1/27) = -3.2958   -> -log(1/27) = 3.2958` | 2.2 s | a=0.2, b=0.05 (under load) |
| exp-and-log | "emma" product vs log-sum | `python labs/exp-and-log.py` | `P(.e em mm ma a.) = [0.0478, 0.0377, 0.0253, 0.3899, 0.196]` / `product         = 3.4782e-06` / `sum of logs     = -12.5690` | (same run) | bigram probabilities recomputed from names.txt; match map-A [recomputed] .e 0.0478, em 0.0377, mm 0.0253, ma 0.3899, a. 0.1960 (under load) |
| exp-and-log | underflow | `python labs/exp-and-log.py` | `1000 x (1/27), float32 product   = 0.0` / `1000 x (1/27), float32 sum of log = -3295.84` / `float32 smallest normal number = 1.175e-38; (1/27)**1000 = 10^-1431.4` | (same run) | (under load) |
| exp-and-log | overflow, subtract max | `python labs/exp-and-log.py` | `exp(88) float32 = 1.6516e+38` / `exp(89) float32 = inf` / `exp(100) float32 = inf` / `softmax naive     : [nan, nan, nan]` / `softmax minus max : [0.09, 0.2447, 0.6652]` | (same run) | logits [100, 101, 102] (under load) |
| vectors-dot-product | the L1 neuron | `python labs/vectors-dot-product.py` | `x.w = 2*(-3) + 0*1 = -6.0` / `n = x.w + b = 0.8814  (full: 0.8813735870195432)` / `o = tanh(n) = 0.7071  (1/sqrt(2) = 0.7071)` | 1.1 s | (under load) |
| vectors-dot-product | alignment | `python labs/vectors-dot-product.py` | `same     u.v =  2.00  cos =  1.00` / `45 deg   u.v =  1.00  cos =  0.71` / `90 deg   u.v =  0.00  cos =  0.00` / `opposite u.v = -1.00  cos = -1.00` / `x vs w: \|x\|=2.0000 \|w\|=3.1623 cos=-0.9487` | (same run) | u = [1, 0] (under load) |
| matmul-by-shapes | 2×3 @ 3×2 | `python labs/matmul-by-shapes.py` | `A (2, 3) @ B (3, 2) -> (2, 2)` / `tensor([[ 4., -1.],` / `        [10., -1.]])` / `B @ B (3,2)@(3,2) -> mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)` | 1.3 s | A=[[1,2,3],[4,5,6]], B=[[1,0],[0,1],[1,-1]] (under load) |
| matmul-by-shapes | batch, batched, params | `python labs/matmul-by-shapes.py` | `X (32, 6) @ W (6, 100) + b (100,) -> (32, 100)` / `same as looping over the 32 rows: True` / `(4,5,80) @ (80,200) -> (4, 5, 200)` / `Linear(30,200): 30*200 + 200 = 6200  (torch counts 6200)` / `Linear(200,27): 200*27 + 27 = 5427  (torch counts 5427)` | (same run) | loop equality needs atol=1e-5 (exact allclose was False: float32 summation order) (under load) |
| tensors-dims-and-indexing | dims, keepdim | `python labs/tensors-dims-and-indexing.py` | `M.sum(dim=0) = [5.0, 7.0, 9.0]  shape (3,)` / `M.sum(dim=1) = [6.0, 15.0]  shape (2,)` / `M.sum(dim=1, keepdim=True) shape (2, 1) = [[6.0], [15.0]]` | 0.8 s | M = [[1,2,3],[4,5,6]] (under load) |
| tensors-dims-and-indexing | view, storage | `python labs/tensors-dims-and-indexing.py` | `arange(18).view(3,3,2) shape (3, 3, 2)  shares storage: True` / `a.view(9, 2)[1] = [2, 3]   a.view(-1, 6).shape = (3, 6)` / `M.t().view(6) -> view size is not compatible with input tensor's size and str ...` | (same run) | (under load) |
| tensors-dims-and-indexing | C[X], dtype | `python labs/tensors-dims-and-indexing.py` | `C (27, 2)  X (32, 3)  C[X] (32, 3, 2)` / `C[5] = [-0.4713, 0.7868]` / `torch.tensor([1, 2, 3]).dtype = torch.int64` / `torch.Tensor([1, 2, 3]).dtype = torch.float32` | (same run) | X is random ints here, not the real names dataset; C from seed 2147483647 (under load) |
| broadcasting-rules | shape pairs | `python labs/broadcasting-rules.py` | `(27, 27) + (27, 1) -> (27, 27)` / `(27, 27) + (27,) -> (27, 27)` / `(32, 64) + (64,) -> (32, 64)` / `(4, 8, 32) + (8, 32) -> (4, 8, 32)` / `(3, 4) + (3,) -> error` / `(5, 1, 4) + (3, 1) -> (5, 3, 4)` | 0.8 s | spacing trimmed (under load) |
| broadcasting-rules | keepdim bug | `python labs/broadcasting-rules.py` | `keepdim=True : [[0.25, 0.25, 0.5], [0.75, 0.0, 0.25], [0.6, 0.2, 0.2]]` / `row sums  : [1.0, 1.0, 1.0]` / `keepdim=False: [[0.25, 0.25, 0.2], [0.75, 0.0, 0.1], [1.5, 0.5, 0.2]]` / `row sums  : [0.7, 0.85, 2.2]` | (same run) | N = [[1,1,2],[3,0,1],[6,2,2]], row sums [4, 4, 10] (under load) |
| probability-distributions-and-sampling | p and cumsum | `python labs/probability-distributions-and-sampling.py` | `p = [0.6064, 0.3033, 0.0903]  sum = 1.0` / `cumsum = [0.6064, 0.9097, 1.0]` | 1.3 s | matches L2 notebook (under load) |
| probability-distributions-and-sampling | notebook replay | `python labs/probability-distributions-and-sampling.py` | `notebook replay, first 20 of 100: [1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]` / `counts of 0/1/2 in 100: [61, 33, 6]` | (same run) | first 20 identical to the L2 notebook output: this multinomial call DOES reproduce on torch 2.14 (under load) |
| probability-distributions-and-sampling | cumsum by hand, multinomial | `python labs/probability-distributions-and-sampling.py` | `u = [0.7081, 0.3542, 0.1054, 0.5996, 0.0904] -> picks [1, 0, 0, 0, 0]` / `multinomial x20: [1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 0, 2, 1, 1, 1, 2, 0, 0, 1]` / `replacement=False (default), 20 of 3 -> cannot sample n_sample > prob_dist.size(-1) samples without replacement` | (same run) | fresh g for each (under load) |
| probability-distributions-and-sampling | law of large numbers | `python labs/probability-distributions-and-sampling.py` | `n=    10: frequencies [0.3, 0.5, 0.2]` / `n=   100: frequencies [0.6, 0.32, 0.08]` / `n= 10000: frequencies [0.6088, 0.3012, 0.09]` | (same run) | (under load) |
| probability-distributions-and-sampling | first letter of names | `python labs/probability-distributions-and-sampling.py` | `32033 names; first-letter counts: a=4410 m=2538; total=32033` / `p['.'] = 0.0000  p['a'] = 0.1377  p['m'] = 0.0792` / `10 first letters: msnaattamm` | (same run) | counts match map-A row 0 (a=4410, m=2538); p['a']=0.1377 matches L2 notebook (under load) |
| mean-variance-gaussian | mean/var/std | `python labs/mean-variance-gaussian.py` | `mean = 5.0  var (n) = 4.0  std = 2.0` / `var (n-1, torch default) = 4.5714  std = 2.1381` | 0.9 s | v = [2,4,4,4,5,5,7,9] (under load) |
| mean-variance-gaussian | randn stats | `python labs/mean-variance-gaussian.py` | `randn(100000): mean -0.0037  std 1.0014` / `within 1 std: 0.6817  within 2: 0.9546  within 3: 0.9974` | (same run) | seed 2147483647 (under load) |
| mean-variance-gaussian | std of sums ~ √n | `python labs/mean-variance-gaussian.py` | `sum of   1 unit gaussians: std 0.997` / `sum of   4 unit gaussians: std 2.002` / `sum of  10 unit gaussians: std 3.166` / `sum of 100 unit gaussians: std 9.960` | (same run) | (under load) |
| mean-variance-gaussian | x@w, fan_in 10 | `python labs/mean-variance-gaussian.py` | `x (1000,10) std 1.0070;  y = x @ w (200 outputs) std 3.1613` / `w * 5 -> y std 15.8067` / `w * 0.2 -> y std 0.6323` / `w / sqrt(10) -> y std 0.9997` / `standardised y (per column): mean 0.0000  std 1.0000` | (same run) | video (L4 ~0:29:37, unseeded) says about 3, 15 and 0.6; seeded torch 2.14 gives 3.16, 15.81, 0.63 (under load) |
| softmax-and-cross-entropy | softmax of [2, 1, 0.1] | `python labs/softmax-and-cross-entropy.py` | `exp     = [7.3891, 2.7183, 1.1052]  sum = 11.2125` / `softmax = [0.659, 0.2424, 0.0986]` / `F.softmax matches: True` | 1.2 s | (under load) |
| softmax-and-cross-entropy | scale, shift, stability | `python labs/softmax-and-cross-entropy.py` | `softmax(logits * 0.5) = [0.5017, 0.3043, 0.194]` / `softmax(logits * 8.0) = [0.9997, 0.0003, 0.0]` / `softmax(logits + 100) = [0.659, 0.2424, 0.0986] (shift does nothing)` / `naive exp on [1000, 999, 998] -> [nan, nan, nan]` / `subtract max first            -> [0.6652, 0.2447, 0.09]` | (same run) | (under load) |
| softmax-and-cross-entropy | cross-entropy | `python labs/softmax-and-cross-entropy.py` | `target 0: -log p = 0.4170   F.cross_entropy = 0.4170` / `target 1: -log p = 1.4170` / `target 2: -log p = 2.3170` / `certain (p=1): -log 1 = 0.0000` / `uniform over 27: -log(1/27) = 3.2958` / `F.cross_entropy(zeros(27), any target) = 3.2958` | (same run) | (under load) |
| softmax-and-cross-entropy | likelihood naming | `python labs/softmax-and-cross-entropy.py` | `likelihood = prod p = 0.0100` / `log likelihood = sum log p = -4.6052` / `average negative log likelihood = 1.5351` | (same run) | p(correct) = [0.4, 0.1, 0.25] (under load) |

Long runs (> 3 min)
- None. Every z2h-1-toolkit lab finishes in under 3 s on the M3; the first run of `python labs/setup-m3.py` also downloads names.txt (~228 KB) and input.txt (~1.1 MB).


## Card list (z2h-1-toolkit)
| # | File | Title | Group | Brief |
|---|---|---|---|---|
| 1 | python-you-need.html | The Python you need for micrograd | Your tools | [S] Teach: class with __init__/__repr__; operator overloading (__add__, __mul__, __radd__/__rmul__ and why 2*a needs __rmul__); closures (a function capturing variables, used for _backward); set() and recursion (used in build_topo); list comprehensions and sum(). Worked example: a 10-line class Money that supports m1 + m2 and 3 * m1, with real output. Point forward to z2h-2 value-object. |
| 2 | setup-m3.html | Setting up your M3 for the course | Your tools | [S] Exact commands (uv venv; uv pip install torch numpy matplotlib jupyter tiktoken; clone karpathy repos; download names.txt and tiny shakespeare). Check: torch.__version__ 2.14.0, torch.backends.mps.is_available() True (measured). When to use device='cpu' (makemore: small, deterministic, matches the video) vs 'mps' (GPT). Comparison table: run location = your M3 / Google Colab free GPU / rented 8xA100, with what each lecture needs (use _facts timings). Mention that seeded samples differ from the video on torch 2.14. |
| 3 | derivative-rules.html | Derivatives: the rules you'll actually use | Calculus | [S] A table of the 7 rules the course uses (x^n, e^x, ln x, tanh x → 1−tanh², a·b, a/b, a+b), each with a numeric check (f(x+h)−f(x))/h at a concrete point computed by labs script. Worked example: f(x)=3x²−4x+5 → f'(x)=6x−4, at x=3 gives 14 (micrograd's first example). |
| 4 | chain-rule.html | The chain rule | Calculus | [S] Chain rule dz/dx = dz/dy · dy/dx with the car/bike/walker analogy (Karpathy uses this Wikipedia analogy in L1). Worked: z = tanh(2x+1) at x=0.5, compute every local derivative and the product; verify numerically. Show it as a tiny sketch graph (sketch.mjs). Multi-path: when x feeds two branches, gradients add (preview of the += bug). |
| 5 | exp-and-log.html | exp and log: turning products into sums | Numbers in bulk | [S] exp/log as inverses; log(a·b)=log a+log b; log of numbers in (0,1] is ≤ 0; log(1/27) = −3.2958 (the uniform loss used again and again). Worked: the probability of a 5-character word as a product of 5 probabilities vs a sum of 5 logs (underflow demo in float32 for 1000 factors). exp blows up: exp(100) overflow in float32 → why softmax subtracts the max. |
| 6 | vectors-dot-product.html | Vectors and the dot product | Numbers in bulk | [S] Vector as a list of numbers; dot product as weighted sum; geometric meaning (alignment). Worked: x=[2,0], w=[-3,1] → x·w=−6, +b=6.8813735870195432 → 0.8814 (the L1 neuron's numbers). Show the neuron as a sketch. |
| 7 | matmul-by-shapes.html | Matrix multiply by shapes | Numbers in bulk | [S] Matmul as many dot products; the shape rule (n,k)@(k,m)→(n,m); a 2×3 @ 3×2 example computed by hand and in torch; why a whole batch of inputs through a layer is one matmul. Batched matmul (4,5,80)@(80,200)→(4,5,200) preview for WaveNet. Parameter count of a Linear layer (fan_in·fan_out + fan_out). |
| 8 | tensors-dims-and-indexing.html | Tensors: shapes, dims and indexing | Numbers in bulk | [S] 0-d to 3-d tensors; .shape; dim=0 vs dim=1 (sum over rows vs columns, with a 2×3 example); keepdim; .view reshapes without copying (storage); indexing with a tensor of indices (C[X] shape (32,3,2)); torch.tensor vs torch.Tensor (float vs int dtype pitfall). Point to the PyTorch docs links in map-D Part 3. |
| 9 | broadcasting-rules.html | Broadcasting: the two rules | Numbers in bulk | [S] The two rules from the PyTorch broadcasting semantics doc (link it). Table of shape pairs → result or error: (27,27)+(27,1), (27,27)+(27,) (the silent bug: divides columns), (32,64)+(64,), (4,8,32)+(8,32). Worked 3×3 example with numbers showing the silent keepdim bug. Forward pointer to z2h-3 broadcasting-keepdim-trap and z2h-6 sum-broadcast duality. |
| 10 | probability-distributions-and-sampling.html | Probability distributions and sampling | Probability | [S] Discrete distribution over 27 characters; normalise counts; sampling by the cumulative-sum method explained with a 3-outcome example p=[0.6064, 0.3033, 0.0903] (the L2 generator example, seed 2147483647 → check with labs), then torch.multinomial with a seeded torch.Generator; replacement=True pitfall. Law of large numbers: 10k samples approach p. |
| 11 | mean-variance-gaussian.html | Mean, variance and the Gaussian | Probability | [S] Mean, variance, std; Gaussian randn (mean 0, std 1); sum of n independent unit-variance numbers has variance n → std √n; so x@w with fan_in 10 gives std ≈ 3.16, and dividing w by √10 restores std 1 (numbers from L4, compute with labs, fixed seed). Standardising (x−mean)/std = what BatchNorm does. Bessel's n−1 in one line (forward link to z2h-6 bessel-correction). |
| 12 | softmax-and-cross-entropy.html | Softmax and cross-entropy, first look | Probability | [S] Softmax step by step on logits [2.0, 1.0, 0.1] (compute), temperature/scale effect (×8 sharpens: forward link to z2h-8 scaled-attention), subtract-max for stability; cross-entropy = −log p(correct) and its range (0 when certain, log 27 = 3.2958 when uniform over 27); likelihood vs log-likelihood vs average negative log-likelihood naming. Preview only: z2h-3 and z2h-4 do it for real. |
