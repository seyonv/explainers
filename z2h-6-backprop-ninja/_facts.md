# Shared facts: makemore 4: becoming a backprop ninja (z2h-6-backprop-ninja)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
Lecture 5 of Zero to Hero: throw away loss.backward() and backpropagate by hand through every tensor op in the 2-layer MLP with BatchNorm (cross-entropy, linear layers, tanh, BatchNorm, the embedding table), then derive the short analytic forms and train with no autograd at all.
- Lecture: L5. 10 concept cards plus `big-ideas.html` and `_overview.html` (written after the concept cards).
- **Running example / conventions for this course:** The Colab exercise net: 4,137 params, forward loss 3.3377, all 26 manual gradients matching torch exactly; dlogits max diff 5.12e-9; dhprebn max diff 9.31e-10; the no-autograd run reaches train 2.0705 / val 2.1099 (map-B).

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
| leaky-abstraction | "clip the loss" bug vs Huber | `python labs/leaky-abstraction.py` | `delta : [-3.0, -0.5, 0.2, 0.8, 2.5]` · `grad, clipped delta : [0.0, -0.2, 0.08, 0.32, 0.0]` · `grad, Huber (intended) : [-0.4, -0.2, 0.08, 0.32, 0.4]` · `outliers ignored by the bug: 2 of 5` | 1.7 s | (under load). Toy numbers of my own choosing, not from the video; loss = mean of clipped delta², Huber ×2 so the two agree inside ±1 |
| leaky-abstraction | saturated tanh, dead ReLU | same | `1 - tanh(x)**2 : ['1.00e+00', '4.20e-01', '7.07e-02', '9.87e-03', '1.82e-04']` for x = 0,1,2,3,5 · `dead ReLU weight grad: [0.0, 0.0]` | (same run) | (under load) |
| chunked-forward-and-cmp | split, params, loss | `python labs/chunked-forward-and-cmp.py` | `splits: (182625, 3) (22655, 3) (22866, 3)` · `parameters: 4137` · `loss: 3.3444` | 1.3 s | (under load). Video/notebook say loss 3.3377; torch 2.14 gives 3.3444. Karpathy draws bngain/bnbias from the unseeded global RNG, so no seed reproduces 3.3377; the labs use `torch.manual_seed(42)` for them. 4,137 params matches |
| chunked-forward-and-cmp | intermediate shapes | same | `emb (32, 3, 10)` · `embcat (32, 30)` · `hprebn (32, 64)` · `bnraw (32, 64)` · `h (32, 64)` · `logits (32, 27)` · `counts_sum (32, 1)` · `logprobs (32, 27)` | (same run) | (under load) |
| chunked-forward-and-cmp | cmp on right / slightly off / wrong | same | `logprobs \| exact: True \| approximate: True \| maxdiff: 0.0` · `logprobs+1e-9 \| exact: False \| approximate: True \| maxdiff: 1.862645149230957e-09` · `logprobs wrong \| exact: False \| approximate: False \| maxdiff: 0.0625` | (same run) | (under load) |
| chunked-forward-and-cmp | the whole Exercise 1 | `python labs/chunked-forward-and-cmp.py --all` | 26 lines `exact: True \| approximate: True \| maxdiff: 0.0` (logprobs … C) · `26 of 26 exact` | 5.3 s | (under load). Matches the notebook: all 26 exact |
| softmax-chain-backward | dlogprobs | `python labs/softmax-chain-backward.py` | `dlogprobs: 32 non-zero of 864, each = -0.031250` · `logprobs \| exact: True \| approximate: True \| maxdiff: 0.0` | 3.2 s | (under load) |
| softmax-chain-backward | dprobs, row 0 | same | `probs \| exact: True` · `row 0: probs[0,Yb[0]] = 0.0178 -> dprobs = -1.7588` | (same run) | (under load) |
| softmax-chain-backward | two branches of counts | same | `counts_sum_inv \| exact: True` · `counts (1 of 2) \| exact: False \| approximate: False \| maxdiff: 0.0061292462050914764` · `counts_sum \| exact: True` · `counts \| exact: True \| approximate: True \| maxdiff: 0.0` · `norm_logits \| exact: True` | (same run) | (under load) |
| softmax-chain-backward | `**-1` vs `1.0/` | same | `counts_sum**-1 \| exact: True \| approximate: True \| maxdiff: 0.0` · `1.0/counts_sum \| exact: False \| approximate: True \| maxdiff: 9.313225746154785e-10` | (same run) | (under load). Confirms the notebook comment on torch 2.14: with `1.0/x` the manual gradient is not bit-exact |
| sum-broadcast-duality | toy duality | `python labs/sum-broadcast-duality.py` | `broadcast forward -> sum backward: b.grad = [3.0, 3.0]` · `sum forward -> broadcast backward: x.grad = [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]` | 2.7 s | (under load) |
| sum-broadcast-duality | dlogit_maxes ≈ 0 | same | `logit_maxes \| exact: True` · `dlogit_maxes[:5] = ['1.86e-09', '-3.73e-09', '0.00e+00', '-1.86e-09', '-2.79e-09']` · `largest \|dlogit_maxes\| = 1.02e-08` | (same run) | (under load). Video: "about 1e-9"; here up to 1e-8 |
| sum-broadcast-duality | one_hot second branch | same | `logits (1 of 2) \| exact: False \| approximate: True \| maxdiff: 1.0244548320770264e-08` · `one_hot of the max positions: shape (32, 27) row sums [1]` · `logits \| exact: True \| approximate: True \| maxdiff: 0.0` | (same run) | (under load). Without the max branch dlogits is already "approximate: True", because dlogit_maxes is so small |
| matmul-backward-by-shapes | 2×2 paper example | `python labs/matmul-backward-by-shapes.py` | `dA = dD @ B.T : [[5.0, 7.0], [4.0, 6.0]] autograd: [[5.0, 7.0], [4.0, 6.0]]` · `dB = A.T @ dD : [[7.0, -3.0], [10.0, -4.0]]` (= autograd) · `dc = dD.sum(0): [3.0, -1.0]` (= autograd) | 3.3 s | (under load). A=[[1,2],[3,4]], B=[[5,6],[7,8]], dD=[[1,0],[2,-1]]: my own numbers |
| matmul-backward-by-shapes | dh, dW2, db2 | same | `shapes: dlogits (32, 27) h (32, 64) W2 (64, 27) b2 (27,)` · `h`, `W2`, `b2` all `exact: True \| maxdiff: 0.0` · `wrong order h.T @ W2 -> mat1 and mat2 shapes cannot be multiplied (64x32 and 64x27)` | (same run) | (under load) |
| tanh-bn-atomic | tanh, bngain/bnbias/bnraw | `python labs/tanh-bn-atomic.py` | `hpreact`, `bngain`, `bnraw`, `bnbias`, `bnvar_inv`: `exact: True \| maxdiff: 0.0` | 3.2 s | (under load) |
| tanh-bn-atomic | bndiff False → True | same | `bndiff (1 of 2) \| exact: False \| approximate: False \| maxdiff: 0.0011906675063073635` → `bndiff \| exact: True \| approximate: True \| maxdiff: 0.0` | (same run) | (under load) |
| tanh-bn-atomic | hprebn False → True | same | `hprebn (1 of 2) \| exact: False \| approximate: False \| maxdiff: 0.0009674632456153631` → `hprebn \| exact: True \| approximate: True \| maxdiff: 0.0` · `shapes: dbngain (1, 64) dbnmeani (64,) dhprebn (32, 64)` | (same run) | (under load) |
| bessel-correction | full-data vs one batch | `python labs/bessel-correction.py` | `full-data variance of neuron 0: 3.0906 (N = 182,625)` · `one batch of 32: var(unbiased=False) = 2.8750 var(unbiased=True) = 2.9677` | 2.6 s | (under load). Neuron 0 of hprebn at init |
| bessel-correction | average over 10,000 batches | same | `1/n (biased) : 2.9913 = 0.9679 x true (expected 31/32 = 0.9688)` · `1/(n-1) (unbiased): 3.0878 = 0.9991 x true` | (same run) | (under load) |
| bessel-correction | what nn.BatchNorm1d uses | same | `nn.BatchNorm1d on that batch: running_var = 2.9677 (unbiased)` · `but the train-mode output uses the biased 2.8750: output std(unbiased=False) = 1.0000` · `(with the unbiased variance it would be 0.9842)` | (same run) | (under load). Confirms the lecture's train/test mismatch on torch 2.14 |
| embedding-backward | linear 1, view, dC | `python labs/embedding-backward.py` | `embcat`, `W1`, `b1`, `emb`, `C`: `exact: True \| maxdiff: 0.0` · `C (index_add_) \| exact: True \| approximate: True \| maxdiff: 0.0` · `max \|loop - index_add_\| = 0.0` | 3.0 s | (under load). The vectorized `index_add_` is bit-exact |
| embedding-backward | repeated characters | same | `Xb holds 96 lookups of 21 distinct characters; 6 rows of dC stay zero` · `most used: '.' x29 -> dC[0] is the sum of 29 rows of demb` | (same run) | (under load) |
| cross-entropy-analytic | loss_fast, dlogits | `python labs/cross-entropy-analytic.py` | `3.344412326812744 diff: 0.0` · `logits \| exact: False \| approximate: True \| maxdiff: 9.778887033462524e-09` | 4.4 s | (under load). Notebook says diff 2.38e-7 and maxdiff 5.12e-9; torch 2.14 gives diff 0.0 and maxdiff 9.78e-9 |
| cross-entropy-analytic | row 0 × n | same | `F.softmax(logits, 1)[0] = tensor([0.0684, 0.0903, 0.0185, 0.0491, 0.0193, 0.0808, 0.0254, 0.0357, 0.0178, ...])` · `dlogits[0] * n = tensor([ 0.0684, 0.0903, ..., 0.0357, -0.9822, 0.0342, ...])` · `Yb[0] = 8: p = 0.0178, so that entry is p - 1 = -0.9822` | (same run) | (under load). Notebook: correct index is 0.0165 → -0.9835 (different bngain draw) |
| cross-entropy-analytic | rows sum to 0 | same | `dlogits[0].sum() = 0.0` · `max \|row sum\| over the batch = 6.05e-09` | (same run) | (under load). Notebook row-0 sum 1.397e-9 |
| cross-entropy-analytic | what imshow shows | same | `imshow(dlogits): the 32 darkest cells are the targets (min -0.0308); every other cell is small and positive (max 0.0052)` · `argmin per row == Yb for all rows: True` | (same run) | (under load). `--plot` saves labs/dlogits.png (6.3 s) |
| batchnorm-analytic-and-train | BN forward and one-line backward | `python labs/batchnorm-analytic-and-train.py --quick` | `forward max diff: 4.76837158203125e-07` · `hprebn \| exact: False \| approximate: True \| maxdiff: 9.313225746154785e-10` | 74.7 s | (under load, heavy: load average 17). Both match the notebook exactly (4.77e-7, 9.31e-10) |
| batchnorm-analytic-and-train | params, gradient check | same | `parameters: 12297` · 7 lines like `(27, 10) \| exact: False \| approximate: True \| maxdiff: 1.1175870895385742e-08`; all 7 approximate True, maxdiffs 3.7e-9 … 1.5e-8 | (same run) | (under load). Lecture: "about 1e-9"; here 1e-9 to 1e-8 |
| batchnorm-analytic-and-train | training, no autograd (quick) | same | `0/ 20000: 3.7954` · `19000/ 20000: 2.3922` · `trained 20000 steps in 67 s` · `train 2.1894` · `val 2.1975` | (same run) | (under load). --quick only (20k steps, lr drop at 10k). Notebook step 0: 3.7805; full run: train 2.0705 / val 2.1099, full run not measured yet (see Long runs). Lr drops at max_steps//2 = step 100,000 as in the notebook (not 150k) |
| batchnorm-analytic-and-train | samples (quick) | same | `samples: carpaiza. jahleigh. mrix. taty. salaysie. rahnel. den. art. kaqhi. nellara. chaiiv. kaleig. dhan. join. quinn. sulin. alian. quwa. elo. dearyxi.` | (same run) | (under load). Samples from the 20k-step model; the video's differ (carmahzamille., khi., ...) |

Long runs (> 3 min)
- `python labs/batchnorm-analytic-and-train.py` (Exercise 4, 200k steps with the manual backward pass and Karpathy's Python dC loop): about 3 min idle (the 20k quick loop took 18 s at a load average of 6, and 42–67 s at load averages of 10–17); under heavy load it can take 7–11 min.


Serial timings (2026-09-26, one job at a time)

| card | what | command | result | wall time | notes |
|---|---|---|---|---|---|
| batchnorm-analytic-and-train | serial timing (one job at a time; background apps running) | `python labs/batchnorm-analytic-and-train.py` | `160000/ 200000: 1.9932` · `170000/ 200000: 1.8121` · `180000/ 200000: 1.9737` · `190000/ 200000: 1.9167` · `trained 200000 steps in 126 s` · `train 2.0712` · `val 2.1086` · `samples: carmahza. jahleigh. mri. reet. khalaysie. mahnen. delynn. jareen. ner. kia. chaiiv. kaleigh. ham. joce. quinn. shoista. jadbi. wavero. dearyxia. kaelli` | 128 s | serial |


## Card list (z2h-6-backprop-ninja)
| # | File | Title | Group | Brief |
|---|---|---|---|---|
| 1 | leaky-abstraction.html | Why write backprop by hand? | Why and how | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 1). Gist: Autograd is leaky: saturation, dead neurons, exploding gradients, and the loss-clipping bug; manual backprop was standard until about 2014. Build step: Read "Yes you should understand backprop"; spot the bug in "clip the loss". |
| 2 | chunked-forward-and-cmp.html | The forward pass in small steps, and a gradient checker | Why and how | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 2). Gist: Break the MLP+BN into about 20 atomic tensors, `retain_grad()` them all, and compare with `cmp()`; use non-zero inits. Build step: Open the Colab; run the forward pass (loss 3.3377); write `cmp`. |
| 3 | softmax-chain-backward.html | Backprop through log, normalise and exp | Exercise 1: atom by atom | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 3 (~0:13–0:33)). Gist: dlogprobs = −1/n at targets; dprobs = 1/p·…; dcounts_sum_inv sums across the broadcast; dcounts needs `+=` from two branches. Build step: Fill in dlogprobs…dnorm_logits one at a time; `cmp` each. |
| 4 | sum-broadcast-duality.html | Sum and broadcast are each other's backward | Exercise 1: atom by atom | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 3 (~0:22–0:41, ~1:09)). Gist: A forward sum becomes a backward broadcast, and a forward broadcast becomes a backward sum; dlogit_maxes≈0 because the shift doesn't change the loss; one_hot routes the max gradient. Build step: Implement dlogit_maxes and the second logits branch with `F.one_hot(...indices)`. |
| 5 | matmul-backward-by-shapes.html | Matmul backward is a matmul | Exercise 1: atom by atom | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 3 (~0:41–0:53)). Gist: A 2×2 paper example gives dA = dD@Bᵀ, dB = Aᵀ@dD, dc = dD.sum(0); in practice just make the shapes fit. Build step: Write dh, dW2, db2 by matching (32,64), (64,27), (27,). |
| 6 | tanh-bn-atomic.html | tanh and BatchNorm, atom by atom | Exercise 1: atom by atom | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 3 (~0:53–1:18)). Gist: `1−h²`; γ,β sums; the power rule on var_inv; 1/(n−1) broadcast; bndiff and hprebn each need two branches. Build step: Fill dhpreact through dhprebn; watch `bndiff` go False → True after the second branch. |
| 7 | bessel-correction.html | Bessel's correction and BatchNorm's train/test mismatch | Exercise 1: atom by atom | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 4). Gist: Dividing by n−1 gives an unbiased variance for small batches; the paper and PyTorch mix biased (train) and unbiased (running); Karpathy calls it a bug. Build step: Compare `x.var(unbiased=False)` vs `True` on a batch of 32 against the full-data variance. |
| 8 | embedding-backward.html | The embedding gradient: undoing view and indexing | Exercise 1: atom by atom | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 3 (~1:18–1:26)). Gist: Linear-1 backward; `view` back to (32,3,10); scatter-add rows into dC with `+=` for repeated characters. Build step: Write the dC double loop; optionally vectorize with `index_add_`. |
| 9 | cross-entropy-analytic.html | Cross-entropy backward in three lines: softmax − one-hot | Exercises 2–4: the short forms | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 5). Gist: dlogits = (softmax − 1[y])/n; rows sum to 0; gradients as push/pull forces proportional to the error. Build step: Implement the 3 lines (maxdiff 5e-9); `imshow(dlogits)`; print row 0 × n. |
| 10 | batchnorm-analytic-and-train.html | BatchNorm backward in one line, then train with no autograd | Exercises 2–4: the short forms | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 6–8). Gist: Derive via x̂→σ²→μ→x (dσ²/dμ = 0); one broadcasted line; plug everything into training under `no_grad`, giving val about 2.11. Build step: Implement the dhprebn one-liner (maxdiff 9e-10); run Exercise 4 with `loss.backward()` removed. |
