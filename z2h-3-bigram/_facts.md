# Shared facts: makemore 1: a bigram language model (z2h-3-bigram)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
Lecture 2 of Zero to Hero: model names one character at a time, first by counting letter pairs, then with a one-layer neural net trained by gradient descent. The two approaches land on the same answer, loss ≈ 2.45.
- Lecture: L2. 10 concept cards plus `big-ideas.html` and `_overview.html` (written after the concept cards).
- **Running example / conventions for this course:** names.txt (32,033 names, 228,146 bigrams), the 27-token alphabet with '.' as start/end, the 27×27 count matrix N, the word 'emma' as the running word, and the loss numbers in map-A's numbers cheat-sheet (2.4540 unsmoothed full-set NLL; neural net 3.7693 → …). Samples differ from the video on torch 2.14 — say so where samples appear.

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
| names-dataset | dataset size and lengths | `python labs/names-dataset.py` | `len(words): 32033` · `shortest: 2` · `longest : 15` · `a longest name: muhammadibrahim` | 0.7 s | (under load) |
| names-dataset | bigrams of 'emma' | `python labs/names-dataset.py` | `zip(w, w[1:]) for emma: ['em', 'mm', 'ma']` · `with '.' start/end    : ['.e', 'em', 'mm', 'ma', 'a.']` · `examples from emma: 5 (len + 1)` · `total bigrams (sum of len+1): 228146` | 0.7 s | (under load) |
| bigram-counts | dict counting with `<S>`/`<E>` | `python labs/bigram-counts.py` | `distinct bigrams in dict: 627` · `top 5: [(('n', '<E>'), 6763), (('a', '<E>'), 6640), (('a', 'n'), 5438), (('<S>', 'a'), 4410), (('e', '<E>'), 3983)]` | 1.6 s | matches notebook; (under load) |
| bigram-counts | the 27×27 tensor N | `python labs/bigram-counts.py` | `stoi a, z, .: 1 26 0` · `N.shape, dtype: (27, 27) torch.int32` · `N[0, :5] (.. .a .b .c .d): [0, 4410, 1306, 1542, 1690]` · `N[stoi['e'], stoi['m']] (em): 769` · `N[stoi['m'], 0] (m.): 516` · `N[0, stoi['m']] (.m): 2538` · `N[0].sum() (names): 32033` · `N.sum() (bigrams): 228146` | 1.6 s | `--plot` saves labs/bigram-counts.png (2.3 s); (under load) |
| count-matrix-visualised | the 'e' row | `python labs/count-matrix-visualised.py` | `'e' row, top 5 next chars: [('e.', 3983.0), ('el', 3248.0), ('en', 2675.0), ('er', 1958.0), ('ee', 1271.0)]` · `'e' row total: 20423` | 1.6 s | (under load) |
| count-matrix-visualised | '.' row / column, top bigrams, zeros | `python labs/count-matrix-visualised.py` | `'.' row (first letters), top 5: [('.a', 4410.0), ('.k', 2963.0), ('.m', 2538.0), ('.j', 2422.0), ('.s', 2055.0)]` · `'.' column (last letters), top 5: [('n.', 6763.0), ('a.', 6640.0), ('e.', 3983.0), ('i.', 2489.0), ('h.', 2409.0)]` · `top 5 bigrams overall: [('n.', 6763.0), ('a.', 6640.0), ('an', 5438.0), ('.a', 4410.0), ('e.', 3983.0)]` · `N[0,0] (empty name): 0` · `cells that are zero: 102 of 729` · `max cell: 6763` | 1.6 s | `--plot` saves the labelled 27×27 imshow (3.2 s); (under load) |
| sampling-multinomial | row 0 as probabilities | `python labs/sampling-multinomial.py` | `row 0 as probs: p['.'] = 0.0000, p['a'] = 0.1377, sum = 1.0000` | 1.5 s | matches notebook; (under load) |
| sampling-multinomial | rand(3) demo | `python labs/sampling-multinomial.py` | `torch.rand(3) normalised: [0.6064, 0.3033, 0.0903]` · `100 draws from it, counts of 0/1/2: [61, 33, 6]` | 1.5 s | rand(3) matches the notebook; (under load) |
| sampling-multinomial | seeded samples | `python labs/sampling-multinomial.py` | `first draw from row 0: 3 'c'` · `bigram samples : ['cexze.', 'momasurailezitynn.', 'konimittain.', 'llayn.', 'ka.']` · `uniform samples: ['cexzm.', 'zoglkurkicqzktyhwmvmzimjttainrlkfukzkktda.', 'sfcxvpubjtbhrmgotzx.', 'iczixqctvujkwptedogkkjemkmmsidguenkbvgynywftbspmhwcivgbvtahlvsu.', 'dsdxxblnwglhpyiw.']` · `bigram: mean length over 1000 samples: 6.31 (real names: 6.12)` · `bigram: 1-letter names in 1000 samples: 117` | 1.5 s | video says first draw 'm' (13) and names `mor.`, `axx.`, `minaymoryles.`, `kondlaisah.`, `anchshizarie.`; torch 2.14 gives 'c' (3) and the names shown; (under load) |
| broadcasting-keepdim-trap | sum shapes | `python labs/broadcasting-keepdim-trap.py` | `P.sum(0, keepdim=True).shape: (1, 27)` · `P.sum(1, keepdim=True).shape: (27, 1)` · `P.sum(1).shape: (27,)` · `P.sum().shape: ()` | 1.8 s | (under load) |
| broadcasting-keepdim-trap | the bug | `python labs/broadcasting-keepdim-trap.py` | `correct: P[0].sum() = 1.0000` · `correct: all rows sum to 1: True` · `bug:     P[0].sum() = 7.0225` · `bug:     P[:, 0].sum() = 1.0000` · `bug:     rows that sum to 1: 0 of 27` · `in-place /= matches: True` | 1.8 s | video says "about 7"; (under load) |
| likelihood-and-nll | 'emma' bigram by bigram | `python labs/likelihood-and-nll.py` | `.e: prob=0.0478 logprob=-3.0408` · `em: prob=0.0377 logprob=-3.2793` · `mm: prob=0.0253 logprob=-3.6772` · `ma: prob=0.3899 logprob=-0.9418` · `a.: prob=0.1960 logprob=-1.6299` · `uniform baseline 1/27 = 0.0370` | 2.1 s | (under load) |
| likelihood-and-nll | first 3 words | `python labs/likelihood-and-nll.py` | `first 3 words: log_likelihood=-38.7856 n=16 nll=38.7856 avg nll=2.4241` | 2.1 s | video says "-38" and "2.4"; (under load) |
| likelihood-and-nll | full training set | `python labs/likelihood-and-nll.py` | `all words: log_likelihood=-559891.75 n=228146 avg nll=2.4541` · `all words, float64: avg nll=2.454014` | 2.1 s | Karpathy's loop sums float32 tensors, so torch 2.14 prints 2.4541; the exact (float64) value is 2.454014, matching facts' 2.4540 and map-A's log-likelihood -559,873.59. Notebook's saved cell prints 2.4765 with `log_likelihood=tensor(-564996.8125, grad_fn=<AddBackward0>)`: the grad_fn suggests P was not the count matrix when that cell last ran [inference]. (under load) |
| likelihood-and-nll | andrej / andrejq | `python labs/likelihood-and-nll.py` | `andrej: avg nll=3.0391` · `andrejq: avg nll=inf` · `P[j, q] = 0.0 \| count N[j, q] = 0` | 2.1 s | video: "about 3"; (under load) |
| smoothing | +0 vs +1 fake counts | `python labs/smoothing.py` | `fake counts +0: P[j,q]=0.000000  andrejq=inf  all words=2.4540` · `fake counts +1: P[j,q]=0.000342  andrejq=3.4834  all words=2.4546` | 3.9 s | video says "roughly 2.47" and the notebook prints 2.4765 for the smoothed model; torch 2.14 gives 2.4546 (this lab sums python floats, so 2.4540 unsmoothed, not 2.4541); (under load) |
| smoothing | more fake counts | `python labs/smoothing.py` | `+10: 2.4621` · `+100: 2.5375` · `+1000: 2.8523` · `+100000: 3.2837` · `uniform, log(27) = 3.2958` | 3.9 s | not in the video; shows the smoothing knob moving toward uniform; (under load) |
| one-hot-and-linear-layer | xs / ys and dtypes | `python labs/one-hot-and-linear-layer.py` | `xs: [0, 5, 13, 13, 1] ys: [5, 13, 13, 1, 0]` · `torch.tensor dtype: torch.int64 \| torch.Tensor dtype: torch.float32` | 0.8 s | (under load) |
| one-hot-and-linear-layer | one-hot | `python labs/one-hot-and-linear-layer.py` | `one_hot without num_classes, shape: (5, 14)` · `one_hot dtype: torch.int64` · `xenc.shape: (5, 27) torch.float32` · `xenc[1] (the e): [0, 0, 0, 0, 0, 1, 0, ... 0]` (27 entries, 1 at index 5) | 0.8 s | (under load) |
| one-hot-and-linear-layer | xenc @ W | `python labs/one-hot-and-linear-layer.py` | `one neuron, (xenc @ W).shape: (5, 1)` · `27 neurons, (xenc @ W).shape: (5, 27)` · `(xenc @ W)[3, 13] = 0.0379` · `(xenc[3] * W[:, 13]).sum() = 0.0379` · `W[13, 13] (row xs[3]=13, col 13) = 0.0379` | 0.8 s | W seeded with 2147483647; (under load) |
| softmax-as-exp-normalise | logits → counts → probs | `python labs/softmax-as-exp-normalise.py` | `logits[0, :5]: [1.5674, -0.2373, -0.0274, -1.1008, 0.2859]` · `counts[0, :5]: [4.794, 0.7888, 0.973, 0.3326, 1.3309]` · `probs[0, :5] : [0.0607, 0.01, 0.0123, 0.0042, 0.0168]` · `probs.shape: (5, 27) \| row sums: [1.0, 1.0, 1.0, 1.0, 1.0]` · `matches torch.softmax: True` | 0.8 s | (under load) |
| softmax-as-exp-normalise | per-example NLL on 'emma' | `python labs/softmax-as-exp-normalise.py` | `.e (indexes 0,5): p=0.0123 logp=-4.3993 nll=4.3993` · `em (indexes 5,13): p=0.0181 logp=-4.0146 nll=4.0146` · `mm (indexes 13,13): p=0.0267 logp=-3.6234 nll=3.6234` · `ma (indexes 13,1): p=0.0737 logp=-2.6081 nll=2.6081` · `a. (indexes 1,0): p=0.0150 logp=-4.2012 nll=4.2012` | 0.8 s | (under load) |
| softmax-as-exp-normalise | the loss three ways | `python labs/softmax-as-exp-normalise.py` | `average nll (loop):   3.7693` · `vectorised loss:      3.7693` · `F.cross_entropy:      3.7693` · `uniform guess log(27) = 3.2958` | 0.8 s | video says "3.76"; notebook cell 52 prints 3.6891887 (stale state); torch 2.14 gives 3.7693. Note the random init is worse than uniform guessing (3.2958); (under load) |
| train-neural-bigram | 3 steps on 'emma', lr 0.1 | `python labs/train-neural-bigram.py` | `step 0: loss 3.7693` · `step 1: loss 3.7492` · `step 2: loss 3.7292` | 8.3 s | video says 3.76 → 3.74 → 3.72; (under load) |
| train-neural-bigram | all words, lr 50, no reg | `python labs/train-neural-bigram.py` | `number of examples: 228146` · `step 0: loss 3.7590` · `step 1: loss 3.3711` · `step 9: loss 2.7115` · `step 49: loss 2.4971` · `step 99: loss 2.4729` | 8.3 s | whole lab 8.3 s (runs both trainings); video says about 2.47, then "2.46, 2.45" with more steps; (under load) |
| train-neural-bigram | all words, lr 50, + 0.01·(W²).mean() | `python labs/train-neural-bigram.py` | `step 0: loss 3.7686` · `step 1: loss 3.3788` · `step 9: loss 2.7188` · `step 49: loss 2.5108` · `step 99: loss 2.4901` | 8.3 s | this is the notebook's loop (with reg); its saved output shows 2.4818 [nb], likely more steps; (under load) |
| train-neural-bigram | 200 steps | `python labs/train-neural-bigram.py --iters 200` | no reg: `step 199: loss 2.4624` · reg: `step 199: loss 2.4830` · `P[qu]  net 0.6143  counting(N+1) 0.6923` | 17.5 s | loss still includes the reg term; (under load) |
| train-neural-bigram | note 1 + net vs counting | `python labs/train-neural-bigram.py` | `one_hot(5) @ W == W[5]: True` · `P[.a]  net 0.1373  counting(N+1) 0.1376` · `P[e.]  net 0.1942  counting(N+1) 0.1948` · `P[n.]  net 0.3674  counting(N+1) 0.3685` · `P[qu]  net 0.3548  counting(N+1) 0.6923` | 8.3 s | common rows already match after 100 steps; rare rows lag (q starts only 272 of 228,146 bigrams, qu = 206), since their gradient is small [inference]; (under load) |
| train-neural-bigram | samples, net vs counting | `python labs/train-neural-bigram.py` | `neural net samples: ['cexze.', 'momasurailezityha.', 'konimittain.', 'llayn.', 'ka.']` · `counting (N+1) samples: ['cexze.', 'momasurailezitynn.', 'konimittain.', 'llayn.', 'ka.']` | 8.3 s | video/notebook: net `mor.`, `axx.`, `minaymoryles.`, `kondlaisah.`, `anchthizarie.` vs counting `anchshizarie.` (last name differs); torch 2.14 gives the names shown, differing in the 2nd name; (under load) |

Long runs (> 3 min)
- None. The longest lab is `python labs/train-neural-bigram.py --iters 200` at about 18 s (under load); the default 100-step run is about 8 s.


## Card list (z2h-3-bigram)
| # | File | Title | Group | Brief |
|---|---|---|---|---|
| 1 | names-dataset.html | The dataset: 32,033 names | Counting | L2 chapters 00:00:00, 00:03:03, 00:06:24 (map-A). Lengths (min 2, max 15), what a bigram is ('emma' → .e em mm ma a.), how many examples one word contributes. Build: open names.txt, print len(words), min/max length, the bigrams of 'emma'. |
| 2 | bigram-counts.html | Counting bigrams in a 27×27 tensor | Counting | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L2 00:03:03 - 00:20:54). Gist: 32,033 names become 228,146 bigrams with '.' as the start/end token, stored in a 27×27 count table N. Build step: Load names.txt; build `stoi/itos`; fill `N[ix1,ix2] += 1`; `imshow` it. Covers 00:09:24, 00:12:45, 00:20:54 (dict counting, the 2D tensor, replacing <S>/<E> with a single '.'). Leave the imshow picture itself to count-matrix-visualised. |
| 3 | count-matrix-visualised.html | Reading the count matrix | Counting | L2 00:18:19 (visualizing the bigram tensor) + the '.' row/column after 00:20:54. Show a small excerpt of N (the 'e' row, top counts) as a flat bar chart with real counts from labs; the full imshow as a description. Most common bigrams (compute top 5). |
| 4 | sampling-multinomial.html | Sampling names from the table | Counting | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L2 00:24:02). Gist: Normalise a row to probabilities, draw with `torch.multinomial`, repeat until '.'. Bigram output is name-ish but bad; uniform is worse. Build step: Seeded generator; the sampling loop; compare against `torch.ones(27)/27`. |
| 5 | broadcasting-keepdim-trap.html | Broadcasting and the keepdim trap | Judging it | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L2 00:36:17). Gist: `P /= P.sum(1, keepdim=True)` normalises rows. Dropping keepdim silently normalises columns (row 0 sums to about 7). Build step: Print shapes for `sum(0)`, `sum(1)`, with and without keepdim; show the bug; check `P[0].sum()`. |
| 6 | likelihood-and-nll.html | Negative log likelihood: one number for 'how good' | Judging it | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L2 00:50:14, 01:00:50). Gist: Product of probs, to sum of logs, negated and averaged, gives 2.454 on the train set. A zero count gives infinite loss ("andrejq"); fix with +1 fake counts. Build step: Loop the NLL over all words; try "andrej" and "andrejq"; switch to `(N+1)`. Loss part only (00:50:14): likelihood, log likelihood, NLL, average NLL 2.4540 on the full set; 'andrej' vs 'andrejq' (inf). |
| 7 | smoothing.html | Model smoothing with fake counts | Judging it | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L2 00:50:14, 01:00:50). Gist: Product of probs, to sum of logs, negated and averaged, gives 2.454 on the train set. A zero count gives infinite loss ("andrejq"); fix with +1 fake counts. Build step: Loop the NLL over all words; try "andrej" and "andrejq"; switch to `(N+1)`. Smoothing part only (01:00:50). Note the smoothed-loss discrepancy in map-A (video 'about 2.47', notebook 2.4765, labs re-run 2.4546): use the labs value and say so. |
| 8 | one-hot-and-linear-layer.html | One-hot inputs and a linear layer | The neural net version | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L2 01:02:57 - 01:26:17). Gist: xs/ys from "emma"; one-hot into `@ W` (27×27) gives logits; exp and normalise (softmax) give probs. The initial loss on "emma" is 3.77. Build step: Build xs/ys; `F.one_hot(...).float()`; logits/counts/probs; per-example NLL printout. Chapters 01:02:57–01:13:53: the (xs, ys) dataset, F.one_hot(...).float(), xenc @ W. Softmax goes on the next card. |
| 9 | softmax-as-exp-normalise.html | Logits → counts → probabilities: the softmax | The neural net version | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L2 01:02:57 - 01:26:17). Gist: xs/ys from "emma"; one-hot into `@ W` (27×27) gives logits; exp and normalise (softmax) give probs. The initial loss on "emma" is 3.77. Build step: Build xs/ys; `F.one_hot(...).float()`; logits/counts/probs; per-example NLL printout. Chapters 01:18:46–01:35:49: softmax, the per-example NLL walk-through on 'emma' (3.7693), the vectorised loss. |
| 10 | train-neural-bigram.html | Gradient descent finds the counting answer | The neural net version | Map: tasks/z2h-kit/map-A-micrograd-bigram.md (chapters L2 01:35:49 - 01:56:16). Gist: Vectorised loss, `W.grad=None`, `backward`, `W.data += -50*W.grad` on all 228k bigrams reaches about 2.46-2.48. one_hot@W is a row lookup (W≈log N); L2 reg acts as smoothing; samples match. Build step: Full training loop with `requires_grad=True`, lr 50, plus `0.01*(W**2).mean()`; sample from the net. Include note 1 (one-hot @ W just selects a row of W) and note 2 (regularisation = smoothing), and sampling from the net. |
