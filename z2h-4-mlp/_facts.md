# Shared facts: makemore 2: the MLP (z2h-4-mlp)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
Lecture 3 of Zero to Hero: follow Bengio et al. (2003) to give each character a learned embedding, feed three characters of context through a tanh hidden layer, and learn the machine-learning basics of minibatches, learning rates and train/dev/test splits along the way.
- Lecture: L3. 12 concept cards plus `big-ideas.html` and `_overview.html` (written after the concept cards).
- **Running example / conventions for this course:** block_size 3; C (27×2 then 27×10), W1 (6×100 then 30×200); the 32 examples from the first five words; 3,481 params → final 11,897 params with train 2.1260 / dev 2.1701 (map-B numbers cheat-sheet). Name the split-size discrepancy between notebooks (map-B) where splits appear.

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
| why-counts-explode | possible vs seen contexts in names.txt | `python labs/why-counts-explode.py` | `32033 names`; `n=1: 27**1 = 27 possible contexts \| seen 27 (100.0%) \| seen <10 times: 0 \| examples per possible row: 8449.85`; `n=2: 27**2 = 729 ... seen 602 ( 82.6%) \| seen <10 times: 148 \| ... 312.96`; `n=3: 27**3 = 19,683 ... seen 5,686 ( 28.9%) \| seen <10 times: 3,374 \| ... 11.59`; `n=4: 27**4 = 531,441 ... seen 22,135 ( 4.2%) \| seen <10 times: 18,749 \| ... 0.43` | 1.8 s | contexts are the padded rolling-window contexts (so `.` counts as a char); video says "~20,000" for n=3 (exactly 19,683). (under load) |
| bengio-embeddings | — | pencil and paper | — | — | no lab: read Fig. 1 and sketch C → tanh → softmax |
| rolling-window-dataset | emma's rows, 5-word dataset shape | `python labs/rolling-window-dataset.py` | `... ---> e` / `..e ---> m` / `.em ---> m` / `emm ---> a` / `mma ---> .`; `first 5 words: ['emma', 'olivia', 'ava', 'isabella', 'sophia']`; `X.shape (32, 3) torch.int64 \| Y.shape (32,) torch.int64`; `X.shape == (32, 3): True`; `rows whose context is "..." -> ['e', 'o', 'a', 'i', 's']`; `all words: X.shape (228146, 3)` | 0.9 s | matches video (32 examples; "228,000"). (under load) |
| embedding-lookup | C[5] vs one-hot matmul, C[X] | `python labs/embedding-lookup.py` | `C[5] tensor([-0.4713, 0.7868])`; `F.one_hot(5, 27).float()@C tensor([-0.4713, 0.7868])`; `equal: True`; `one_hot(int) -> TypeError: one_hot(): argument 'input' (position 1) must be Tensor, not int`; `long @ float -> RuntimeError: expected m1 and m2 to have the same dtype, but got: long long != float`; `C[[5, 6, 7]].shape (3, 2)`; `X.shape (32, 3) -> emb = C[X] .shape (32, 3, 2)`; `X[13, 2] = 1 \| emb[13, 2] = tensor([-0.0274, -1.1008]) \| C[X[13, 2]] = tensor([-0.0274, -1.1008])` | 0.7 s | C seeded with 2147483647 (video's C here is unseeded, so its values differ). (under load) |
| view-storage | strides, cat vs view | `python labs/view-storage.py` | `a.view(2, 9): stride (9, 1), same storage: True`; `a.view(9, 2): stride (2, 1), same storage: True`; `a.view(3, 3, 2): stride (6, 2, 1), same storage: True`; `storage: 144 bytes = 18 int64s`; `emb.shape (32, 3, 2) -> view(-1, 6).shape (32, 6)`; `cat == cat(unbind) == view: True`; `view shares emb memory: True \| cat shares it: False`; `row 0 of view(-1, 6): tensor([ 1.5674, -0.2373, 1.5674, -0.2373, 1.5674, -0.2373])` | 0.7 s | row 0 is `...` so the same C[0] repeats 3 times. (under load) |
| hidden-and-output | shapes and parameter count | `python labs/hidden-and-output.py` | `emb.view(-1, 6) @ W1 : (32, 100) + b1 (100,) -> broadcast as (1, 100) copied down 32 rows`; `h.shape (32, 100) \| h range [-1.0000, 1.0000]`; `fraction of \|h\| > 0.99: 0.327`; `logits.shape (32, 27)`; `C (27, 2) 54` / `W1 (6, 100) 600` / `b1 (100,) 100` / `W2 (100, 27) 2700` / `b2 (27,) 27`; `total parameters: 3481` | 0.7 s | video says "about 3,400"; exact 3,481. (under load) |
| cross-entropy | manual NLL vs F.cross_entropy, overflow | `python labs/cross-entropy.py` | `prob of the correct char, first 5 rows: 1.52e-14 1.28e-12 1.96e-08 3.18e-10 5.68e-12`; `manual loss 17.7697`; `F.cross_entropy 17.7697`; `logits [-2, 3, -3, 0, 5] exp -> 148.4 max; manual probs [0.0008, 0.1184, 0.0003, 0.0059, 0.8746]`; `logits [-100, 3, -3, 0, 5] ... [0.0, 0.1185, 0.0003, 0.0059, 0.8753]`; `logits [-2, 3, -3, 0, 100] exp -> inf max; manual probs [0.0, 0.0, 0.0, 0.0, nan]`; `manual softmax with 100 gives nan: True`; `subtract the max first: [0.0, 0.0, 0.0, 0.0, 1.0]`; `adding a constant changes nothing: True` | 0.7 s | seeded 3,481-param net. Map/video: unseeded first try "loss 17" with correct-char probs ~0.2/0.07; seeded "respectable" version gives 17.7697 here. (under load) |
| overfit-one-batch | 1000 full-batch steps on 32 examples | `python labs/overfit-one-batch.py` | `step 0 loss 17.7697`; `step 1 loss 13.6564`; `step 10 loss 3.9858`; `step 100 loss 0.3354`; `step 500 loss 0.2635`; `step 999 loss 0.2561`; `final loss 0.2561`; `correct: 28/32`; `row 0 context '...': predicted 'i', label 'e'`; `row 5 ... label 'o'`; `row 12 ... label 'a'`; `row 25 ... label 's'` | 0.8 s | lr 0.1. The 4 misses are exactly the `...` rows: that context has 5 labels (e, o, a, i, s) and the net picks `i` (the one it gets right). Map lists only "e, o, a and s". (under load) |
| minibatches | step cost, minibatch training on all 228,146 | `python labs/minibatches.py` | `X.shape (228146, 3)`; `one full-batch step (228,146 rows): 126.0 ms \| one minibatch step (32 rows): 0.119 ms \| ratio 1063x`; `full-set loss at init: 19.5052`; `after 100 steps (lr 0.1): ... full-set loss 3.7658`; `after 1000 ...: 2.6531`; `after 2000 ...: 2.5870`; `after 5000 ...: 2.5227`; `after 10000 ...: 2.5214` | 3.8 s | step times vary under load (another run: 168.2 ms / 0.251 ms, 671x); losses are deterministic. Video: full loss ~2.7 → 2.6 → 2.57 → 2.53. (under load) |
| learning-rate-finder | lr sweep 10^-3..10^0, then decay | `python labs/learning-rate-finder.py` | per 100-step bin: `-3.00..-2.70 loss 18.166`, `-2.70..-2.40 15.222`, `-2.40..-2.10 11.973`, `-2.10..-1.80 9.381`, `-1.80..-1.50 6.960`, `-1.50..-1.20 4.638`, `-1.20..-0.90 3.464`, `-0.90..-0.60 3.221`, `-0.60..-0.30 3.957`, `-0.30..+0.00 5.925`; `lowest 50-step moving average at step 731: exponent -0.80, lr 0.157`; `10000 more steps at lr 0.1: full-set loss 2.5214`; `... 2.4536`; `... 2.4175`; `10000 more steps at lr 0.01: full-set loss 2.3401` | 6.5 s | video picks exponent -1 (lr 0.1); the valley here is flat from about -1.1 to -0.7 and the loss is still falling from training during the sweep, so the minimum lands at -0.8. Video after decay "about 2.3"; here 2.3401 (beats bigram 2.45). `--plot` saves learning-rate-finder.png. (under load) |
| splits-and-fit | split sizes, train/dev at 100 vs 300 hidden | `python labs/splits-and-fit.py` | `words: train 25626 / dev 3203 / test 3204`; `examples: Xtr (182625, 3) / Xdev (22655, 3) / Xte (22866, 3)`; `hidden 100: 3481 params`; `+30000 steps at lr 0.1: train 2.4001 \| dev 2.3988`; `+20000 steps at lr 0.01: train 2.3288 \| dev 2.3279`; `hidden 300: 10281 params`; `+30000 steps at lr 0.1: train 2.4961 \| dev 2.4920`; `+20000 steps at lr 0.01: train 2.2967 \| dev 2.2976` | 15.8 s | Part 2 notebook prints 182441 / 22902 / 22803; torch 2.14 + a single seed-42 shuffle gives 182625 / 22655 / 22866, the same as the Part 3–5 notebooks (shuffling twice does not reproduce Part 2's numbers either). Video 300-hidden: 2.23 / 2.24 after longer training with a halved lr; here 2.2967 / 2.2976 after 50k steps. (under load) |
| embeddings-scale-sample | 2-D embeddings; final 10-D model (quick); samples | `python labs/embeddings-scale-sample.py --quick` | `2-D model (10,281 params): train 2.2967 \| dev 2.2976`; farthest from centre: `'u' (+0.43, +1.56) dist 1.38`, `'q' (-0.98, +0.19) dist 0.89`, `'g' (+0.20, +1.04) dist 0.81`, `'y' (-0.65, -0.15) dist 0.70`, `'.' (+0.44, -0.13) dist 0.69`; nearest neighbours: `a: o (0.16), e (0.22), . (0.41)`, `e: o (0.12), a (0.22), i (0.25)`, `i: e (0.25), y (0.27), o (0.36)`, `o: e (0.12), a (0.16), i (0.36)`, `u: g (0.57), b (0.94), v (1.03)`; `10-D model: 11897 params`; `step 0/20000 minibatch loss 27.8817`; `quick 20000 steps: train 2.3093 \| dev 2.3190`; samples: `careah. aal. hari. kimri. rehty. salays. esrahnen. amerync. kaqui. nerise. jamaiir. kaleig. dham. jore. quinn. srochem. jadii. wantho. dearyxi. jaxee.` | 19.2 s | video: vowels cluster, q and `.` outliers. Here a/e/i/o (+ y) cluster, q is an outlier, but u sits farthest out (near g) and `.` is only 5th. Notebook final (full run, not measured yet): train 2.1260 / dev 2.1701; `--quick` gives 2.3093 / 2.3190. Samples differ from the notebook's (carmahela., jhovi., ...) on torch 2.14 and because the quick model is undertrained. (under load) |

Long runs (> 3 min)
- `python labs/embeddings-scale-sample.py` (full 200,000-step final model): not run; estimated 1 to 1.5 min on the M3 (0.22 ms/step × 200k plus 13 s for the 2-D part). It is under 3 min but listed here because it is the lecture's final model; its train/dev (notebook 2.1260 / 2.1701) and samples need timing and recording on an idle machine.


Serial timings (2026-09-26, one job at a time)

| card | what | command | result | wall time | notes |
|---|---|---|---|---|---|
| embeddings-scale-sample | serial timing (one job at a time; background apps running) | `python labs/embeddings-scale-sample.py` | `jore.` · `quint.` · `salin.` · `alianni.` · `wazthoniearyxi.` · `jace.` · `pirran.` · `eddeci.` | 50 s | serial |


## Card list (z2h-4-mlp)
| # | File | Title | Group | Brief |
|---|---|---|---|---|
| 1 | why-counts-explode.html | Why a bigger count table fails | The idea | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 1). Gist: Context of n chars means 27ⁿ rows (729, then 19,683), and most rows get too few counts. Build step: Compute `27**n` for n=1..4 and count how many 3-char contexts actually occur in names.txt. |
| 2 | bengio-embeddings.html | Bengio 2003: characters as points in space | The idea | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 2). Gist: Learn a vector per token; similar tokens drift together, which lets the model generalize to unseen phrases. Build step: Read the paper's Fig. 1; sketch C → tanh hidden → softmax. |
| 3 | rolling-window-dataset.html | Names into (context → next char) pairs | Building it | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 3). Gist: `block_size=3`, padded with dots, rolling window; 5 words give 32 examples. Build step: Write `build_dataset`, print `emma`'s 5 rows, check `X.shape == (32,3)`. |
| 4 | embedding-lookup.html | An embedding is just indexing | Building it | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 4). Gist: `C[5] == one_hot(5) @ C`; `C[X]` embeds a whole (32,3) tensor at once. Build step: `C = torch.randn(27,2)`; verify the equality; `emb = C[X]` gives (32,3,2). |
| 5 | view-storage.html | .view: reshaping for free | Building it | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 5). Gist: Tensors are 1-D storage plus strides; `view(-1,6)` concatenates the 3 embeddings with no copy. Build step: `torch.arange(18).view(3,3,2)`; compare `cat(unbind(emb,1),1)` to `emb.view(-1,6)`. |
| 6 | hidden-and-output.html | Hidden tanh layer and output logits | Building it | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 5–6, 8). Gist: `h = tanh(emb.view(-1,6)@W1+b1)` then `logits = h@W2+b2` gives (32,27); about 3.4k params. Build step: Build W1, b1, W2, b2; check broadcasting of `+b1`; count parameters (3,481). |
| 7 | cross-entropy.html | Logits to loss, and why F.cross_entropy | Building it | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 7, 9). Gist: Manual exp→normalize→pick→−log→mean gives 17; F.cross_entropy is the same number but fused, simpler to backprop, and overflow-safe. Build step: Compute the loss by hand, then `F.cross_entropy`; try logits containing 100 to see `nan`. |
| 8 | overfit-one-batch.html | First training loop: overfit 32 examples | Training it | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 10). Gist: Zero the grads, backward, `p.data += -lr*p.grad`; the loss goes very low but not 0 because `...` has many answers. Build step: Run 1000 steps on the 32 examples; inspect `logits.max(1)` against Y. |
| 9 | minibatches.html | Minibatches: noisy but fast | Training it | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 11). Gist: 228k examples are too slow per step; 32 random rows give an approximate gradient and far more steps. Build step: Add `ix = torch.randint(0, X.shape[0], (32,))`; watch the full-set loss reach about 2.5. |
| 10 | learning-rate-finder.html | Finding the learning rate, then decaying it | Training it | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 12). Gist: Sweep lr from 10⁻³ to 10⁰ exponentially, pick the valley (0.1), train, then drop to 0.01; about 2.3, which beats the bigram's 2.45. Build step: `lrs = 10**torch.linspace(-3,0,1000)`; plot loss against exponent. |
| 11 | splits-and-fit.html | Train/dev/test and under- vs overfitting | Training it | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 13–14). Gist: 80/10/10; train≈dev means underfitting, so grow the model; 300 hidden units alone barely helps. Build step: Split with seed 42; evaluate `Xdev`; retrain with 300 hidden units. |
| 12 | embeddings-scale-sample.html | Seeing the embeddings, scaling up, sampling | Training it | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 15–18). Gist: 2-D embeddings cluster vowels (q and `.` are outliers); 10-D × 200 hidden gives 11,897 params, train 2.126 / dev 2.170; sample names. Build step: Scatter `C`; set `C=randn(27,10)`, `W1=randn(30,200)`; run 200k steps with decay; run the sampling loop. |
