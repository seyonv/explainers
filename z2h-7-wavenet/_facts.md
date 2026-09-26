# Shared facts: makemore 5: building a WaveNet (z2h-7-wavenet)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
Lecture 6 of Zero to Hero: make the MLP deeper with a tree-shaped structure like DeepMind's WaveNet (2016), fusing context two characters at a time, and learn how torch.nn really works along the way.
- Lecture: L6. 8 concept cards plus `big-ideas.html` and `_overview.html` (written after the concept cards).
- **Running example / conventions for this course:** block_size 8; the 22K-param flat baseline vs the hierarchical net; performance log 2.105 → 2.027 → 2.029 → 2.022 → 1.993; final 76,579 params, train 1.7690 / val 1.9937 (map-B). The 200-vs-300 hidden-unit discrepancy in map-B: go with 200 and note it.

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
| smooth-loss-plot | baseline (3-char context) full run | `python labs/smooth-loss-plot.py` | `parameters: 12097` · `0/ 200000: 3.2966` · `trained 200000 steps in 72 s` · `train 2.0583` · `val 2.1065` | 73 s | (under load). Notebook log: "original (3 character context + 200 hidden neurons, 12K params): train 2.058, val 2.105"; torch 2.14 gives train 2.0583, val 2.1065 |
| smooth-loss-plot | raw vs smoothed curve | same | `lossi: 200000 points -> smoothed: 200 points` · `raw spread over the last 1000 steps: min 0.137, max 0.451 (log10 loss)` · `smoothed log10 loss, every 10 windows: 0.406 0.352 0.346 0.344 0.339 0.339 0.337 0.337 0.335 0.333 0.332 0.335 0.334 0.332 0.330 0.320 0.320 0.317 0.316 0.317` | (same run) | (under load) |
| smooth-loss-plot | the lr-decay drop | same | `lr decay at step 150000: mean of 10 windows before 0.331 -> after 0.319 (loss 2.142 -> 2.084)` | (same run) | (under load) |
| smooth-loss-plot | quick run | `python labs/smooth-loss-plot.py --quick --plot` | `smoothed: 20 points` · `lr decay at step 15000: mean of 1 windows before 0.349 -> after 0.339 (loss 2.234 -> 2.181)` · `train 2.1572` · `val 2.1709` · `saved .../loss-raw.png`, `saved .../loss-smooth.png` | 8.5 s | (under load) |
| torch-nn-containers | the Sequential model | `python labs/torch-nn-containers.py` | `layers: Embedding -> Flatten -> Linear -> BatchNorm1d -> Tanh -> Linear` · `parameters: 12097 = 270 + 6000 + 200 + 200 + 5400 + 27` · `Embedding out (4, 3, 10)` · `Flatten out (4, 30)` · `Linear out (4, 200)` · `BatchNorm1d out (4, 200)` · `Tanh out (4, 200)` · `Linear out (4, 27)` | 9.9 s | (under load). Map says "about 12k" params: exactly 12,097 |
| torch-nn-containers | eval-mode samples (20k-step model) | same | `val 2.1709` · `eval mode samples : ama. ele. lia. alden. xalisse. deza. smartri. rizleigh.` | (same run) | (under load). Trains only 20k steps so it runs fast |
| torch-nn-containers | the forgotten `training=False` bug | same | `batch of 1, the variance BatchNorm1d computes: x.var(0)[:3] = [nan, nan, nan]` · `logits with training=True: [nan, nan, nan, nan]` · `train mode sampling crashes: probability tensor contains either inf, nan or element < 0` (torch prints inf and nan in backticks) · `and the running stats are poisoned too: running_var has nan: True` · `eval mode sampling now crashes too: ...` | (same run) | (under load). The video shows garbage output; on torch 2.14 `torch.multinomial` raises an error on the nan probabilities instead, and the train-mode forward writes nan into running_var, which breaks eval mode too |
| more-context-baseline | the block_size-8 rows | `python labs/more-context-baseline.py` | `Xtr (182625, 8) Xdev (22655, 8) Xte (22866, 8)` · `........ --> y` · `.......y --> u` · `......yu --> h` · `.....yuh --> e` · … · `.diondre --> .` · `........ --> x` | 114 s | (under load). Same as the notebook |
| more-context-baseline | flat model, 8 chars, full run | same | `parameters: 22097 (block_size 3 had 12097)` · `0/ 200000: 3.2847` · `trained 200000 steps in 112 s` · `train 1.9163` · `val 2.0342` | (same run) | (under load). Notebook log: "context: 3 -> 8 (22K params): train 1.918, val 2.027"; torch 2.14 gives train 1.9163, val 2.0342. The notebook comment's `n_hidden = 300` would give 32,997 params, so 200 is right (22,097) |
| more-context-baseline | quick run | `python labs/more-context-baseline.py --quick` | `train 2.0770` · `val 2.1074` | 8.5 s | (under load) |
| flatten-consecutive | batched matmul | `python labs/flatten-consecutive.py` | `(4, 80) @ (80, 200) -> (4, 200)` · `(4, 5, 80) @ (80, 200) -> (4, 5, 200)` · `(4, 4, 20) @ (20, 200) -> (4, 4, 200)` | 1.6 s | (under load) |
| flatten-consecutive | grouping pairs | same | `Xb (4, 8) -> Embedding (4, 8, 10) -> Flatten (4, 80)` · `e.view(4, 4, 20) == cat([e[:, ::2], e[:, 1::2]], 2): True` · `FlattenConsecutive(2) (4, 4, 20) \| FlattenConsecutive(8) (4, 80) (squeezed)` | (same run) | (under load) |
| flatten-consecutive | 3-level shape walk (n_embd 10, n_hidden 200) | same | `Embedding : (4, 8, 10)` · `FlattenConsecutive : (4, 4, 20)` · `Linear : (4, 4, 200)` · `BatchNorm1d : (4, 4, 200)` · `Tanh : (4, 4, 200)` · `FlattenConsecutive : (4, 2, 400)` · `Linear : (4, 2, 200)` · … · `FlattenConsecutive : (4, 400)` · `Linear : (4, 200)` · … · `Linear : (4, 27)` | (same run) | (under load). Matches the map's walk-through |
| flatten-consecutive | param counts | same | `parameters, n_hidden 200: 170897 \| n_hidden 68: 22397` | (same run) | (under load). 22,397 confirms the map's [DERIVED] count |
| batchnorm-3d-bug | the shapes | `python labs/batchnorm-3d-bug.py` | `input (32, 4, 68): x.mean(0) -> (1, 4, 68) \| x.mean((0,1)) -> (1, 1, 68)` · buggy: `model.layers[3].running_mean.shape = (1, 4, 68)`, `layers[7] = (1, 2, 68)`, `layers[11] = (1, 68)` · fixed: `(1, 1, 68)`, `(1, 1, 68)`, `(1, 68)` | 907 s | (under load, extreme: load averages 100–170 during the run; 441 s user CPU). `parameters: 22397` for both |
| batchnorm-3d-bug | buggy vs fixed, full runs | same | `buggy: reduce over 0 only: train 1.941 val 2.027` · `fixed: reduce over (0,1): train 1.911 val 2.020` (4 decimals: 1.9411 / 2.0265 and 1.9110 / 2.0202) | (same run) | (under load). Notebook: buggy 1.941 / 2.029, fixed 1.912 / 2.022; torch 2.14 gives 1.941 / 2.027 and 1.911 / 2.020. Same 0.007 val gain from the fix |
| batchnorm-3d-bug | quick runs | `python labs/batchnorm-3d-bug.py --quick` | `buggy: reduce over 0 only: train 2.086 val 2.107` · `fixed: reduce over (0,1): train 2.072 val 2.097` | 39 s | (under load) |
| scale-up | final model, quick | `python labs/scale-up.py --quick` | `parameters: 76579` · `0/ 20000: 3.3167` · `10000/ 20000: 2.0576` · `trained 20000 steps in 19 s` · `train 1.9930` · `val 2.0478` | 22.6 s | (under load). --quick only (20k steps, lr drop at 15k). Step 0 (3.3167) and step 10,000 (2.0576) equal the notebook's, so the full run follows the notebook's random stream up to at least there; notebook final train 1.7690 / val 1.9937 (full run not measured yet, see Long runs) |
| scale-up | samples (quick) | same | `samples: cadlee. kemia. haemy. peb. derrki. kainon. garshi. jayan. oghtlanoren. garren. tanick. tumb. eleneas. jeniel. derelyn. shur. maniyah. aliyam. andhai. mylync.` | (same run) | (under load). From the 20k-step model; the notebook's (arlij., chetta., heago., ...) come from the full run |
| convolutions-and-workflow | the 8 rows of one name | `python labs/convolutions-and-workflow.py` | `........ --> d` · `.......d --> i` · `......di --> o` · `.....dio --> n` · `....dion --> d` · `...diond --> r` · `..diondr --> e` · `.diondre --> .` | 1.7 s | (under load). The name in Xtr[7:15] is "diondre", not "DeAndre" as the card list says |
| convolutions-and-workflow | forward in a loop | same | `one example: (1, 27)` · `8 examples in a Python loop: (8, 27)` · `same as one batched call: True` | (same run) | (under load). Untrained model in eval mode |
| convolutions-and-workflow | repeated tree nodes | same | `tree nodes computed by the loop: 56; distinct (level, position) nodes: 34; computed more than once: 22` · `level 1 (covers 2 chars): 14 distinct nodes, 32 computed` · `level 2 (covers 4 chars): 12 distinct nodes, 16 computed` · `level 3 (covers 8 chars): 8 distinct nodes, 8 computed` · `e.g. the level-1 node for 'on' is recomputed by examples [3, 5, 7]` | (same run) | (under load). My own count (nodes labelled by level and start position in the padded name), not from the video |

Long runs (> 3 min)
- `python labs/scale-up.py` (the lecture's final model, 200k steps): not run. The 20k quick run took 13–19 s of training, so expect about 2.5–4 min on an idle M3. It should land near train 1.7690 / val 1.9937, because its first 10,000 steps follow the notebook's random stream.
- `python labs/batchnorm-3d-bug.py` (2 × 200k steps): already run in full (results above): 907 s wall at load averages of 100–170 with 441 s of user CPU. Expect about 6–8 min on an idle M3.


Serial timings (2026-09-26, one job at a time)

| card | what | command | result | wall time | notes |
|---|---|---|---|---|---|
| scale-up | serial timing (one job at a time; background apps running) | `python labs/scale-up.py` | `160000/ 200000: 1.8806` · `170000/ 200000: 1.6266` · `180000/ 200000: 1.6476` · `190000/ 200000: 1.8555` · `trained 200000 steps in 144 s` · `train 1.7690` · `val 1.9937` · `samples: arlij. chetta. heago. rocklei. hendrix. jamylie. broxin. denish. anslibt. marianah. astavia. annayve. aniah. jayce. nodiel. remita. niyelle. jaylene. a` | 147 s | serial |
| batchnorm-3d-bug | serial timing (one job at a time; background apps running) | `python labs/batchnorm-3d-bug.py` | `model.layers[3].running_mean.shape = (1, 1, 68)` · `model.layers[7].running_mean.shape = (1, 1, 68)` · `model.layers[11].running_mean.shape = (1, 68)` · `train 1.9110` · `val 2.0202` · `video/notebook: buggy train 1.941 val 2.029 ¦ fixed train 1.912 val 2.022` · `buggy: reduce over 0 only: train 1.941 val 2.027` · `fixed: reduce over (0,1): train 1.911 val 2.020` | 301 s | serial |


## Card list (z2h-7-wavenet)
| # | File | Title | Group | Brief |
|---|---|---|---|---|
| 1 | smooth-loss-plot.html | Reading a noisy loss curve | Cleanup | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 1–3). Gist: Average each 1000 steps with `view(-1,1000).mean(1)`; the lr-decay drop becomes visible. Build step: Replace `plt.plot(lossi)` with the averaged version. |
| 2 | torch-nn-containers.html | Embedding, Flatten, Sequential: rebuilding torch.nn | Cleanup | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 4, 15). Gist: Every operation becomes a module; `model(Xb)` is the whole forward pass; the BN-in-train-mode nan bug. Build step: Write `Embedding`, `Sequential`; forget `training=False` and watch the samples break. |
| 3 | more-context-baseline.html | Just widen the window: block_size 8 | Going hierarchical | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 6–7). Gist: 8 characters of context in the flat MLP: val 2.105 → 2.027 at about 22K params, but it squashes everything at once. Build step: Set `block_size = 8`; print the `........ --> y` rows; retrain. |
| 4 | wavenet-tree-idea.html | WaveNet's idea: fuse context slowly, in a tree | Going hierarchical | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 5). Gist: Pairs → bigrams → 4-grams → 8-grams (receptive field doubles per layer); dilated convolution is only the fast implementation. Build step: Sketch the WaveNet figure; label the receptive field 2/4/8(/16). |
| 5 | flatten-consecutive.html | Batched matmul and FlattenConsecutive | Going hierarchical | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 8). Gist: `@` treats leading dimensions as batch; view (B,8,10) as (B,4,20) to group pairs; stack 3 levels. Build step: Check that `(4,5,80)@(80,200)` gives `(4,5,200)`; write `FlattenConsecutive`; print layer shapes. |
| 6 | batchnorm-3d-bug.html | The silent BatchNorm bug: reduce over (0, 1) | Going hierarchical | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 9–11). Gist: 3-D input made running_mean (1,4,68); the fix gives 68 channel stats; val 2.029 → 2.022; PyTorch expects NCL. Build step: Inspect `model.layers[3].running_mean.shape`; add the `dim=(0,1)` branch. |
| 7 | scale-up.html | Scaling up: 76,579 params, val 1.993 | Results | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 12, 18). Gist: n_embd 24, n_hidden 128 gives train 1.769, val 1.993; the full performance log; the challenge to beat it. Build step: Train 200k steps (lr 0.1 → 0.01 at 150k); run eval with `training=False`; sample. |
| 8 | convolutions-and-workflow.html | Convolutions, and how deep-learning work actually happens | Results | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 13–14, 16–17). Gist: A convolution slides the same tree over all positions in a kernel and reuses shared nodes; the workflow is docs plus shape babysitting, Jupyter to repo, and you need a harness. Build step: Forward the 8 "DeAndre" rows in a loop (logits (8,27)); note which nodes repeat. |
