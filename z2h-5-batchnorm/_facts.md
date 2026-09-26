# Shared facts: makemore 3: activations, gradients & BatchNorm (z2h-5-batchnorm)

Every card writer reads this file before writing. Reuse these values exactly.

## This course
Lecture 4 of Zero to Hero: look inside the MLP while it trains, fix a bad initial loss and saturated tanh units, derive Kaiming init, add Batch Normalization, and learn the diagnostic plots that tell you if a deep net is healthy.
- Lecture: L4. 11 concept cards plus `big-ideas.html` and `_overview.html` (written after the concept cards).
- **Running example / conventions for this course:** The z2h-4 MLP (block 3, emb 10, hidden 200): initial loss 27 vs expected −ln(1/27) = 3.2958; per-fix dev losses 2.1682 → 2.13 → 2.1027 → 2.1070 → 2.1048 (map-B); Kaiming gain 5/3 → (5/3)/√30 ≈ 0.3; the 6-layer PyTorch-ified net with 47,024 params; update:data ratio target ≈ 1e-3.

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
| expected-initial-loss | uniform-guess loss | `python labs/expected-initial-loss.py` | `expected initial loss -ln(1/27) = 3.2958` | 3 s | (under load) |
| expected-initial-loss | 4-class toy | `python labs/expected-initial-loss.py` | `all zeros (uniform) : logits [0.0, 0.0, 0.0, 0.0] -> loss 1.3863` · `confident and right : logits [5.0, 0.0, 0.0, 0.0] -> loss 0.0200` · `confident and wrong : logits [0.0, 0.0, 5.0, 0.0] -> loss 5.0200` · `randn * 10 : mean loss over 10k draws 10.3656` · `randn * 100: mean loss over 10k draws 103.0973` | 3 s | (under load). Video says 1.38 for the all-zeros toy (= ln 4). |
| expected-initial-loss | starter MLP step-0 loss | `python labs/expected-initial-loss.py` | `starter MLP, step-0 loss: 27.8817` · `logits of example 0 range from -24.4 to 36.4` · `probability given to the correct next char: median 6.08e-13 (uniform would be 0.0370)` | 3 s | (under load). The video says "27"; this is the same run, bit for bit. |
| squash-the-logits | step-0 loss vs output scale | `python labs/squash-the-logits.py --quick` | `W2 * 1.0 : step-0 loss 27.8817` · `W2 * 0.1 , b2 * 0: step-0 loss 4.2326` · `W2 * 0.01, b2 * 0: step-0 loss 3.3221` · `W2 * 0.0 , b2 * 0: step-0 loss 3.2958` | 15 s | (under load). Video: about 4.2 and 3.32, which matches. |
| squash-the-logits | hockey stick + final losses (full) | `python labs/squash-the-logits.py` | `original  : mean loss steps 0-999 5.9756 \| train 2.1268 val 2.1698` · `fix-logits: mean loss steps 0-999 2.5134 \| train 2.0696 val 2.1311` | 8 min 22 s | (under load; heavy, with load average > 100). With a light load it took 3 min 5 s, and about 50 s when idle-ish. The notebook's "original" is train 2.1245 / val 2.1682; torch 2.14 gives 2.1268 / 2.1698 (the notebook's log entry probably came from a slightly different run). "fix-logits" is 2.07 / 2.13 in the log, which matches. |
| squash-the-logits | same, `--quick` | `python labs/squash-the-logits.py --quick` | `original  : ... train 2.3093 val 2.3190  (20000 steps)` · `fix-logits: ... train 2.1832 val 2.2065  (20000 steps)` | 15 s | (under load) |
| saturated-tanh | h at init, W1 x 1 (after fix 1) | `python labs/saturated-tanh.py --quick` | `hpreact range: -19.1 to 18.1` · `h histogram, 10 bins from -1 to 1 (6400 values): [2460, 199, 126, 114, 113, 97, 119, 131, 185, 2856]` · `\|h\| > 0.99: 61.0% of the 32x200 activations` · `mean local gradient (1 - h**2): 0.154` · `dead columns (\|h\|>0.99 for all 32 examples): 0 of 200` · `dead over all 182625 training examples: 0 of 200` | 5 s | (under load). The video says the preacts range from about −5 to 15; the step-0 batch here spans −19.1 to 18.1. |
| saturated-tanh | h at init, W1 x 0.2, b1 x 0.01 | `python labs/saturated-tanh.py --quick` | `hpreact range: -3.8 to 3.4` · `h histogram ...: [783, 663, 558, 482, 525, 509, 570, 582, 763, 965]` · `\|h\| > 0.99: 1.0%` · `mean local gradient (1 - h**2): 0.605` · `dead columns ...: 0 of 200` | 5 s | (under load). The video says "about −1.5 to 1.5"; that is the bulk, and the extremes here are −3.8 / 3.4. |
| saturated-tanh | full run after fix 2 | `python labs/saturated-tanh.py` | `after training fix-tanh (200000 steps, 50s): train 2.0356 val 2.1027` | 59 s | (under load). This matches the notebook log exactly (2.0356 / 2.1027). |
| dead-neurons | values and local grads | `python labs/dead-neurons.py` | x = −4, −2, −0.5, 0.5, 2, 4 · `tanh df 0.001 0.071 0.786 0.786 0.071 0.001` · `sigmoid df 0.018 0.105 0.235 0.235 0.105 0.018` · `relu df 0.000 0.000 0.000 1.000 1.000 1.000` · `leaky_relu df 0.010 0.010 0.010 1.000 1.000 1.000` | 3 s | (under load). Leaky slope 0.01. |
| dead-neurons | flat share of [−5, 5] | `python labs/dead-neurons.py` | `tanh: 40.1%` · `sigmoid: 8.3%` · `relu: 50.0%` · `leaky_relu: 0.0% of [-5, 5] has \|local grad\| < 0.01` | 3 s | (under load) |
| dead-neurons | ReLU killed by one huge-lr step | `python labs/dead-neurons.py` | `relu shock lr 0.1 : all-negative neurons before 0/200, after 0/200 \| mean loss of last 100 steps 2.2806` · `relu shock lr 30 : ... after 23/200 \| ... 2.5262` · `relu shock lr 100 : ... after 188/200 \| ... 2.8470` · `leaky_relu shock lr 30 : ... after 25/200 \| ... 2.4754` | 3 s | (under load). The protocol is 1000 steps at lr 0.1, then 1 step at the shock lr, then 1000 steps at lr 0.1. Leaky ReLU at shock 100 goes to nan, so it is not shown. For Leaky ReLU the "all-negative" neurons still get 1% of the gradient (not dead). This is my own demo, not from the lecture. |
| kaiming-init | x @ w std toy | `python labs/kaiming-init.py --quick` | `x std 1.006` · `y = x @ (w): y std 3.161` · `(w * 5): 15.804` · `(w * 0.2): 0.632` · `(w / sqrt(10)): 1.000` · `relu(x @ w * 1 / sqrt(10)): mean square 0.500` · `relu(x @ w * sqrt(2) / sqrt(10)): mean square 1.000` | 16 s | (under load). Video: about 3, 15 and 0.6, which matches. |
| kaiming-init | gains and W1 scale | `python labs/kaiming-init.py --quick` | `calculate_gain('linear') = 1.0000` · `('relu') = 1.4142` · `('tanh') = 1.6667` · `W1 scale (5/3)/30**0.5 = 0.3043` · `init: hpreact std 1.524, \|h\| > 0.99 on 8.1%` | 16 s | (under load) |
| kaiming-init | full run | `python labs/kaiming-init.py` | `kaiming (200000 steps, 75s): train 2.0377 val 2.1070` | 1 min 19 s | (under load). This matches the notebook log (2.0377 / 2.1070). |
| batchnorm-forward | before/after BN, step 0 | `python labs/batchnorm-forward.py --quick` | `parameters: 12097` · `before BN: column means -1.82..+1.97, column stds 0.63..2.45` · `after  BN: column means 1.1e-07 (max \|.\|), column stds 1.0000..1.0000` · `step-0 loss: 3.3239` · `b1.grad max \|.\|: 4.4e-10` | 17 s | (under load). The 3.3239 step-0 loss matches notebook cell 7. |
| batchnorm-forward | full run (running stats) | `python labs/batchnorm-forward.py` | `batchnorm (200000 steps, 98s): train 2.0674 val 2.1057` · `learned bngain: mean 1.279, range 0.20..3.27; bnbias: range -0.88..+1.64` | 1 min 41 s | (under load). This matches notebook cell 10 (2.0674 / 2.1057). The loss log's "add batch norm layer" row says 2.0668 / 2.1048, and I could not reproduce it with running or calibrated stats (calibrated gives 2.0678 / 2.1056). Use 2.1057 on cards, or say the log differs. |
| batchnorm-inference | parameters vs buffers | `python labs/batchnorm-inference.py --quick` | `trained by backprop (requires_grad): ['C', 'W1', 'W2', 'b2', 'bngain', 'bnbias']` · `buffers (EMA, no grad): ['bnmean_running', 'bnstd_running']` | 11 s | (under load) |
| batchnorm-inference | calibrated vs running (full) | `python labs/batchnorm-inference.py` | `calibrated vs running mean: max \|diff\| 0.0266 (mean values span -3.07..2.94)` · `calibrated vs running std : max \|diff\| 0.0537 (std values span 1.64..3.30)` · `calibrated stats: train 2.0678 val 2.1056` · `running    stats: train 2.0674 val 2.1057` | 1 min 40 s | (under load) |
| batchnorm-inference | batch jitter, one example, momentum (full) | `python labs/batchnorm-inference.py` | `p(correct next char) for training example 0 in 5 batches: [0.0121, 0.0134, 0.0268, 0.0158, 0.013]` · `one example with batch stats : logits contain nan? True` · `one example with running stats: p(correct) = 0.0129` · `momentum 0.1  : running mean vs calibrated, mean \|diff\| 0.0570` · `momentum 0.001: running mean vs calibrated, mean \|diff\| 0.0076` | 1 min 40 s | (under load). The momentum comparison is an EMA of 10,000 batch-32 means on the trained net: 0.1 is 7.5x noisier. This is my own demo of the video's "momentum 0.1 thrashes at batch 32". |
| resnet-and-torch-nn | torch defaults | `python labs/resnet-and-torch-nn.py` | `nn.BatchNorm1d(num_features: int, eps: float = 1e-05, momentum: float \| None = 0.1, affine: bool = True, track_running_stats: bool = True, device=None, dtype=None, *, bias: bool = True)` · `buffers: [('running_mean', (200,)), ('running_var', (200,)), ('num_batches_tracked', ())]` | 2 s | (under load). torch 2.14's BatchNorm1d has an extra keyword-only `bias` argument that is not in the video's signature. |
| resnet-and-torch-nn | nn.Linear init | `python labs/resnet-and-torch-nn.py` | `nn.Linear(30, 200): max \|w\| 0.1825 vs 1/sqrt(30) = 0.1826` · `std(w) 0.1051 vs 1/sqrt(3*30) = 0.1054 (uniform) vs (5/3)/sqrt(30) = 0.3043 (Kaiming tanh)` | 2 s | (under load). The map says nn.Linear uses "gain 1"; its bound is 1/√fan_in, but uniform in ±1/√fan_in has std 1/√(3·fan_in), so its effective gain is 1/√3 ≈ 0.58. |
| resnet-and-torch-nn | motif params; our BN vs torch | `python labs/resnet-and-torch-nn.py` | `Linear(30,200,bias=False) -> BatchNorm1d(200) -> Tanh: 6400 parameters (with bias: 6600)` · `our BN vs nn.BatchNorm1d output: max \|diff\| 0.0588` · `running_var after one step: max \|diff\| 2.4e-07` | 2 s | (under load). The lecture's class normalizes with the unbiased var; torch normalizes with the biased var and stores the unbiased var in running_var. This connects to z2h-6 bessel-correction. |
| pytorchify-layers | counts | `python labs/pytorchify-layers.py` | `parameters with BatchNorm: 47,024` · `parameters without BatchNorm (Linear with bias): 46,497` | 4 s | (under load). The video says "46k" for no-BN; the exact count is 46,497. |
| pytorchify-layers | 1001 steps, eval, samples | `python labs/pytorchify-layers.py` | `step-0 loss 3.2870, step-1000 loss 2.2807` · `train loss (eval mode): 2.4003` · `val loss (eval mode): 2.3982` · `samples: carpah. qarlileif. jmrix. thty. sacansa. jazhnte. dpn. arciigqeiunellaia. chriiv. kalein.` | 4 s | (under load). This matches the notebook exactly, including the samples (the series note says samples differ, but here they reproduce). The notebook's cell 13 uses hidden gain **1.0** (`*= 1.0 #5/3`), not 5/3. |
| activation-gradient-plots | no BN, gain sweep, at init | `python labs/activation-gradient-plots.py` | gain 0.5: `act std 0.41 0.20 0.10 0.05 0.02`, `grad std 1.9e-05 ... 3.1e-04` · gain 1: `act std 0.62 0.48 0.41 0.35 0.32`, `sat % 3.50 0.03 0.06 0.00 0.00` · gain 5/3: `act std 0.75 0.69 0.67 0.66 0.66`, `sat % 20.25 8.38 6.62 5.47 6.13`, `grad std 4.2e-04 4.0e-04 3.7e-04 3.3e-04 3.1e-04` · gain 3: `act std 0.85 0.84 0.84 0.84 0.84`, `sat % 47.66 40.47 42.38 42.00 42.41`, `grad std 1.0e-03 7.4e-04 5.6e-04 4.0e-04 3.1e-04` | 5 s | (under load). Video (gain 5/3): layer 1 about 20% saturated, then std about 0.65 and about 5%, which matches. Saturated means \|t\| > 0.97. |
| activation-gradient-plots | fully linear case | `python labs/activation-gradient-plots.py` | gain 5/3: `act std 1.65 2.72 4.67 7.50 12.78`, `grad std 2.6e-03 1.6e-03 9.5e-04 5.5e-04 3.2e-04` · gain 1: `act std 0.99 0.98 1.01 0.97 0.99`, `grad std 3.2e-04 3.2e-04 3.2e-04 3.1e-04 3.1e-04` | 5 s | (under load) |
| activation-gradient-plots | BN, 1001 steps | `python labs/activation-gradient-plots.py` | gain 1: `act std 0.63 0.64 0.65 0.65 0.65`, `sat % 2.78 2.56 2.25 1.69 1.88`, `grad std 2.6e-03 2.2e-03 2.0e-03 2.0e-03 2.0e-03` · gain 5/3: `act std 0.63 0.64 0.64 0.65 0.65`, `sat % 2.62 2.47 2.16 1.81 1.78`, `grad std 3.7e-03 3.3e-03 3.0e-03 2.7e-03 2.6e-03` | 5 s | (under load). Gain 1 reproduces notebook cells 15 and 16 exactly. The map's BN numbers are the gain-1.0 run. |
| update-to-data-ratio | grad:data ratio | `python labs/update-to-data-ratio.py` | `(27, 10) 8.01e-03` · `(30, 100) 4.88e-02` · `(100, 100) 6.96e-02 / 6.07e-02 / 5.63e-02 / 5.57e-02` · `(100, 27) grad std 1.21e-02 \| grad:data ratio 1.16e-01` | 13 s | (under load). This matches notebook cell 17. |
| update-to-data-ratio | ud (log10) per setting | `python labs/update-to-data-ratio.py` | ud 900-1000 (C, W30x100, 4xW100x100, W100x27): `BN, gain 1, lr 0.1: -3.11 -2.29 -2.14 -2.21 -2.23 -2.26 -1.96` · `BN, gain 1, lr 0.001: -5.70 -4.76 -4.51 -4.58 -4.62 -4.67 -4.44` · `no BN, gain 5/3, lr 0.1: -2.96 -2.52 -2.34 -2.41 -2.42 -2.48 -1.49 \| Tanh sat 14.0%` (step 0 last layer `-0.63`) · `no BN, no fan_in scaling, lr 0.1: -1.75 -2.08 -2.58 -2.93 -3.28 -3.65 -1.22 \| Tanh sat 84.1%` · `BN, gain 0.2, lr 0.1: -3.15 -1.63 -1.67 -1.70 -1.71 -1.71 -1.55` · `BN, no fan_in scaling, lr 0.1: -3.21 -3.74 -4.01 -4.08 -4.13 -4.19 -3.97` · `BN, no fan_in scaling, lr 1.0: -2.08 -2.58 -2.87 -2.93 -2.95 -2.96 -2.68` | 13 s | (under load). Video: lr 0.001 gives about −4 to −5; the current lr gives about −2.5; no fan_in without BN gives −1 to −1.5; BN without fan_in needs about 10x the lr. All four match. BN gain 0.2 keeps saturation at 1.6% but lifts ud to about −1.7 (the video only says "the ratios change"). |

Long runs (> 3 min)
- `python labs/squash-the-logits.py`: two 200k-step runs. It measured 8 min 22 s under heavy load (load average > 100) and 3 min 5 s under lighter load; expect about 1 min on an idle M3.
- Under this load, these also came close: `python labs/batchnorm-forward.py` (1 min 41 s) and `python labs/batchnorm-inference.py` (1 min 40 s). Each is under 1 min when idle (a single 200k run of the BN MLP took 44 s under light load).


Serial timings (2026-09-26, one job at a time)

| card | what | command | result | wall time | notes |
|---|---|---|---|---|---|
| squash-the-logits | serial timing (one job at a time; background apps running) | `python labs/squash-the-logits.py` | `W2 * 1.0 : step-0 loss 27.8817` · `W2 * 0.1 , b2 * 0: step-0 loss 4.2326` · `W2 * 0.01, b2 * 0: step-0 loss 3.3221` · `W2 * 0.0 , b2 * 0: step-0 loss 3.2958` · `original  : mean loss steps 0-999 5.9756 ¦ train 2.1268 val 2.1698  (200000 steps, 34s)` · `fix-logits: mean loss steps 0-999 2.5134 ¦ train 2.0696 val 2.1311  (200000 steps, 21s)` | 56 s | serial |
| batchnorm-forward | serial timing (one job at a time; background apps running) | `python labs/batchnorm-forward.py` | `parameters: 12097 (the Kaiming MLP had 11897: -200 for b1, +400 for bngain/bnbias)` · `before BN: column means -1.82..+1.97, column stds 0.63..2.45` · `after  BN: column means 1.1e-07 (max ¦.¦), column stds 1.0000..1.0000` · `step-0 loss: 3.3239` · `b1.grad max ¦.¦: 4.4e-10  (zero up to float rounding: the mean subtraction removes b1)` · `batchnorm (200000 steps, 41s): train 2.0674 val 2.1057` · `learned bngain: mean 1.279, range 0.20..3.27; bnbias: range -0.88..+1.64` | 42 s | serial |
| batchnorm-inference | serial timing (one job at a time; background apps running) | `python labs/batchnorm-inference.py` | `calibrated vs running std : max ¦diff¦ 0.0537 (std values span 1.64..3.30)` · `calibrated stats: train 2.0678 val 2.1056` · `running    stats: train 2.0674 val 2.1057` · `p(correct next char) for training example 0 in 5 batches: [0.0121, 0.0134, 0.0268, 0.0158, 0.013]` · `one example with batch stats : logits contain nan? True (std of 1 value is nan)` · `one example with running stats: p(correct) = 0.0129` · `momentum 0.1  : running mean vs calibrated, mean ¦diff¦ 0.0570` · `momentum 0.001: running mean vs calibrated, mean ¦diff¦ 0.0076` | 46 s | serial |


## Card list (z2h-5-batchnorm)
| # | File | Title | Group | Brief |
|---|---|---|---|---|
| 1 | expected-initial-loss.html | The loss you should see at step 0 | Initialisation | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 1–3). Gist: Uniform over 27 chars means −ln(1/27) = 3.29, but we get 27, so the init is confidently wrong. Build step: `-torch.tensor(1/27.).log()`; run 1 step and print the loss. |
| 2 | squash-the-logits.html | Fix 1: make the output layer humble | Initialisation | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 3). Gist: `b2*=0`, `W2*=0.01` gives an init loss of about 3.32; the hockey stick disappears; val 2.17 → 2.13. Build step: Edit the init, re-run 1 step, then a full run; compare loss curves. |
| 3 | saturated-tanh.html | Fix 2: tanh units stuck at ±1 | Initialisation | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 4). Gist: `1−t²`≈0 in the tails kills gradients; look for all-white dead-neuron columns; scaling W1 gives val 2.10. Build step: `plt.hist(h.view(-1).tolist(),50)`; `plt.imshow(h.abs()>0.99)`; set `W1*=0.2`. |
| 4 | dead-neurons.html | Dead neurons across nonlinearities | Initialisation | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 4). Gist: tanh, sigmoid and ReLU have flat regions; ReLU neurons can die at init or from a high lr; Leaky ReLU doesn't. Build step: Plot tanh, sigmoid, ReLU and Leaky ReLU with their local gradients. |
| 5 | kaiming-init.html | Kaiming init: gain / √fan_in | Initialisation | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 5). Gist: Var grows by fan_in; divide by √fan_in; tanh gain 5/3 → W1 × (5/3)/√30 ≈ 0.3. Build step: Toy `x@w` std check (1 → 3.16 → 1); replace 0.2 with `(5/3)/30**0.5`. |
| 6 | batchnorm-forward.html | BatchNorm: standardise the preactivations | Batch normalization | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 6). Gist: Batch mean/std normalize, then learnable γ,β scale and shift; drop the bias before BN. Build step: Add `bngain`, `bnbias`; normalize `hpreact` over dim 0. |
| 7 | batchnorm-inference.html | BatchNorm at test time: running statistics | Batch normalization | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 6–7). Gist: The batch couples examples (jitter acts as a regularizer); inference uses calibrated or EMA running mean/std; parameters vs buffers. Build step: Add `bnmean_running` with momentum 0.001 under `no_grad`; compare to the calibrated stats. |
| 8 | resnet-and-torch-nn.html | Real nets: the Linear → BatchNorm → nonlinearity motif | Batch normalization | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 8–9). Gist: ResNet repeats weight→norm→nonlinearity with `bias=False`; PyTorch's `Linear` and `BatchNorm1d` defaults. Build step: Read torchvision `resnet.py` Bottleneck; read the `nn.BatchNorm1d` signature. |
| 9 | pytorchify-layers.html | Rebuilding torch.nn: Linear, BatchNorm1d, Tanh | Diagnostics | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 10). Gist: Classes with `__call__`, `.parameters()`, a `.training` flag and buffers; a 6-layer deep MLP. Build step: Write the three classes; stack 5×(Linear,BN,Tanh)+(Linear,BN); 47,024 params. |
| 10 | activation-gradient-plots.html | Activation and gradient histograms | Diagnostics | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 11–13). Gist: Gain 5/3 gives std about 0.65 and about 5% saturation; wrong gains shrink or explode; a linear stack needs gain 1 and collapses to a single linear map. Build step: Histogram loop over Tanh `.out` and `.out.grad`; try gains 0.5, 1, 5/3, 3; remove the Tanh layers. |
| 11 | update-to-data-ratio.html | The update:data ratio (≈ 1e-3) | Diagnostics | Map: tasks/z2h-kit/map-B-makemore-2-5.md (chapters 14–17). Gist: Track log10(std(lr·grad)/std(w)) to target −3; the last layer is an outlier; BN makes the forward pass gain-proof but not the lr. Build step: Append `ud` each step; plot with a line at −3; try lr=0.001, and BN with no fan_in scaling. |
