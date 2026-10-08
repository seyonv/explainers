# Explainers

**[seyonv.github.io/explainers](https://seyonv.github.io/explainers/)**

My visual notes cards on technical concepts, all on one page. Each card has a worked example with
real, computed numbers. Courses open as a gallery you can page through with ← →.

Everything here is made with two Claude Code skills:
[concept-explainer](https://github.com/seyonv/concept-explainer) makes one card, and
[concept-curriculum](https://github.com/seyonv/concept-curriculum) makes a whole course.

## Run the labs yourself

Many courses have a `labs/` folder of small Python scripts. A lab doesn't contain a model. Most labs
load a real one (Qwen2.5-0.5B or GPT-2) or build a tiny one, then print the numbers its card shows.
The point is to predict the output, run the script, compare, and then change things: a different
prompt, more top-k tokens, a bigger learning rate.

This section covers the two LLM study paths in order: **LLM basics** (look inside a trained model),
then **Build an LLM** (write one yourself with Raschka's book and Karpathy's Zero to Hero).

### Setup, once

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/) (`brew install uv`). One environment at
the root of this repo serves every hub lab:

```bash
cd ~/Desktop/repos/explainers
uv venv --python 3.11
source .venv/bin/activate          # do this in every new terminal
uv pip install torch transformers numpy matplotlib tiktoken datasets graphviz jupyterlab
```

Two rules for running any hub lab:

- **Run it from its course folder**, not from inside `labs/`: `cd text-to-answer`, then
  `python labs/the-whole-route.py`.
- The Qwen labs download the model (about 1 GB) on their first run. After that, put
  `HF_HUB_OFFLINE=1` in front of the command to skip the network check.

Each lab's first lines (its docstring) say what it measures. Open the lab in your editor next to its
card while it runs.

### Part 1 · LLM basics: look inside a trained model (3 days)

Course: `text-to-answer/`. Every lab follows the question "What is the capital of India?" through
Qwen2.5-0.5B. Nothing is trained here.

```bash
cd ~/Desktop/repos/explainers/text-to-answer
HF_HUB_OFFLINE=1 python labs/the-whole-route.py
```

| # | Lab (`text-to-answer/labs/`) | Card | What it shows |
|---|---|---|---|
| 1 | `the-whole-route.py` | the-whole-route | Every stop once: text → 7 tokens → IDs → 896-number vectors → 24 blocks → 151,936 scores → probabilities → "New Delhi" |
| 2 | `embedding-space.py` | embedding-space | The embedding table: lookups, cosine similarity, nearest neighbours, king − man + woman, a 2-D picture |
| 3 | `positions.py` | positions | Layer 0's attention rebuilt by hand: shuffle the words and see what changes with and without RoPE and the causal mask |
| 4 | `ffn-and-experts.py` | ffn-and-experts | Where the weights and the arithmetic go (mostly the feed-forward part), and the same model as a mixture of experts |
| 5 | `decoding.py` | decoding | Greedy, temperature, top-k, top-p and beam search, all run on the same scores |
| 6 | `the-loop.py` | the-loop | Generation as a plain loop, KV-cache timing, and how a chat template changes the first token |

`common.py` is a shared helper (it loads the model and holds `PROMPT`); you don't run it. Edit
`PROMPT` there to send your own question through every lab.

### Part 2 · Build an LLM (5 days)

There are three kinds of code here:

1. **The book's repo.** [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) is
   the spine. Each chapter's code is in `chNN/01_main-chapter-code/chNN.ipynb`.
2. **The hub's Zero to Hero labs.** `z2h-*/labs/` has one script per card. Each course's
   `labs/README.md` lists its run order and what each lab prints.
3. **The hub's gap labs.** `build-an-llm/labs/` covers what the book assumes you know.

**Set up the book's repo once**, outside this repo, in its own environment (its requirements
include TensorFlow, which is used only to read OpenAI's GPT-2 weight files):

```bash
cd ~/Desktop/repos
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
jupyter lab                        # opens in the browser; navigate to the chapter folder
```

How to work through a chapter: read the section in the book and type the code into your own
notebook or `.py` file. Use the repo's `chNN.ipynb` to check your work when you get stuck. Most
chapters also include a standalone script that runs the whole chapter at once. Run it from that
chapter's `01_main-chapter-code/` folder.

#### Day 1 · Tools, and what training does

| Order | What | Where | Run |
|---|---|---|---|
| 1 | PyTorch basics | book repo `appendix-A/01_main-chapter-code/` | open `code-part1.ipynb`, then `code-part2.ipynb` |
| 2 | Backprop from one neuron (micrograd) | hub `z2h-2-micrograd/labs/` | `cd z2h-2-micrograd && python labs/derivative-as-nudge.py`, then on in the README's order |
| 3 | A first language model (bigram) | hub `z2h-3-bigram/labs/` | `cd z2h-3-bigram && python labs/names-dataset.py`, then on in order |

Optional depth: `z2h-1-toolkit/labs/` (maths refresher) and `z2h-4-mlp/labs/` (the MLP model).
Karpathy's lecture notebooks for the same material are in
[karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) under `lectures/`.

#### Day 2 · Text → batches, then attention

| Order | What | Where | Run |
|---|---|---|---|
| 1 | Tokens, IDs, BPE (§2.1–2.5) | book `ch02/01_main-chapter-code/` | `ch02.ipynb` (`dataloader.ipynb` is the short version) |
| 2 | Every position is a training example | hub `build-an-llm/labs/` | `cd build-an-llm && HF_HUB_OFFLINE=1 python labs/training-pairs.py` |
| 3 | Data loader, embeddings, positions (§2.6–2.8) | book `ch02/01_main-chapter-code/` | `ch02.ipynb` |
| 4 | Self-attention to multi-head causal attention | book `ch03/01_main-chapter-code/` | `ch03.ipynb` (`multihead-attention.ipynb` is the short version) |

Optional depth: build your own tokenizer with `z2h-9-tokenizer/labs/`
([karpathy/minbpe](https://github.com/karpathy/minbpe) is Karpathy's code for it).

#### Day 3 · The GPT model

| Order | What | Where | Run |
|---|---|---|---|
| 1 | Layer norm, GELU, shortcuts, the block, the full GPT | book `ch04/01_main-chapter-code/` | `ch04.ipynb`; the whole chapter as a script: `python gpt.py` (it generates gibberish because the model is untrained, and that's expected) |
| 2 | Optional second pass: Let's build GPT | hub `z2h-8-gpt/labs/` | `cd z2h-8-gpt && python labs/shakespeare-lm.py`, then on in order ([karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) has the lecture's `gpt.py`) |

#### Day 4 · Pretrain it

| Order | What | Where | Run |
|---|---|---|---|
| 1 | Loss and the training loop (§5.1–5.2) | book `ch05/01_main-chapter-code/` | `ch05.ipynb`; script version: `python gpt_train.py` (a few minutes on a laptop) |
| 2 | Warmup, cosine decay, gradient clipping | book `appendix-D/01_main-chapter-code/` | `appendix-D.ipynb` |
| 3 | Temperature and top-k (§5.3) | book `ch05` | `ch05.ipynb` |
| 4 | Load OpenAI's GPT-2 weights (§5.4–5.5) | book `ch05` | `ch05.ipynb`; script version: `python gpt_generate.py` (downloads GPT-2 the first time) |

Optional stretch: reproduce GPT-2 with `z2h-10-gpt2/labs/` and
[karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt). The labs run on a Mac and skip
the steps that need an NVIDIA GPU. A real pretraining run (`train_gpt2.py` on FineWeb-edu) needs a
rented GPU.

#### Day 5 · Make it useful

| Order | What | Where | Run |
|---|---|---|---|
| 1 | Fine-tune for classification (needed for step 5) | book `ch06/01_main-chapter-code/` | `ch06.ipynb`; script version: `python gpt_class_finetune.py` |
| 2 | Instruction tuning: format and loss masking | hub `build-an-llm/labs/` | `cd build-an-llm && HF_HUB_OFFLINE=1 python labs/instruction-tuning.py` |
| 3 | Instruction fine-tuning (§7.1–7.7) | book `ch07/01_main-chapter-code/` | `ch07.ipynb`; script version: `python gpt_instruction_finetuning.py` |
| 4 | Judge the answers with another LLM (§7.8) | book `ch07` | `brew install ollama`, then `ollama serve` in one terminal and `ollama pull llama3` in another, then `python ollama_evaluate.py --file_path instruction-data-with-response.json` |
| 5 | LoRA | hub `build-an-llm/labs/`, then book `appendix-E/01_main-chapter-code/` | `cd build-an-llm && HF_HUB_OFFLINE=1 python labs/lora.py`, then `appendix-E.ipynb` |

### Alongside Part 2 · The math of an LLM

Course: `llm-math/`. Every equation the book's GPT uses, worked on one tiny model you can compute
by hand: 5 words, 3 tokens, 4 numbers per vector, 220 weights. Read a card the evening before its
chapter (the course overview has the map). Nothing downloads; each lab runs in a second.

```bash
cd ~/Desktop/repos/explainers/llm-math
python labs/tiny.py
```

| Lab (`llm-math/labs/`) | What it shows |
|---|---|
| `tiny.py` | The whole model: one training step in plain numpy (forward, loss, backward, update), checked against PyTorch's autograd for all 220 weights. Every card's numbers come from here. |
| `why.py` | The "why" measurements: the nudge test, softmax overflow, why √d, two linear layers collapsing, activations through depth, residuals, RoPE, learning rates, Adam |
| `tiny_net.py` | The 6-weight network that card 3 trains on paper |
| `<card>.py` | One per card (`attention-scores.py`, `grad-attention.py`, …): that card's worked example and its "Change it" experiment |

The finish line is `worksheet.html`: print it, do the step by hand, and check your numbers against
`python labs/one-training-step.py`.

### After the paths (optional)

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT): a clean, fast GPT training repo.
- [karpathy/nanochat](https://github.com/karpathy/nanochat): a whole small ChatGPT pipeline (pretraining, fine-tuning, chat).
- [karpathy/llm.c](https://github.com/karpathy/llm.c): GPT-2 training in plain C/CUDA.

## How it's organised

```
explainers/
  index.html            the hub page (generated, don't edit)
  llm-latency/          a course: has its own index.html gallery
  nucleus-sampling/     a single card
  …
```

- A folder with an `index.html` is a **course**; its tile shows the card count and opens its gallery.
- A folder with one card is a **single card**.
- The title and description come from each folder's page. To override them, add
  `<folder>/explainer.json` with `{"title": "…", "description": "…", "date": "YYYY-MM-DD"}`.

## Adding an explainer

Put the folder here, then:

```bash
./publish.sh <folder>
```

This rebuilds `index.html` from the folders (`build-hub.mjs` + `hub-template.html`), commits and
pushes. GitHub Pages updates in about a minute. The skills do this automatically when a card or
course is finished.

`.nojekyll` must stay: without it, GitHub Pages drops every file whose name starts with `_`, such
as `_overview.html`.

## Course navigation

`course-nav.mjs` gives every course the same navigation. A course is any folder whose `index.html`
has a `GROUPS` list. Each card gets breadcrumbs with prev/next at the top and a big "next up"
preview at the bottom, and each course index gets a contents sidebar in the reader. `publish.sh`
runs it on every publish (papers included), so a new course picks it up with no extra step.

The `perf-*` folders form one linked series. `perf-series.json` lists the courses in order, what each course needs, and each card's prerequisites. For those courses the navigation also adds a course strip, a "before this" line on each card, and "next up" links that cross into the next course. Adding a course to the series only needs an entry in `perf-series.json`. `cs-series.json` does the same for the `cs-*` courses ("The core CS canon").

On the hub, each series shows as one wide tile (on its map folder) that lists its courses. Its member courses are hidden from the grid unless a search matches them.

## Study path

`study-path.json` holds a list of study paths (`{"paths": [...]}`), and each one gets its own section above the hub grid: the LLM inference path, then "How a coding agent works". A path can name an overview page (`overview`, e.g. `study-path-overview`) and a dashed "later" card. Each day has a checklist folder (`plan`) and a one-page folder (`flow`). Both leave the grid (search still finds them), and each section reads its checklists' ticks from localStorage to show progress and a "Continue" button. The pages are generated by `tasks/study-plans/gen.mjs` (checklists) and `gen-flow.mjs` (one-page versions), which live only on the author's machine. The agents path's data is in `tasks/study-plans/agents.mjs`. It reads its readings and checkpoint questions from the Agents field guide's own data in `ai-systems-field-guides/guides/agents/`.

### Mission versions

A path can open with a **mission**: one result you build and post ("my X beats Y on a fair benchmark"), shown as a tweet with blanks that fill in from your own logged numbers. Training agents (`training-agents-roadmap/mission.html`) and Build an LLM (`build-mission/`) have one. Each mission folder holds the mission page (`index.html`), one page per day (`day-N.html`) and its labs. Every page has a **Path version: Mission | Classic** switch; Classic is the reading-first version, and ticks are shared between the two. In `study-path.json` the mission version is a `roadmap` path and the old one sits under `classic`. The scoreboard is kept in your browser only.

- `build-mission/labs/`: `afd_corpus.py` (downloads weather-service forecast discussions), `bpb.py` (bits per byte for compressors, a byte bigram and GPT-2), `tiny_gpt.py`, `finetune_gpt2.py`, `sweep.py` and `sample.py`. Standard library for the first two; `uv run --python 3.12 --with torch [--with transformers]` for the rest.

## Study Mode

Cards teach through a sequence of study stages: Prime → Learn → Encode → Compress → Retrieve → Compare → Predict → Apply → Reflect. [learning-principles/](learning-principles/learning-principles.html) explains why.

- Each card's stages live in `<course>/_study/<slug>.json`. `study-lib.mjs` has the schema and the rules.
- `study-build.mjs` injects a **Prime** block under the card's subtitle and a CSS-only **Study this card** stepper before the footer (cards can't run script). Each stage appears only once the previous one is opened.
- `publish.sh` runs it before `course-nav.mjs`. Use `--card course/slug` to build a single card.
- `verify-study.mjs <course>` checks the JSON, flagging generic or repeated questions, missing causal chains and links to cards that don't exist.
- For interactive tutoring, `/study <course>/<slug>` in Claude Code walks through the stages one at a time and grades your answers.

## Papers

Cards made from a paper (via `concept-curriculum` pointed at an arXiv paper) land in a
`paper-<id>/` folder, each with an `explainer.json` that has `"kind": "paper"`. `sync-papers.mjs`
keeps these folders in sync with the paper source list; it only touches `paper-*` folders and
leaves everything else alone. `publish.sh` runs `node sync-papers.mjs` before `build-hub.mjs`, so
every publish re-syncs papers first. A launchd job (`launchd/com.seyonv.explainers-sync.plist`,
installed separately) runs `./publish.sh papers` every 30 minutes so new papers show up without
manual intervention.
