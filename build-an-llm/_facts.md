# Build an LLM yourself: course facts

The reader has finished the LLM basics path (`text-to-answer/`): they can trace text through a model but haven't coded one. This course comes with the "Build an LLM yourself" study path, which follows Sebastian Raschka's *Build a Large Language Model (From Scratch)* (Manning, 2025), chapter by chapter, with Karpathy's Zero to Hero for extra depth. The book does the building. These cards cover the ideas the book moves through quickly and the hub has no card for, measured so the reader knows what to look for when coding.

## Running example

The same as the basics course: **Qwen2.5-0.5B** base, float32, on an M3 Mac, with the question `What is the capital of India?`. Its facts are in `text-to-answer/_facts.md` (24 layers, hidden size 896, vocab 151,936, 494,032,768 parameters, tied embeddings). The book itself uses GPT-2 124M (12 layers, hidden 768, vocab 50,257). Mention that contrast where it matters.

The shared loader is `build-an-llm/labs/common.py`. Run labs with `HF_HUB_OFFLINE=1 ~/repos/nanoRL/.venv/bin/python build-an-llm/labs/<slug>.py`.

## Cards in this course (`build-an-llm/<slug>`)

| Slug | Title (claim) | Book chapter | One line |
|---|---|---|---|
| `_overview` | From reading to building | all | the route through the book, and where each Zero to Hero course plugs in |
| `training-pairs` | Every position is a training example | ch 2.6, 5.1 | sliding-window input/target pairs shifted by one; one sequence of n tokens gives n predictions; context length, stride, batches |
| `instruction-tuning` | Teach the format, grade only the answer | ch 7.2–7.6 | the instruction template, padding and batching, masking the loss on the instruction and padding (ignore index −100), why |
| `lora` | Fine-tune a sliver of the model | appendix E | freeze W and learn a low-rank B·A, parameter and memory counts measured on Qwen, rank and alpha, when LoRA falls short |
| `checkpoint-1` | Checkpoint | — | closed-book questions over the path |

## Existing hub cards to link (as `course/slug`)

- Training basics: `z2h-2-micrograd/one-step-along-gradient`, `z2h-2-micrograd/mlp-and-training-loop`, `z2h-3-bigram/likelihood-and-nll`, `z2h-1-toolkit/softmax-and-cross-entropy`, `z2h-4-mlp/minibatches`
- Building GPT: `z2h-8-gpt/batches-of-chunks` (the closest existing card to training pairs), `z2h-8-gpt/masked-softmax`, `z2h-10-gpt2/the-target`, `z2h-10-gpt2/optimizer-recipe`
- The route: every `text-to-answer/<slug>` card
- Post-training: `reasoning-models/reward-models-rlhf` (SFT is step 1), `training-agents-roadmap/stage-4-sft-on-traces`, `perf-7-training-economics/post-training`
- Judges: `stack-5-training/rubric-judge-rewards`

## Papers and links (use only these)

LoRA, Hu et al. 2021: arXiv 2106.09685 (verify against the PDF before citing numbers) · the book's code: https://github.com/rasbt/LLMs-from-scratch

## Sketch colour roles

- yellow: input text and tokens
- blue: vectors, numbers and targets
- violet: the model and its weights
- green: what the loss counts, what's trained, the winner
- red: masked out, frozen, failure
