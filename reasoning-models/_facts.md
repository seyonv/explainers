# How reasoning models work: course facts

The audience is someone new to deep learning. They know roughly what a transformer is but have no ML training background.
The course fills the gaps between the hub's existing cards (z2h, perf, training-agents) and the Models field guide stations.

## Cards in this course (`reasoning-models/<slug>`)

| Slug | Title (claim) | One line |
|---|---|---|
| `_overview` | How a reasoning model is made | The map: pretrain → assistant → think at inference → learn to think with RL |
| `modern-block` | The 2017 block, four upgrades later | RoPE, RMSNorm, SwiGLU, GQA: what each replaced and why |
| `policy-gradient` | Make good samples more likely | RL from zero: policy, reward, REINFORCE, baseline |
| `reward-models-rlhf` | Learn what people prefer, then chase it | InstructGPT: SFT → reward model (pairwise) → PPO with a KL leash |
| `dpo` | Skip the reward model | DPO: the same preference goal as a classification loss |
| `chain-of-thought` | Show your working | CoT prompting: steps in the prompt unlock multi-step reasoning, only at scale |
| `self-consistency` | Ask many times, take the vote | Sample many chains, majority vote on the answer |
| `process-rewards` | Grade the steps, not just the answer | PRM vs ORM (Let's Verify Step by Step) — DONE, the reference card |
| `test-time-compute` | Think longer or be bigger? | Snell et al.: compute-optimal search vs revision, when it beats a 14× bigger model |
| `budget-forcing` | "Wait" | s1: 1,000 examples + forcing the model to keep thinking |
| `star` | Learn from your own right answers | STaR: generate rationales, keep the correct ones, fine-tune, repeat |
| `r1-zero` | Reasoning from reward alone | DeepSeek-R1-Zero/R1: GRPO + rule rewards → long CoT, the "aha moment", then the R1 pipeline |

## Existing hub cards to link (as `course/slug`)

- Basics: `z2h-2-micrograd/one-step-along-gradient`, `z2h-1-toolkit/softmax-and-cross-entropy`, `z2h-8-gpt/shakespeare-lm`, `z2h-8-gpt/self-attention-head`, `z2h-8-gpt/embeddings-and-positions`, `z2h-9-tokenizer/why-tokenization`
- Sampling: `nucleus-sampling/nucleus-sampling`, `z2h-10-gpt2/sampling`, `paper-2107-03374/sampling-temperature`
- Architecture: `perf-1-foundations/attention`, `perf-1-foundations/transformer-forward`, `perf-4-attention-kv/gqa`, `llm-latency/moe`, `perf-6-scaling-out/moe-basics`
- Training: `perf-7-training-economics/compute-budgets`, `perf-7-training-economics/post-training`
- RL: `the-group-is-the-baseline/the-group-is-the-baseline` (GRPO on a coding agent)
- Latency of thinking: `llm-latency/thinking-tokens`

## Models field guide stations (https://agents-inference-software-factory.svas.workers.dev/models/)

ST-1 `#tensors-and-backprop` · ST-2 `#character-language-models` · ST-3 `#the-transformer` · ST-4 `#pretraining-and-systems` · ST-5 `#scaling-and-compute` · ST-6 `#sft-peft-preferences` · ST-7 `#reasoning-without-weight-changes` · ST-8 `#rl-for-reasoning`

## Sketch colour roles (use the same colours in the prose, `--blue`, `--green`, `--red`, `--yellow`, `--violet`)

- yellow: the problem or prompt
- green: correct, rewarded, or "this one wins"
- red: wrong step, penalised, failure
- blue: the scorer, grader or checker (what looks at the output)
- violet: the model or policy being trained, where it needs its own colour
