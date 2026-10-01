# Shared facts: stack-5 From evidence to training

Course 5 of the Agent Stack series. How agents get better: what changes in RL when the policy is an agent, rewards beyond verifiable, failure modes, and why penalizing a monitor backfires.

## Series facts (shared by every stack-* course)

### What this series is
"The Agent Stack" is a compact course, about 31 hours including a capstone paper.

- **Sources:**
  - Vinoth Govindarajan's newsletter *The Agent Stack* (theagentstack.substack.com). He is a systems engineer at OpenAI, co-author of *Engineering Lakehouses with Open Table Formats*, and has a data-infrastructure background.
  - Cameron R. Wolfe's *Deep (Learning) Focus* (cameronrwolfe.substack.com), used for the training and eval side.
- **Goals:** a deep model of the agent stack organized by **ownership** (who owns the run, the working set, the capability, the side effect, the evidence), labs in his forensics style, and a research paper at the end.
- **Spec:** `~/Desktop/repos/explainers/tasks/agent-stack/spec.md`.

### Where to read the sources (read your card's sources IN FULL before writing)
- Agent Stack posts as full text: `~/Desktop/repos/explainers/tasks/agent-stack/source/agentstack-posts/NN-*.md`.
  - 01–06 OpenClaw
  - 07–14 Agent Stack v1, Pts 1–8
  - 15–21 Data Agent Stack, Pts 1–7
  - 22–26 Hermes
  - 27–28 OpenCode
- Wolfe posts as full text: `~/Desktop/repos/explainers/tasks/agent-stack/source/wolfe-posts/<slug>.txt`. For example: `agent-evals.txt`, `agentic-rl.txt`, `agentic-world-models.txt`, `ai-agents.txt`, `stats-llm-evals.txt`, `rubric-rl.txt`, `reward-models.txt`, `llm-as-a-judge.txt`, `finetuned-judge.txt`, `grpo-tricks.txt`, `llm-bench.txt`.
- Research digests, with the posts' URLs and per-post bullets:
  - `~/Desktop/repos/explainers/tasks/agent-stack/research/agentstack-research.md`
  - `.../research/wolfe-research.md`
  - `.../research/novelty-check.md`, the verified paper list for the capstone
- **Cite the live post URL in the footer**, not the local file. URLs are in the digests and the `.md` headers. Agent Stack URLs look like `https://theagentstack.substack.com/p/<slug>`; Wolfe URLs look like `https://cameronrwolfe.substack.com/p/<slug>`.
- **Before you use any number, confirm it in the post's full text.** The digests are summaries; the post is the source of truth. If they disagree, use the post and say so in your return.
- When a post *comments on* something, write "X says Y". Only write "X did Y" when the post describes doing it.

### The series (link sibling cards as `../<course>/<card>.html`, using the card lists)
stack-0-map · stack-1-runs · stack-2-context-memory · stack-3-authority · stack-4-evidence · stack-5-training · stack-6-data-agents · stack-7-capstone

### Existing hub material: link to it, never re-teach it
The reader has already studied these. Where your card touches one, add a one-line "You already know: <link>" and move on.

| Topic | Card |
|---|---|
| Agent loop: call, tool, append, repeat, every way a run ends | `../agents-basics/agent-loop.html` |
| Anatomy of a tool call | `../agents-basics/tool-call.html` |
| What the model sees on each call (prefix, history, compaction basics, skills, MCP, subagents) | `../agents-basics/what-the-model-sees.html` |
| Measuring an agent (hidden tests, repeats, intervals, cost per solved task) | `../agents-basics/agent-evals.html` |
| RL from zero, SFT on traces, distillation, GRPO, environments | `../training-agents-roadmap/index.html` (stage files: `stage-1-rl-from-zero.html`, `stage-3-rl-for-agents.html`, `stage-6-grpo.html`, `stage-7-environments.html`) |
| Prompt caching and context growth (speed side) | `../llm-latency/index.html` |
| "The group is the baseline" (GRPO advantages) | `../the-group-is-the-baseline/the-group-is-the-baseline.html` |

### The reader
- An engineer who builds agents (hackathon agents, coding-agent harnesses, the `mini-swe-agent` and `trajviz` tools) and has done the hub's agents and training-agents material.
- **Not** an ML researcher. They want a deep, ownership-level understanding, then to ask research questions that lab people would find thoughtful.
- They are new to ML notation. **Any literal identifier** (a flag, a file name like `USER.md`, a model tag) goes in `<code>`, and its first use leads with plain words.
- Machine: Apple M3, 24 GB, macOS. Python 3.10 is `python3`; numpy is available, scipy isn't.

### The running example: "the Tuesday run" (illustrative, constructed for this series)
Reuse it on every card where it fits. Label it illustrative the first time it appears on a card.
- A coding agent is asked to fix the failing test `test_parse_due_date` in a small repo, `invoice-tools`.
- **Step 6:** it runs `pytest`, which reports `1 failed, 41 passed` and **exit code 1**.
- **Step 7:** the context hits its limit and the harness **compacts**. The summary reads: "Fixed due-date parsing; tests passing."
- **Step 9:** the agent ends with "Done — fixed the parser and all tests pass."
- **The checker** re-runs `pytest` and gets exit code 1, with the same test still failing.
- **The run's claim chain:**
  - request completed ✓
  - tool calls completed ✓
  - command succeeded ✗ (exit 1)
  - effect verified ✗
  - delivered ✓, but with a false claim
- This is the phenomenon the capstone paper measures. Stack-6 (data agents) uses its own example: "What was Q3 net revenue for EMEA?"

### Code: Python only, where code helps
- Cards on statistics (pass^k, error bars, power analysis), claim checking or mechanisms that can be simulated should include a short, runnable Python snippet (3.10, standard library plus numpy).
- Run it, put its real output on the card, and save it to `DIR/code/<card-slug>.py` with an `if __name__ == "__main__":` demo.
- Link it in the footer: `Run it: <a href="https://github.com/seyonv/explainers/blob/main/<course>/code/<slug>.py">code/<slug>.py</a>`.
- Lines are at most 72 characters. Conceptual cards may have no code.
- Code block CSS (verbatim in `<style>`):
```css
.code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;line-height:1.5;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin:10px 0;white-space:pre;overflow-x:auto;tab-size:4;color:var(--text)}
.code .k{color:var(--accent);font-weight:600}.code .c{color:var(--muted)}.code .o{color:var(--muted)}
```

### Colour meanings (the same on every card)
- **Green `--accent`:** verified, owned, or the recommended choice. Evidence that was checked against the world.
- **Red `--red`:** failure, a false claim, authority that leaked or broadened, or lost state.
- **Grey `--faint`:** out of scope, illustrative, or "you already know this".

### Terms (use these names; give the aliases once in the subtitle)
- **control plane:** owns the run and the session.
- **runtime:** owns progress, not authority.
- **working set:** what the model sees this turn. It is not the transcript.
- **compaction:** replacing history with a summary. It is a lossy state change.
- **capability surface:** what the model may *ask for*.
- **execution surface:** where effects actually happen.
- **identity envelope:** the authority carried with a request.
- **claim chain:** request completed → tool completed → command succeeded → effect verified → delivered.
- **false success claim:** the final message asserts success, but the world state disagrees.
- **status inflation:** a summary upgrades failed or unknown to passed or done.
- **pass@k / pass^k:** at least one of k trials succeeds / all k trials succeed.

### Voice
- Plain, concrete and short. Write as an engineer teaching an engineer.
- Every card has one honest clarification box saying where the idea breaks down or what the author does not claim.
- Govindarajan is careful about his evidence; be equally careful. He ran no no-skill control, for example, so don't overstate his findings.


## This course's card list
| File | Title | Group | Owner |
|---|---|---|---|
| agentic-rl-changes.html | Agentic RL: the design choices | Agentic RL | subagent |
| rubric-judge-rewards.html | Rewards you can't verify | Rewards | subagent |
| rl-failure-modes.html | How agentic RL goes wrong | Failure modes | subagent |
| monitor-obfuscation.html | Penalize the monitor, teach the hiding | Rewards | subagent |
| world-models.html | Agents that model their environment | Frontier | subagent |
| lab-judge-calibration.html | Lab: be the second annotator | Lab | subagent |
| _overview.html | (overview, written last) | Start here | main |
| open-questions.html | Open questions | Research | main |

## Every card in the series (for Related links; link even if not written yet)
- stack-0-map/ten-layers.html: The ten layers, by what they own
- stack-0-map/same-label.html: Same label, different authority
- stack-0-map/forensics-method.html: The forensics method
- stack-1-runs/session-ownership.html: Owning the session
- stack-1-runs/durable-execution.html: Retry, replay, resume
- stack-1-runs/done-chain.html: 'Done' is a chain of states
- stack-2-context-memory/assembled-context.html: Context is assembled, not given
- stack-2-context-memory/prompt-snapshots.html: Prompt snapshots and freshness
- stack-2-context-memory/compaction-lossy.html: Compaction is a lossy state change
- stack-2-context-memory/memory-lifecycle.html: Memory is owned state
- stack-3-authority/capability-surface.html: A tool is a capability surface
- stack-3-authority/untrusted-output.html: Tool output is untrusted input
- stack-3-authority/identity-envelopes.html: Authority only narrows
- stack-3-authority/approval-not-isolation.html: Approval is not containment
- stack-4-evidence/evidence-loop.html: Trace, audit, eval, feedback
- stack-4-evidence/graders.html: Tasks, trials and graders
- stack-4-evidence/pass-k.html: pass@k vs pass^k
- stack-4-evidence/error-bars.html: Error bars that don't lie
- stack-4-evidence/power-analysis.html: How many runs do you need?
- stack-5-training/agentic-rl-changes.html: Agentic RL: the design choices
- stack-5-training/rubric-judge-rewards.html: Rewards you can't verify
- stack-5-training/rl-failure-modes.html: How agentic RL goes wrong
- stack-5-training/monitor-obfuscation.html: Penalize the monitor, teach the hiding
- stack-5-training/world-models.html: Agents that model their environment
- stack-6-data-agents/data-agent-anatomy.html: What a data agent is
- stack-6-data-agents/data-foundation.html: The foundation is part of the agent
- stack-6-data-agents/context-cards.html: Meaning lives in code
- stack-6-data-agents/receipts.html: Every answer ships with a receipt
- stack-6-data-agents/changing-data-evals.html: Evaluating against data that moves
- every course also has _overview.html, open-questions.html and index.html
