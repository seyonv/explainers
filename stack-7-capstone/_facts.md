# Shared facts: stack-7 The paper

Course 7 of the Agent Stack series, the capstone. Eight guided steps that take the research question from an anecdote to a released paper, using the real research repo `~/Desktop/repos/compaction-claims`. Every number below comes from that repo's docs (commit and file named) or from the series spec; nothing here is re-derived differently.

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


## This course's sources (read in full before writing a card)
- Repo `~/Desktop/repos/compaction-claims` (another session is developing it; read-only for card writers):
  - `README.md` (commands, design summary, layout)
  - `docs/preregistration.md` (registered at `c5d7e9f`; addenda 1 `5298718`, 2 `4d05af2`)
  - `docs/trigger-design.md` (why live triggers failed; branching)
  - `docs/baseline-pilot.md` (baseline outcomes, classifier validation, second model, power and cost)
  - `pilot/results.md` (the 21,438-run public-trajectory decomposition, licensing)
  - `harness/prompts/PROVENANCE.md` (transplanted prompts; why no Claude Code prompt)
- Series spec `tasks/agent-stack/spec.md`: "The paper", the novelty addendum, "REFRAME after pilot v2", "Baseline result and design update". The latest section wins.
- Verified related work: `tasks/agent-stack/research/novelty-check.md`.

## Key numbers (source in brackets)
| Number | Meaning | Source |
|---|---|---|
| 21,438 runs, 45 submissions, 500 tasks | public SWE-bench Verified trajectories in the pilot | pilot/results.md §1–2 |
| 8,418 (39.3%) | runs ending in a success claim | results.md §1 |
| 426 (5.1%) / 105 (1.2%) / 14 (0.17%) | strict self-contradiction / task-relevant (automatic) / hand-verified genuine, of claims | results.md §1 |
| 0.4% | rough ceiling on genuine self-contradiction | results.md §3.3 |
| 1,310 (15.6%) | honest misses (last test passed, hidden tests failed) | results.md §1 |
| 3,092 (36.7%); 2–73% by scaffold | claims with no executed test | results.md §1, §4 |
| ρ 0.055 / 0.32 | intra-task correlation: task-relevant SC / honest miss | results.md §3.5 |
| ~5,110 vs ~160 tasks | to detect a doubling from 0.17% vs from 5% (ρ 0.055, m 3) | stack-4 lab-pilot-audit step 5 |
| Δ ≈ 0.22–0.27; power 21–27% at Δ 0.10 | what 30 tasks × 3 repeats could detect (baseline 0.20, ρ 0.2–0.5) | spec novelty addendum |
| 73% | live runs reaching the post-edit trigger (T2; 71% for 5.4-mini) | trigger-design.md |
| 34% (14 of 41) | T1 compactions that fired before any edit | trigger-design.md |
| 83–92% within 2 tries | prefix yield, branching pilot T5 | trigger-design.md |
| $3.71 | spend across all trigger pilots | trigger-design.md |
| 40/40 prefixes first try; median step 4; $0.0061/prefix | baseline prefix yield | baseline-pilot.md |
| P1 0/40, 1/40, 0/40 | d0, d1-codex, d1-opencode | baseline-pilot.md |
| tampering 7/40 = 17.5%, 15/40 = 37.5%, 4/40 = 10.0% | d0, d1-codex, d1-opencode | baseline-pilot.md |
| +20.0 pp [+5.0, +37.5], p = 0.015 | paired tampering d1-codex − d0 (exploratory) | baseline-pilot.md; re-computed in scratch with analysis/stats.py |
| −7.5 pp [−17.5, +2.5], p = 0.24 | paired tampering d1-opencode − d0 | baseline-pilot.md (0.235 unrounded) |
| 25 of 26 | tampering runs that disclosed it | baseline-pilot.md |
| 0/80 | inflated summaries (0/40 per prompt) | baseline-pilot.md |
| 32/40 vs 4/40 | verdict line in the summarizer's input, Codex vs OpenCode | baseline-pilot.md |
| 113/120 | branches resolved (hidden tests pass) | baseline-pilot.md |
| $1.98 / $3.07 / $8 cap | baseline pilot / whole baseline phase / phase cap | baseline-pilot.md |
| 65/66 (98.5%), κ 0.000; 24/24 summaries; 49/49 attribution | held-out extractor validation (all hand labels one class) | baseline-pilot.md; label_agreement.py --cached |
| 36/36, κ 1.000 (20 positives) | dev set | label_agreement.py on labels/dev.json --cached |
| κ ≥ 0.80 and accuracy ≥ 90%; < 5 positives → 100% recall | acceptance rule | preregistration.md §3.1 |
| `44bdd8fe…` | frozen prompt hash after addendum 2 (`dcdd1eef…` at registration) | preregistration.md |
| 40 tasks × 2 repeats × 2 models = 160 prefixes, 1,120 branches, ≈ $80 | recommended sweep size and cost (7 arms; superseded by addendum 3: 6 arms, 960 branches, $120 cap) | baseline-pilot.md |
| power 0.77–0.90 at +5 pp, ≥ 0.97 at +10 pp | at d0 1–2.5%, T = 40 | baseline-pilot.md |
| ≈ $195 | the earlier 100-task plan | baseline-pilot.md |
| sweep: 40 new tasks × 2 prefixes × 2 models; arms d0, d1-codex, d1-opencode, d3-codex, d1-codex-cut2000, d1-codex-intervention; H2 tampering confirmatory; cap $120 | the confirmatory sweep as the spec records it (superseded by addendum 3 below) | spec "Baseline result and design update" |
| 40 fresh tasks (disjoint, seed 1) × 2 models × 2 repeats = 160 prefixes; 6 arms (one d0) = 960 branches; 2 chunks (4 then 36 tasks); $120 hard cap, phase `main`; H2 tampering confirmatory, one-sided p < 0.025; families T (5), S (6), M (1) | the binding sweep design | preregistration.md addendum 3, `ce5bd74` |
| gpt-5.5: $0.074/prefix, $0.033/d0 branch, $0.092/d1 branch (n = 6) | second model viability | baseline-pilot.md |
| budget: about $150–200, ceiling $250 | whole series | spec |

## Not yet known (placeholders on the cards)
- Every confirmatory sweep result (H1a/H1b, H2–H5, per-model rates, cut2000, intervention). Cards mark these "pending: filled when the sweep completes".

## This course's card list
| File | Title | Group | Owner |
|---|---|---|---|
| the-question.html | How the question got sharp | The question | main |
| the-design.html | Branch from the agent's own run | The design | subagent |
| preregistration.html | Write the analysis down first | The design | subagent |
| baseline-and-power.html | Measure the floor, then size the study | The design | subagent |
| the-sweep.html | Run the sweep | Running it | subagent |
| analysis.html | Analyse what you registered | Running it | subagent |
| writing-the-paper.html | Write it against the evidence | Shipping | subagent |
| release.html | Release and pitch it | Shipping | subagent |
| _overview.html | (overview, written last) | Start here | main |
| open-questions.html | What to do after the paper | Research | main |

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
- stack-5-training/agentic-rl-changes.html: What changes when the policy is an agent
- stack-5-training/rubric-judge-rewards.html: Rewards you can't verify
- stack-5-training/rl-failure-modes.html: How agentic RL goes wrong
- stack-5-training/monitor-obfuscation.html: Penalize the monitor, teach the hiding
- stack-5-training/world-models.html: Agents that model their environment
- stack-6-data-agents/data-agent-anatomy.html: What a data agent is
- stack-6-data-agents/data-foundation.html: The foundation is part of the agent
- stack-6-data-agents/context-cards.html: Meaning lives in code
- stack-6-data-agents/receipts.html: Every answer ships with a receipt
- stack-6-data-agents/changing-data-evals.html: Evaluating against data that moves
- stack-1-runs/lab-claim-checker.html: Lab: build the claim checker
- stack-2-context-memory/lab-compaction-rig.html: Lab: the compaction rig
- stack-4-evidence/lab-pilot-audit.html: Lab: audit 21,000 public runs
- stack-5-training/lab-judge-calibration.html: Lab: be the second annotator
- stack-7-capstone/the-question.html … release.html: see the card list above
- every course also has _overview.html, open-questions.html and index.html
