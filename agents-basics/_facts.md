# Shared facts: How a coding agent works (the basics)

Every card reads this before writing. The values below appear on the cards exactly as written here.

## The reader
- New to ML. Has done Day 1 of the inference study path (the `llm-latency` course), so knows what a token, time to first token (TTFT) and prompt caching are.
- Wants: a gentle on-ramp before each day of the "How a coding agent works" study path, whose readings are external (anthropic.com posts, docs, papers). These cards must be gentler than those readings.

## The running example
A small Python package with one failing test, `tests/test_slugify.py`. The agent's job: find the bug, fix it, make the tests pass. (Same task as the Agents field guide's worked trace.)

## Prices used (Claude Sonnet 5, per 1M tokens)
| Item | Price | Source |
|---|---|---|
| Input | $2 | platform.claude.com/docs/en/about-claude/pricing (via the field guide's sources.json, read 2026-09-26) |
| Cache write (5-minute) | $2.50 (1.25×) | same; prompt-caching docs |
| Cache read | $0.20 (0.1×) | same |
| Output | $10 | same |

## Sourced numbers (all from the Agents field guide's data/sources.json, data/lessons.json or official docs fetched 2026-09-29)
| Fact | Value | Source |
|---|---|---|
| Stop reasons | end_turn, max_tokens, stop_sequence, tool_use, pause_turn, refusal, model_context_window_exceeded | Claude docs, handling stop reasons |
| Budget example | ~$0.10 per task affords ~30,000–50,000 tokens | Barry Zhang, AI Engineer Summit 2025 |
| Succeed fast, fail slow | SWE-agent: successes median 12 steps, $1.21; failures average 21 steps, $2.52 | SWE-agent paper, Section 5 |
| mini-swe-agent budgets | step_limit, cost_limit (default $3), wall_time_limit_seconds | mini-swe-agent default.py |
| Tool result rules | all results in the next user message, matched by tool_use_id, tool_result blocks before any text; missing ones → 400 "tool_use ids were found without tool_result blocks immediately after" | Claude docs, handle tool calls + parallel tool use (fetched 2026-09-29) |
| SWE-agent file viewer | 100 lines 18.0%, 30 lines 14.3%, whole file 12.7% (SWE-bench Lite) | SWE-agent paper, Table 3 |
| SWE-agent search | summarized 18.0%, none 15.7%, iterative 12.0% | same |
| SWE-agent lint-on-edit | 18.0% vs 15.0% without | same |
| Output caps | mini-swe-agent: ≥10,000 chars → first and last 5,000 plus a warning; Claude Code bash ≈30,000 chars | field guide principle 5 |
| Claude Code startup example | system prompt 4,200 · project CLAUDE.md 1,800 · skill descriptions 450 · user CLAUDE.md 320 · environment 280 · deferred MCP tool names 120 | code.claude.com/docs/en/context-window |
| Skill disclosure | ~100 tokens of name + description per skill at startup; body under 5,000 tokens recommended, loaded on activation | agentskills.io spec |
| Subagent | starts with own system prompt, the delegation message, CLAUDE.md and git status, no conversation history; only its final report returns | Claude Code subagent docs |
| Hooks | exit 2 on PreToolUse blocks the tool call | Claude Code hooks reference |
| 566 MCP tools | ≈208,000 tokens of definitions | field guide F17 |
| Context rot | 18 models; accuracy drops as input grows, even on simple tasks | Chroma, 2025 |
| pass^k | chance all k trials pass | Anthropic, Demystifying evals |
| Hidden tests | ImpossibleBench: GPT-5 "passed" 54% of impossible tasks with test access; hiding tests took cheating near zero | field guide principle 8 |
| Per-run cost estimates | Sonnet 5 ≈ $0.49, Haiku 4.5 ≈ $0.25 for a cached 40-step run | field guide pricing notes |

## Assumptions stated on cards
- About 4 characters per token (the brief's round number). Claude's newer tokenizer is closer to 3.1, which makes every token count about 1.3× larger.
- Loop example: 4,500-token start, +1,800 tokens per turn (300 output + 1,500 tool result), 6 calls: illustrative.

## Terms
harness, agent loop, tool, tool schema, tool_use / tool_result, stop reason, budget, context (window), prefix, compaction, skill, MCP, subagent, hook, task, trial (run), grader, hidden tests, pass rate, pass^k, cost per solved task.

## Colour meanings
- green `--accent`: the useful or new part, the winner
- grey `--faint`: already sent / cached / overhead
- red `--red`: ✗, cut-offs, waste

## Card list
`_overview` · The loop: `agent-loop`, `tool-call` · The context: `what-the-model-sees` · Measuring: `agent-evals`
