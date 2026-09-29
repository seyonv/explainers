# Agents basics: hands-on lab

Two small builds. On Day 1 you write a whole coding agent in about 60 lines. On Day 3 you measure how often it works.

You need [uv](https://docs.astral.sh/uv/) and an Anthropic API key (`export ANTHROPIC_API_KEY=...`). All commands below run from this `labs/` folder. `uv run --python 3.12 --with anthropic` fetches Python 3.12 and the SDK on first use, so there is nothing to install.

## Files

| File | What it is |
|---|---|
| `loop.py` | The Day 1 agent. It has one tool, `bash`, and one loop: ask the model, run the commands it asks for, send back the output, and repeat until it says it is done or a budget runs out. `run(task_dir)` can be imported. |
| `tasks/NN-name/` | 10 tiny Python projects, each with one planted bug. Each has `TASK.md` (the instruction the agent gets), the buggy module and a visible `test_*.py`. |
| `hidden/NN-name/test_hidden.py` | The grader's tests. The agent never sees them. They check more cases than the visible test does, so special-casing the visible test does not pass. |
| `run_tasks.py` | The Day 3 harness. It copies each task to a fresh temp folder, runs `loop.run` on it, copies the hidden test in afterwards and runs it. It writes one JSON line per run to `runs.jsonl` and prints the summary. |
| `_selftest.py` | Checks the plumbing without an API key. A fake client plays back scripted replies. It covers parallel tool calls, every stop reason, both budgets, truncation, timeout, and that every task ships broken. |

The bugs are: an off-by-one error (01), a wrong comparison (02), a mutable default argument (03), a regex that only reads one digit (04), integer division (05), a dict key typo (06), string cleanup (07), leap-year logic (08), unsorted input (09) and a shared dict that gets mutated (10).

## Day 1: build the loop (about 1 hour)

Read `loop.py` from top to bottom. It is the whole agent. Then run it on a copy of the first task, so the agent can't reach `../../hidden` and the original stays broken for next time:

```bash
cp -r tasks/01-paginate /tmp/day1 && uv run --python 3.12 --with anthropic loop.py /tmp/day1
```

If you do run it in place (`loop.py tasks/01-paginate`), reset the task afterwards with `git checkout -- tasks/01-paginate`.

What to notice in the code:
- **`stop_reason` drives everything.** `end_turn` means done. `tool_use` means run the tools. Anything else (`max_tokens`, `refusal`, ...) ends the run as a failure. The loop never guesses.
- **Parallel calls.** One reply can contain several `tool_use` blocks. They all run, and all their results go back in **one** user message, in the same order as the calls.
- **Errors are data.** A failing command returns `is_error: true` with its output, so the model can read the traceback and try again.
- **Budgets.** There is a 30-step cap and a $0.50 cap, both counted from `usage`. The dollar cap is checked before each call, so one expensive turn can overshoot it slightly.

## Day 3: measure it (about 1 hour)

```bash
uv run --python 3.12 --with anthropic run_tasks.py      # 10 tasks x 3 repeats = 30 runs
uv run --python 3.12 --with anthropic run_tasks.py 1    # quicker: 10 runs
```

The summary gives:
- **Pass rate** (x/30), with a **95% Wilson interval**. 30 runs is a small sample, so the interval is wide. Read it before comparing two versions of your agent.
- **Per-task majority** (x/10): the tasks the agent solved in at least 2 of its 3 tries.
- **Total cost** and **cost per solved task**, which is total cost divided by passing runs. A cheap agent that rarely succeeds is not cheap.

`runs.jsonl` is appended to, never overwritten, so older batches stay in it. The summary covers only the batch that just ran.

## Check the plumbing (no API key needed)

```bash
uv run --python 3.12 --with anthropic _selftest.py
```

## What you should see

**Self-test.** This was run on 2026-09-29. It prints 30 `ok` lines and ends with:

```
ok   all 3 results in ONE user message, in call order
ok   non-zero exit -> is_error
...
ok   dollar cap stops the loop (turns=2, $0.6)
ok   step cap stops the loop at 30
ok   long output truncated to head+tail (4026 chars)
ok   timeout kills the command (1.0s)
...
ok   10-config: visible and hidden tests fail as shipped
ok   runner writes one JSON line per run
ok   do-nothing agent fails every hidden test
ok   Wilson interval at 0/30 and 30/30
ok   Wilson interval at 24/30 = 0.627..0.905

all checks passed
```

**Without an API key.** This was run on 2026-09-29. `loop.py` and `run_tasks.py` stop at once with this error:

```
TypeError: "Could not resolve authentication method. Expected one of api_key, auth_token, or credentials to be set. ...
```

**Day 1 and Day 3 runs.** Your numbers will vary from run to run: models are random, so the number of turns, the tokens and even whether a task passes change each time. Before each run, write down what you expect (turns, cost, pass rate) and compare it with what the script prints. On Day 3, read the Wilson interval before trusting any difference between two batches.
