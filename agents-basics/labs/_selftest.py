"""Check the lab's plumbing without an API key: a fake client plays back scripted replies.
Usage: uv run --python 3.12 --with anthropic _selftest.py"""
import json, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace as NS
import loop, run_tasks

LABS = Path(__file__).resolve().parent

def text(t): return NS(type="text", text=t)
def call(id, cmd): return NS(type="tool_use", id=id, name="bash", input={"command": cmd})
def reply(stop, *blocks, tin=100, tout=50):
    return NS(stop_reason=stop, content=list(blocks), usage=NS(input_tokens=tin, output_tokens=tout))

class FakeClient:
    """Stands in for anthropic.Anthropic(): `script(n)` returns the n-th reply; every request is recorded."""
    def __init__(self, script):
        self.script, self.requests = script, []
        self.messages = self
    def create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return self.script(len(self.requests) - 1)

def check(name, ok):
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not ok: sys.exit(1)

quiet = lambda *a: None
task = LABS / "tasks" / "01-paginate"

# 1. Parallel tool calls: three calls in one turn -> one user message, three results, in call order.
first = reply("tool_use", text("Looking."), call("t1", "echo one"), call("t2", "exit 3"), call("t3", "cat"))
fake = FakeClient(lambda n: first if n == 0 else reply("end_turn", text("Done.")))
stats = loop.run(task, client=fake, log=quiet)
sent = fake.requests[1]["messages"]
results = sent[-1]["content"]
check("end_turn after tools -> done in 2 turns", stats["status"] == "done" and stats["turns"] == 2)
check("history is user, assistant, user", [m["role"] for m in sent] == ["user", "assistant", "user"])
check("assistant turn keeps every block", sent[1]["content"] == first.content)
check("all 3 results in ONE user message, in call order", [r["tool_use_id"] for r in results] == ["t1", "t2", "t3"])
check("non-zero exit -> is_error", [r["is_error"] for r in results] == [False, True, False])
check("output reaches the model", "one" in results[0]["content"] and "[exit code 3]" in results[1]["content"])
check("request carries model, tools and system", fake.requests[0]["model"] == loop.MODEL and fake.requests[0]["tools"] == [loop.TOOL])
check("tokens and cost from usage", stats["input_tokens"] == 200 and stats["output_tokens"] == 100
      and abs(stats["cost"] - (200 * loop.PRICE_IN + 100 * loop.PRICE_OUT)) < 1e-9)

# 2. Stop reasons other than end_turn/tool_use are failures.
for reason in ["max_tokens", "refusal"]:
    s = loop.run(task, client=FakeClient(lambda n: reply(reason, text("cut off"))), log=quiet)
    check(f"{reason} -> stopped:{reason} after 1 turn", s["status"] == f"stopped:{reason}" and s["turns"] == 1)

# 3. Budgets.
pricey = lambda n: reply("tool_use", call(f"c{n}", "true"), tin=100_000, tout=10_000)  # $0.30 a turn
s = loop.run(task, client=FakeClient(pricey), log=quiet)
check(f"dollar cap stops the loop (turns={s['turns']}, ${s['cost']})", s["status"] == "dollar_cap" and s["turns"] == 2)
s = loop.run(task, client=FakeClient(lambda n: reply("tool_use", call(f"c{n}", "true"))), log=quiet)
check(f"step cap stops the loop at {loop.MAX_STEPS}", s["status"] == "step_cap" and s["turns"] == loop.MAX_STEPS)

# 4. The bash tool itself.
out, failed = loop.bash("python3 -c \"print('x' * 10000)\"", task)
check(f"long output truncated to head+tail ({len(out)} chars)", "chars cut" in out and len(out) < 4100 and not failed)
t0 = time.time()
out, failed = loop.bash("sleep 5; echo never", task, timeout=1)
check(f"timeout kills the command ({time.time() - t0:.1f}s)", failed and "timed out" in out and time.time() - t0 < 3)
check("stdin is closed (cat returns at once)", loop.bash("cat", task, timeout=5)[1] is False)
check("PAGER=cat", loop.bash("echo $PAGER", task)[0].startswith("cat"))

# 5. Every task ships broken: visible and hidden tests both fail on the original code.
def passes(work, module):
    return subprocess.run([sys.executable, "-m", "unittest", module], cwd=work, capture_output=True, timeout=60).returncode == 0
for t in sorted((LABS / "tasks").iterdir()):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / t.name
        shutil.copytree(t, work)
        visible = next(work.glob("test_*.py")).stem
        shutil.copy(LABS / "hidden" / t.name / "test_hidden.py", work)
        check(f"{t.name}: visible and hidden tests fail as shipped", not passes(work, visible) and not passes(work, "test_hidden"))

# 6. The runner end to end, with an agent that gives up at once: 10 rows, all graded FAIL.
with tempfile.TemporaryDirectory() as tmp:
    out_file = Path(tmp) / "runs.jsonl"
    rows = run_tasks.main(repeats=1, client=FakeClient(lambda n: reply("end_turn", text("I give up."))), out=out_file, log=quiet)
    lines = [json.loads(l) for l in out_file.read_text().splitlines()]
    check("runner writes one JSON line per run", len(lines) == 10 and lines == rows)
    check("do-nothing agent fails every hidden test", not any(r["passed"] for r in rows))
check("Wilson interval at 0/30 and 30/30", run_tasks.wilson(0, 30)[0] == 0 and run_tasks.wilson(30, 30)[1] == 1)
lo, hi = run_tasks.wilson(24, 30)
check(f"Wilson interval at 24/30 = {lo:.3f}..{hi:.3f}", abs(lo - 0.627) < 0.001 and abs(hi - 0.905) < 0.001)
print("\nall checks passed")
