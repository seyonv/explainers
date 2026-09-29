"""A whole coding agent: one model, one tool (bash), one loop.
Usage: uv run --python 3.12 --with anthropic loop.py tasks/01-paginate"""
import os, signal, subprocess, sys
from pathlib import Path
import anthropic

MODEL = "claude-sonnet-5-5"
# Claude Sonnet 5.5 list prices: $2 per million input tokens, $10 per million output (Sept 2026).
PRICE_IN, PRICE_OUT = 2.00 / 1_000_000, 10.00 / 1_000_000
MAX_STEPS, MAX_DOLLARS, KEEP = 30, 0.50, 2000
SYSTEM = "You fix bugs in a small Python project. The current directory is the project. Use the bash tool; run tests with `python3 -m unittest -v`. When the tests pass, reply with a one-line summary."
TOOL = {"name": "bash", "description": "Run a shell command in the project directory and return its output.",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}

def bash(command, cwd, timeout=60):
    p = subprocess.Popen(command, shell=True, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, env={**os.environ, "PAGER": "cat"}, start_new_session=True)
    try:
        out = p.communicate(timeout=timeout)[0].decode(errors="replace") + f"\n[exit code {p.returncode}]"
        failed = p.returncode != 0
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)  # kill the whole process group, not just the shell
        p.communicate()
        out, failed = f"[timed out after {timeout}s]", True
    if len(out) > 2 * KEEP:
        out = out[:KEEP] + f"\n... [{len(out) - 2 * KEEP} chars cut] ...\n" + out[-KEEP:]
    return out, failed

def run(task_dir, client=None, log=print):
    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": (Path(task_dir) / "TASK.md").read_text()}]
    turns = tokens_in = tokens_out = 0
    cost = lambda: tokens_in * PRICE_IN + tokens_out * PRICE_OUT
    while True:
        if turns >= MAX_STEPS: status = "step_cap"; break
        if cost() >= MAX_DOLLARS: status = "dollar_cap"; break
        r = client.messages.create(model=MODEL, max_tokens=16000, system=SYSTEM, tools=[TOOL], messages=messages)
        turns += 1; tokens_in += r.usage.input_tokens; tokens_out += r.usage.output_tokens
        messages.append({"role": "assistant", "content": r.content})  # keep every block, thinking included
        for b in r.content:
            if b.type == "text" and b.text.strip(): log(f"[{turns}] {b.text.strip()[:200]}")
        if r.stop_reason == "end_turn": status = "done"; break
        if r.stop_reason != "tool_use": status = f"stopped:{r.stop_reason}"; break  # max_tokens, refusal, ...
        results = []  # one tool_result per tool_use, in call order, all in ONE user message
        for b in [b for b in r.content if b.type == "tool_use"]:
            out, failed = bash(b.input.get("command", "echo missing command; exit 2"), task_dir)
            log(f"[{turns}] $ {b.input.get('command', '')[:100]}  -> {'ERROR' if failed else 'ok'}, {len(out)} chars")
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": failed})
        messages.append({"role": "user", "content": results})
    stats = {"status": status, "turns": turns, "input_tokens": tokens_in, "output_tokens": tokens_out, "cost": round(cost(), 4)}
    log(f"{status}: {turns} turns, {tokens_in} in / {tokens_out} out tokens, ${stats['cost']:.4f}")
    return stats

if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1])["status"] == "done" else 1)
