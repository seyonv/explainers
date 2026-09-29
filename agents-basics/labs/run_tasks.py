"""Run the agent on every task several times, grade each run with a hidden test, report the numbers.
Usage: uv run --python 3.12 --with anthropic run_tasks.py [repeats]"""
import json, math, shutil, subprocess, sys, tempfile, time
from collections import defaultdict
from pathlib import Path
import anthropic
import loop

LABS = Path(__file__).resolve().parent

def grade(work):
    """Copy the hidden test in only after the agent has finished, then run it."""
    shutil.copy(LABS / "hidden" / work.name / "test_hidden.py", work)
    try:
        p = subprocess.run([sys.executable, "-m", "unittest", "test_hidden"], cwd=work,
                           capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def wilson(passes, n, z=1.96):
    """95% confidence interval for a pass rate; behaves sensibly at 0/n and n/n."""
    if n == 0: return 0.0, 1.0
    p = passes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)

def main(repeats=3, client=None, out=LABS / "runs.jsonl", log=print):
    tasks = sorted(d for d in (LABS / "tasks").iterdir() if d.is_dir())
    rows = []
    with open(out, "a") as f:
        for task in tasks:
            for rep in range(repeats):
                with tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp) / task.name  # fresh copy: no leftovers, no view of hidden/
                    shutil.copytree(task, work)
                    log(f"\n=== {task.name} run {rep + 1}/{repeats} ===")
                    start = time.time()
                    try:
                        stats = loop.run(work, client=client, log=log)
                    except anthropic.APIError as e:  # rate limit, overload...: fail this run, not the batch
                        stats = {"status": f"error:{type(e).__name__}", "turns": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
                    row = {"task": task.name, "repeat": rep, "model": loop.MODEL, **stats,
                           "passed": grade(work), "seconds": round(time.time() - start, 1)}
                log(f"--> {'PASS' if row['passed'] else 'FAIL'} ({row['status']}, ${row['cost']:.4f})")
                f.write(json.dumps(row) + "\n"); f.flush()
                rows.append(row)
    report(rows, log)
    return rows

def report(rows, log=print):
    n, passes = len(rows), sum(r["passed"] for r in rows)
    by_task = defaultdict(list)
    for r in rows: by_task[r["task"]].append(r["passed"])
    majority = sum(sum(v) * 2 > len(v) for v in by_task.values())
    lo, hi = wilson(passes, n)
    cost = sum(r["cost"] for r in rows)
    log(f"\npass rate: {passes}/{n} = {passes / max(n, 1):.0%}   (95% Wilson interval {lo:.0%} to {hi:.0%})")
    log(f"per-task majority: {majority}/{len(by_task)} tasks solved in most of their runs")
    for task, v in sorted(by_task.items()): log(f"  {task:16} {''.join('P' if x else '.' for x in v)}")
    log(f"total cost: ${cost:.2f}   cost per solved task: " + (f"${cost / passes:.3f}" if passes else "n/a (none solved)"))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
