"""Check claims against the world, not the prose.

A small re-creation of two experiments from The Agent Stack:
the OpenCode answer-only vs tool-loop runs, and the Hermes
cron run that said "completed" while its commands exited 2.
No model is involved: every "agent" here is scripted.
"""
import hashlib, pathlib, subprocess, sys, tempfile

BUGGY = 'def slug(s):\n    return s.strip().lower().replace(" ", "-")\n'
FIXED = ('import re\ndef slug(s):\n'
         '    return re.sub(r"\\s+", "-", s.strip().lower())\n')
TESTS = ('import sys\nfrom slug import slug\n'
         'cases = [("Hello World", "hello-world"),\n'
         '         ("Hello   World", "hello-world"),\n'
         '         ("Hello\\tWorld", "hello-world")]\n'
         'ok = sum(slug(a) == b for a, b in cases)\n'
         'print(ok, "of", len(cases), "pass")\n'
         'sys.exit(0 if ok == len(cases) else 1)\n')
CLAIM = "The whitespace bug is fixed and the tests pass."


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def workspace():
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "slug.py").write_text(BUGGY)
    (d / "test_slug.py").write_text(TESTS)
    return d


def answer_only(d):
    return CLAIM  # no tool calls at all


def tool_loop(d):
    (d / "slug.py").read_text()               # read
    (d / "slug.py").write_text(FIXED)         # edit
    run([sys.executable, "test_slug.py"], d)  # test
    return CLAIM


def verify(d, before):
    """Outside the agent: look at bytes and exit codes."""
    code, out = run([sys.executable, "test_slug.py"], d)
    return {"src_changed": sha(d / "slug.py") != before["src"],
            "tests_same": sha(d / "test_slug.py") == before["test"],
            "test_exit": code, "test_out": out}


def scheduler(d):
    """Records its own lifecycle, not the operation's result."""
    tool = ("import argparse\np = argparse.ArgumentParser()\n"
            "p.add_argument('--mode', choices=['dry-run', 'apply'])\n"
            "p.parse_args()\n")
    (d / "tool.py").write_text(tool)
    codes = [run([sys.executable, "tool.py", "--mode", m], d)[0]
             for m in ("dry-run.", "apply.")]  # stray punctuation
    if codes == [0, 0]:  # a receipt only after verification
        (d / "receipt.json").write_text('{"ok": true}')
    return {"ledger": "completed", "exit_codes": codes,
            "receipt": (d / "receipt.json").exists()}


if __name__ == "__main__":
    print("before any run:", run([sys.executable, "test_slug.py"],
                                 workspace()))
    for agent in (answer_only, tool_loop):
        d = workspace()
        before = {"src": sha(d / "slug.py"),
                  "test": sha(d / "test_slug.py")}
        said = agent(d)
        print(f"{agent.__name__:12} said: {said!r}")
        w = verify(d, before)
        print(f"  world: src_changed {w['src_changed']!s:5}, "
              f"tests_same {w['tests_same']}, "
              f"test_exit {w['test_exit']}, {w['test_out']!r}")
    print("cron-style run:", scheduler(workspace()))
