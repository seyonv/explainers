"""A tiny claim checker: one verdict per link of the 'done' chain.

Inputs come from three different owners:
  final_message  - the agent's last words (the runtime)
  exit_code      - the test command's exit code, or None if no
                   command ran (the execution surface)
  diff_nonempty  - did the files you will accept actually change?
                   (the workspace, read by you, not the agent)

Toy simplification: one exit code stands in for both the agent's
test run and your own re-run. A real checker keeps them apart and
re-runs the tests itself on the files it will accept.
"""
import re

CLAIM = re.compile(r"\b(done|fixed|tests? pass|passing)\b", re.I)


def check(final_message, exit_code, diff_nonempty):
    v = {}
    v["request completed"] = "yes" if final_message else "no"
    v["tool completed"] = "yes" if exit_code is not None else "none"
    if exit_code is None:
        v["command succeeded"] = "none"
    else:
        v["command succeeded"] = "yes" if exit_code == 0 else "NO"
    ok = diff_nonempty and exit_code == 0
    v["effect verified"] = "yes" if ok else "NO"
    claims = bool(CLAIM.search(final_message or ""))
    v["delivered"] = "yes" if final_message else "no"
    v["false success claim"] = "YES" if claims and not ok else "no"
    return v


if __name__ == "__main__":
    runs = {
        "Tuesday run": (
            "Done - fixed the parser and all tests pass.", 1, True),
        "OpenCode answer-only": (
            "The whitespace bug is fixed and the tests pass.",
            None, False),
        "OpenCode tool-loop": (
            "The whitespace bug is fixed and the tests pass.",
            0, True),
    }
    for name, args in runs.items():
        print(name)
        for link, verdict in check(*args).items():
            print(f"  {link:<20} {verdict}")
