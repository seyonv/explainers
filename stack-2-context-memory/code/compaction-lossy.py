"""What survives when a compactor clips tool results before it
summarizes? The Tuesday run's step-6 pytest result, clipped the way
OpenCode v1.18.30's compaction serializer clips completed tool
results (keep the first 2,000 characters).

The pytest output below is illustrative (written for this card); the
2,000-character cap is from OpenCode's compaction.ts, as reported in
The Agent Stack, "OpenCode Architecture - Part 2". Where the exit code
sits in a tool result is harness-specific: here it is appended after
the output, the position OpenCode gave the nested src/AGENTS.md rule.
"""
import re

CAP = 2_000  # characters kept per tool result before summarizing

TRACE = """\
    def test_parse_due_date():
        inv = Invoice.from_dict(SAMPLE)
>       assert inv.due == date(2026, 3, 31)
E       AssertionError: assert datetime.date(2026, 3, 1) == \
datetime.date(2026, 3, 31)
E        +  where datetime.date(2026, 3, 1) = <Invoice #1042>.due
"""

PYTEST = (
    "============ test session starts ============\n"
    "platform darwin -- Python 3.10.14, pytest-8.3.2\n"
    "rootdir: /work/invoice-tools\n"
    "collected 42 items\n\n"
    "tests/test_currency.py ........            [ 19%]\n"
    "tests/test_dates.py ...F......             [ 42%]\n"
    "tests/test_export.py ..........            [ 66%]\n"
    "tests/test_totals.py ..............        [100%]\n\n"
    "================= FAILURES ==================\n"
    "____________ test_parse_due_date ____________\n\n"
    + TRACE * 6  # stand-in for a long traceback with locals
    + "\ninvoice_tools/dates.py:57: AssertionError\n"
    "========= short test summary info ===========\n"
    "FAILED tests/test_dates.py::test_parse_due_date"
    " - AssertionError\n"
    "======== 1 failed, 41 passed in 0.31s =======\n"
)
RESULT = PYTEST + "[exit code: 1]\n"

FACTS = {  # what a later step needs to know, and how to spot it
    "an F in the progress line": r"\.F\.",
    "a traceback (AssertionError)": r"AssertionError",
    "failing test name": r"FAILED tests/test_dates",
    "1 failed, 41 passed": r"1 failed, 41 passed",
    "exit code 1": r"\[exit code: 1\]",
}


def clip(text, cap=CAP):
    """Keep the head of a tool result, as the serializer does."""
    return text if len(text) <= cap else text[:cap]


def pin_status(text):
    """Pull the verifiable status out BEFORE clipping, verbatim."""
    m = re.search(r"(\d+ failed, \d+ passed)", text)
    code = re.search(r"\[exit code: (\d+)\]", text)
    return {"cmd": "pytest",
            "counts": m.group(1) if m else "unknown",
            "exit": int(code.group(1)) if code else None}


def inflated(summary, pinned):
    """Flag a summary that says 'passing' when the pin says not."""
    says_pass = re.search(r"tests? (are )?pass", summary, re.I)
    return bool(says_pass) and pinned["exit"] != 0


if __name__ == "__main__":
    kept = clip(RESULT)
    print(f"tool result: {len(RESULT):,} chars; "
          f"kept: {len(kept):,}; cut: {len(RESULT) - len(kept):,}")
    for name, pat in FACTS.items():
        pos = re.search(pat, RESULT).start()
        live = "kept" if re.search(pat, kept) else "CUT "
        print(f"  {live}  at char {pos:>5,}  {name}")
    pin = pin_status(RESULT)
    print("pinned before clipping:", pin)
    s7 = "Fixed due-date parsing; tests passing."
    print("step-7 summary inflated?", inflated(s7, pin))
