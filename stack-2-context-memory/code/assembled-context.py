"""Transcript vs working set for the Tuesday run (illustrative).

The transcript is the record the runtime keeps. The working set
is what one model call receives. The assembler below derives the
working set from the record; it never edits the record.

All token counts are illustrative, not measured.
"""

PREFIX = [  # (piece, owner, tokens): sent on every call
    ("tool definitions", "harness config", 6000),
    ("system prompt", "harness", 4200),
    ("AGENTS.md", "repo maintainer", 1800),
]

TRANSCRIPT = [  # (piece, owner, tokens, facts it carries)
    ("task: fix test_parse_due_date", "user", 150, ""),
    ("steps 1-5: read, search, edit", "runtime log", 52000, ""),
    ("step 6: pytest", "execution surface", 3000,
     "1 failed, 41 passed, exit 1"),
]

SUMMARY = ("step 7 summary", "model-written, runtime-chosen", 60,
           "Fixed due-date parsing; tests passing.")

COMPACT_AT = 60000  # the harness's own budget, illustrative


def assemble(history, keep_last=0):
    """Return (working set, compacted?) for the next call."""
    size = sum(t for _, _, t in PREFIX)
    size += sum(h[2] for h in history)
    if size <= COMPACT_AT:
        return PREFIX + history, False
    task, rest = history[0], history[1:]
    kept = rest[len(rest) - keep_last:] if keep_last else []
    return PREFIX + [task, SUMMARY] + kept, True


def tokens(items):
    return sum(i[2] for i in items)


def shows(items, text):
    return any(text in (i[3] if len(i) > 3 else "") for i in items)


if __name__ == "__main__":
    call7 = tokens(PREFIX) + tokens(TRANSCRIPT)
    print(f"call 7 would send {call7:,} > {COMPACT_AT:,}")
    record = TRANSCRIPT + [SUMMARY]
    print(f"transcript (record): {tokens(record):,} tokens")
    for keep in (0, 1):
        ws, compacted = assemble(TRANSCRIPT, keep_last=keep)
        print(f"\nworking set at step 8, keep_last={keep}:")
        for piece, owner, n, *_ in ws:
            print(f"  {piece:<31} {n:>6,}  {owner}")
        print(f"  {'total':<31} {tokens(ws):>6,}")
        print("  sees 'exit 1':", shows(ws, "exit 1"),
              "| record has it:", shows(record, "exit 1"))
