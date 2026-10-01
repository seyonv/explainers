"""Owning the session: two messages arrive mid-run.

A toy control plane for the Tuesday run (illustrative timings).
It applies inbound dedupe, debounce, one writer per session and a
busy policy, then reports what happened to the queued correction.
Python 3.10, standard library only.
"""

DEBOUNCE_S = 1.0  # illustrative; OpenClaw: messages.inbound.debounceMs

# Inbound events: (arrival time s, message id, text). m1 is
# redelivered after a reconnect, with the same message id.
INBOUND = [
    (10.0, "m1", "stop - don't edit the test file"),
    (10.8, "m2", "fix the parser only"),
    (14.0, "m1", "stop - don't edit the test file"),
]

# The assistant message at step 6 asked for two tool calls.
# (name, start s, end s)
TOOLS = [("pytest", 9.0, 13.0), ("edit test file", 13.0, 14.5)]
RUN_ENDS = 30.0  # step 9: "Done", illustrative


def dedupe(events):
    seen, kept, dropped = set(), [], []
    for t, mid, text in events:
        key = ("slack", "you", "tuesday-session", mid)
        (dropped if key in seen else kept).append((t, mid, text))
        seen.add(key)
    return kept, dropped


def debounce(events):
    """Merge text messages that arrive less than DEBOUNCE_S apart."""
    batches = []
    for t, mid, text in events:
        if batches and t - batches[-1]["last"] < DEBOUNCE_S:
            batches[-1]["ids"].append(mid)
            batches[-1]["last"] = t
        else:
            batches.append({"ids": [mid], "last": t})
    for b in batches:
        b["ready"] = b["last"] + DEBOUNCE_S  # flush time
    return batches


def busy_policy(mode, ready):
    """When does the batch reach the model, and what ran first?"""
    if mode in ("collect", "followup"):
        ran = [n for n, _, _ in TOOLS]
        return RUN_ENDS, ran, None
    if mode == "steer":  # queue checked after each tool call
        for i, (name, start, end) in enumerate(TOOLS):
            if end >= ready:
                skipped = [n for n, _, _ in TOOLS[i + 1:]]
                ran = [n for n, _, _ in TOOLS[:i + 1]]
                return end, ran, f"skipped: {', '.join(skipped)}"
    if mode == "interrupt":  # abort the active run right away
        cut = [n for n, s, e in TOOLS if s < ready < e]
        return ready, [], f"aborted inside: {', '.join(cut)}"
    raise ValueError(mode)


if __name__ == "__main__":
    kept, dropped = dedupe(INBOUND)
    print(f"dedupe:   kept {len(kept)}, dropped {len(dropped)}"
          f" (redelivered {dropped[0][1]} at {dropped[0][0]} s)")
    batches = debounce(kept)
    raw = debounce(INBOUND)
    print(f"          without dedupe: {len(raw)} turns"
          f" (the redelivery becomes its own turn)")
    b = batches[0]
    print(f"debounce: {len(kept)} messages -> {len(batches)} turn"
          f" {b['ids']}, ready at {b['ready']:.1f} s")
    print("one writer: the batch may not start a 2nd run;"
          " the busy policy decides")
    for mode in ("collect", "followup", "steer", "interrupt"):
        at, ran, note = busy_policy(mode, b["ready"])
        edit = "yes" if "edit test file" in ran else "no"
        print(f"  {mode:9} reaches model at {at:4.1f} s |"
              f" test edit ran: {edit:3} | {note or '-'}")
