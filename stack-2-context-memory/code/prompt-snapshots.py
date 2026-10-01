"""When does an edited USER.md reach the model, and what does it cost
in prompt-cache misses? Four refresh policies on the Tuesday run.

All sizes are illustrative. The cache model is simple: a request
reuses the longest run of leading blocks it shares with the previous
request; everything after the first difference is prefilled again.
"""

SYS = 10_000      # system prompt tokens (illustrative)
TURN = 1_800      # tokens each step appends (illustrative)
SUMMARY = 1_000   # compaction summary tokens (illustrative)
NOTE = 50         # an "USER.md changed" note (illustrative)
EDIT_AFTER = 4    # USER.md edited between steps 4 and 5
COMPACT_AT = 7    # harness compacts before the step-7 call
STEPS = 9


def request(step, policy):
    """Blocks (label, tokens) sent on one step's model call."""
    new = step > EDIT_AFTER
    if policy == "snapshot per session":
        # rebuilt only at a refresh boundary (here: compaction)
        v = "v2" if new and step >= COMPACT_AT else "v1"
        sys = (f"sys:{v}", SYS)
    elif policy == "rebuild, no clock":
        # rebuilt each call; bytes change only when USER.md does
        sys = (f"sys:{'v2' if new else 'v1'}", SYS)
    elif policy == "rebuild with clock":
        # rebuilt each call; the volatile tier carries a timestamp
        sys = (f"sys:{'v2' if new else 'v1'}:t{step}", SYS)
    else:  # inject on change: snapshot stays, note goes at tail
        sys = ("sys:v1" if step < COMPACT_AT else
               f"sys:{'v2' if new else 'v1'}", SYS)
    if step < COMPACT_AT:
        hist = [(f"t{i}", TURN) for i in range(1, step)]
    else:
        hist = [("summary", SUMMARY)]
        hist += [(f"t{i}", TURN) for i in range(COMPACT_AT, step)]
    if policy == "inject on change" and new and step < COMPACT_AT:
        at = EDIT_AFTER  # note sits after turn 4 in history
        hist = hist[:at] + [("note:v2", NOTE)] + hist[at:]
    return [sys] + hist


def sees_new(blocks):
    return any("v2" in label for label, _ in blocks)


def simulate(policy):
    prev, missed, first = [], 0, None
    for step in range(1, STEPS + 1):
        req = request(step, policy)
        shared = 0
        for a, b in zip(prev, req):
            if a != b:
                break
            shared += 1
        missed += sum(n for _, n in req[shared:])
        if first is None and sees_new(req):
            first = step
        prev = req
    total = sum(sum(n for _, n in request(s, policy))
                for s in range(1, STEPS + 1))
    return first, missed, total


if __name__ == "__main__":
    print(f"{'policy':22}{'sees edit':>10}"
          f"{'prefilled':>11}{'of input':>10}")
    for p in ["snapshot per session", "rebuild, no clock",
              "rebuild with clock", "inject on change"]:
        first, missed, total = simulate(p)
        print(f"{p:22}{'step ' + str(first):>10}"
              f"{missed:>11,}{missed / total:>10.0%}")
