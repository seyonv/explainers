"""The network is unreliable: six ways a call goes silent.

A caller sends one resize request to a worker and waits for a
reply. Six different faults all look the same to the caller (no
reply before the timeout), but the work ran in four of them.
Delays are illustrative; r = 100 ms comes from the course's image
service (10 conversions/s per core).
"""

D, R = 5, 100          # max one-way delay, max processing (ms)
TIMEOUT = 2 * D + R    # the source's 2d + r -> 110 ms

# (case, request delay, worker pause, reply delay) in ms.
# None means it never arrives (lost packet, crashed worker).
CASES = [
    ("healthy",          2,    0,    2),
    ("request lost",     None, 0,    2),
    ("request queued",   400,  0,    2),
    ("node crashed",     2,    None, 2),
    ("node paused (GC)", 2,    1000, 2),
    ("response lost",    2,    0,    None),
    ("response delayed", 2,    0,    300),
]


def call(req, pause, reply):
    """Return (what the caller sees, when the work ran or None)."""
    if req is None or pause is None:
        return "timeout", None             # the work never ran
    done = req + pause + R                 # the worker finishes here
    if reply is None or done + reply > TIMEOUT:
        return "timeout", done             # ran, but caller gave up
    return f"reply at {done + reply} ms", done


if __name__ == "__main__":
    print(f"timeout = 2d + r = 2*{D} + {R} = {TIMEOUT} ms")
    ran = silent = 0
    for name, *delays in CASES:
        seen, done = call(*delays)
        work = "no" if done is None else f"yes, at {done} ms"
        print(f"{name:<17} {seen:<16} work ran: {work}")
        if seen == "timeout":
            silent += 1
            ran += done is not None
    print(f"silent cases: {silent}; work ran anyway in {ran}")
