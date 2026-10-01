"""Retry, replay, resume: count duplicate side effects.

A six-step run (the dependency upgrade from The Agent Stack, Part 4).
Two steps change the outside world: push the branch, open the PR.
After any step the worker may crash with probability P_CRASH, *after*
the step's effect happened but *before* the runtime recorded it.
That gap is the "ambiguity" the post says to close.

Four recovery strategies:
  restart          plain retry loop: start the whole run again
  restart+key      same, but side effects carry an idempotency key
  journal          resume from the first unrecorded step
  journal+key      resume, and side effects carry a key
P_CRASH is illustrative; the counts it produces are computed.
"""
import random

STEPS = ["edit", "test", "push", "wait_ci", "approve", "open_pr"]
EFFECTS = {"push", "open_pr"}
P_CRASH = 0.2


class Remote:
    """The outside world: a git host that may dedupe by key."""

    def __init__(self):
        self.effects = []
        self.seen = set()

    def do(self, run_id, step, use_key):
        key = f"{run_id}:{step}"
        if use_key and key in self.seen:
            return  # same key again: no new effect
        self.seen.add(key)
        self.effects.append(step)


def run(strategy, rng, run_id="run-1"):
    use_key = strategy.endswith("+key")
    journal = strategy.startswith("journal")
    remote, done, executions = Remote(), set(), 0
    while len(done) < len(STEPS):
        for step in STEPS:
            if journal and step in done:
                continue  # recorded: skip, don't redo
            executions += 1
            if step in EFFECTS:
                remote.do(run_id, step, use_key)
            if rng.random() < P_CRASH:
                if not journal:
                    done.clear()  # process memory is gone
                break  # crash before recording
            done.add(step)
    dupes = len(remote.effects) - len(EFFECTS)
    return dupes, executions


def trial(strategy, n=100_000, seed=7):
    rng = random.Random(seed)
    dupes = execs = hit = 0
    for _ in range(n):
        d, e = run(strategy, rng)
        dupes, execs, hit = dupes + d, execs + e, hit + (d > 0)
    return dupes / n, execs / n, hit / n


if __name__ == "__main__":
    print(f"{len(STEPS)} steps, 2 side effects, P_CRASH={P_CRASH}")
    print(f"{'strategy':<13}{'dupes/run':>10}"
          f"{'steps run':>11}{'runs w/ dupe':>14}")
    for s in ["restart", "restart+key", "journal", "journal+key"]:
        d, e, h = trial(s)
        print(f"{s:<13}{d:>10.2f}{e:>11.2f}{h:>13.1%}")
    p = P_CRASH
    print(f"check, journal: 2 x p/(1-p) = {2 * p / (1 - p):.2f}"
          f" dupes; 6/(1-p) = {6 / (1 - p):.2f} steps")
    q = 1 - p
    print(f"check, restart: (1/q^6 - 1)/(1-q) = "
          f"{(1 / q**6 - 1) / (1 - q):.2f} steps")
