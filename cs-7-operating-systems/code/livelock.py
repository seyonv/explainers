"""Livelock: polite threads that back off, simulated in time slots.

Model (a simulation, not real threads): after a collision every
contender picks a backoff slot; if the earliest slot is shared, they
collide again. Two contenders in lockstep is OSTEP's trylock livelock.
"""
import random

CAP = 1_000                      # give up after this many retries


def fixed(c, rng):               # everyone waits the same 1 slot
    return 1


def jitter(w):                   # uniform in 0 .. w-1 slots
    return lambda c, rng: rng.randrange(w)


def expo(c, rng):                # Ethernet: 0 .. 2^min(c,10) - 1
    return rng.randrange(2 ** min(c, 10))


def retries(policy, n, rng):
    """Retries until one contender goes alone, and slots spent."""
    slots = 0
    for c in range(1, CAP + 1):  # c = collisions so far
        picks = [policy(c, rng) for _ in range(n)]
        first = min(picks)
        slots += first + 1       # wait, then one slot to try
        if picks.count(first) == 1:
            return c, slots      # earliest one got through
    return None, slots           # still colliding: livelock


def trace(policy, rng, rounds=4):
    """Two threads: T1 holds L1 wants L2, T2 holds L2 wants L1."""
    t = 0
    for c in range(1, rounds + 1):
        a, b = policy(c, rng), policy(c, rng)
        print(f"t={t:<2} both trylock, both fail, release;"
              f" waits T1={a} T2={b}")
        if a != b:
            who = "T1" if a < b else "T2"
            print(f"t={t + 1 + min(a, b):<2} {who} retries alone,"
                  f" gets both locks")
            return
        t += 1 + a


def exact(w, n):
    """Mean retries for a fixed window w: 1 / P(earliest is unique)."""
    p = sum(n / w * ((w - 1 - k) / w) ** (n - 1) for k in range(w))
    return 1 / p


def average(policy, n, trials=10_000, seed=7):
    rng = random.Random(seed)
    runs = [retries(policy, n, rng) for _ in range(trials)]
    stuck = sum(r is None for r, _ in runs)
    mr = sum(r for r, _ in runs if r) / (trials - stuck)
    ms = sum(s for r, s in runs if r) / (trials - stuck)
    return f"{mr:6.2f} retries {ms:6.2f} slots, stuck {stuck}"


if __name__ == "__main__":
    print("fixed 1-slot wait:")
    trace(fixed, random.Random(1))
    print("random 0..1 wait, seed 5:")
    trace(jitter(2), random.Random(5))
    print("fixed wait: every pick is equal, so no trial ever ends")
    for n in (2, 8, 32):
        print(f"\n{n} contenders, 10,000 trials, seed 7")
        for w in (2, 8):
            e = exact(w, n)
            sim = average(jitter(w), n) if e < 50 else "not run"
            print(f"  random 0..{w - 1}: exact {e:,.2f}  sim {sim}")
        print(f"  binary exponential:  sim {average(expo, n)}")
