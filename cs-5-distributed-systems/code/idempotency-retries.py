"""Idempotency and retries, in Python.

1. Capped exponential backoff with full jitter (AWS, Brooker 2015):
   sleep = random(0, min(cap, base * 2**n)), base 100 ms, cap 10 s.
2. Why jitter: 100 clients fail at once and retry; peak retries
   in any 10 ms slot, with and without jitter.
3. Retry amplification: r retries at each of L layers ->
   (r + 1)**L attempts on the bottom layer (SRE book: 4**3 = 64).
4. An idempotency-key table on the server: the response to the
   first charge is lost, the client retries, and the customer is
   charged once with a key and twice without.
"""
import random
import uuid

BASE, CAP = 0.100, 10.0          # seconds


def ceiling(n, base=BASE, cap=CAP):
    """Upper bound of the n-th retry's sleep (n = 0 is the first)."""
    return min(cap, base * 2 ** n)


def full_jitter(n, rng):
    return rng.uniform(0, ceiling(n))


def schedule(k=6, seed=1):
    rng = random.Random(seed)
    print("retry  ceiling   E[sleep]  one draw  E[total wait]")
    total = 0.0
    for n in range(k):
        c = ceiling(n)
        total += c / 2
        d = full_jitter(n, rng)
        print(f"{n + 1:>5}  {c * 1000:>6.0f} ms "
              f"{c / 2 * 1000:>6.0f} ms {d * 1000:>6.0f} ms "
              f"{total * 1000:>8.0f} ms")
    first = next(n for n in range(64) if BASE * 2 ** n >= CAP)
    print(f"cap reached at retry {first + 1}: "
          f"100 ms * 2**{first} = {BASE * 2 ** first:g} s -> 10 s")


def herd(clients=100, n=0, slot=0.010, seed=1):
    """Peak retries in one slot when all clients fail at t = 0."""
    rng = random.Random(seed)
    no_jit = clients                         # all at t = ceiling(n)
    slots = [0] * int(round(ceiling(n) / slot))
    for _ in range(clients):
        t = full_jitter(n, rng)
        slots[min(int(t / slot), len(slots) - 1)] += 1
    print(f"{clients} clients, retry window {ceiling(n) * 1000:.0f}"
          f" ms, {slot * 1000:.0f} ms slots: peak without jitter "
          f"{no_jit}, with full jitter {max(slots)} "
          f"(mean {clients / len(slots):g})")


def amplification():
    print("attempts reaching the bottom layer = (r + 1) ** layers")
    for r in (1, 2, 3):
        row = "  ".join(f"L={L}: {(r + 1) ** L:>3}"
                        for L in (1, 2, 3, 4))
        print(f"  r = {r} retries ({r + 1} attempts)  {row}")


class Server:
    """Charges an account; remembers each idempotency key's reply."""

    def __init__(self):
        self.balance = 0
        self.keys = {}                     # key -> (request, reply)

    def charge(self, amount, key=None):
        req = ("charge", amount)
        if key in self.keys:
            seen, reply = self.keys[key]
            if seen != req:                # same key, other request
                return ("422 key reused", None)
            return reply                   # replay, no new charge
        self.balance += amount             # the side effect
        reply = ("201 charged", self.balance)
        if key is not None:
            self.keys[key] = (req, reply)
        return reply


def call(server, amount, key, lose_first=True, tries=4):
    """Client: retry until a reply arrives; reuse the same key."""
    rng = random.Random(3)
    waits = []
    for n in range(tries):
        reply = server.charge(amount, key)
        if n == 0 and lose_first:
            waits.append(full_jitter(n, rng))   # timed out: back off
            continue                            # reply was lost
        return reply, waits
    return None, waits


def dedup_demo():
    for label, key in (("no key  ", None),
                       ("with key", str(uuid.uuid4()))):
        s = Server()
        reply, waits = call(s, 40, key)
        print(f"{label}: reply {reply}, slept "
              f"{[round(w * 1000) for w in waits]} ms, "
              f"balance {s.balance}")
    s = Server()
    k = "k1"
    s.charge(40, k)
    print("same key, amount 50:", s.charge(50, k)[0])


if __name__ == "__main__":
    print("== 1. backoff: base 100 ms, x2, cap 10 s, full jitter")
    schedule()
    print("\n== 2. why jitter")
    herd(n=0)
    herd(n=3)
    print("\n== 3. retry amplification")
    amplification()
    print("\n== 4. idempotency key: first reply lost, client retries")
    dedup_demo()
