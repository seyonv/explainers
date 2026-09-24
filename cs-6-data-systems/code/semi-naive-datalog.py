"""Transitive closure in Datalog: naive vs semi-naive evaluation.

    tc(x, y) :- edge(x, y).
    tc(x, y) :- tc(x, z), edge(z, y).          # linear rule

A "derivation" is one match of a rule body, duplicates included:
that is the work an evaluator does. Graph: the chain 1 -> 2 -> ... -> 6.
Run with python3 (3.10+).
"""
import sqlite3


def chain(n):
    return {(i, i + 1) for i in range(1, n)}


def join(left, right):                # left(x, z), right(z, y)
    by_z = {}
    for z, y in right:
        by_z.setdefault(z, []).append(y)
    return [(x, y) for x, z in left for y in by_z.get(z, ())]


def naive(edge):
    tc, rounds = set(), []
    while True:
        derived = list(edge) + join(tc, edge)   # redo everything
        new = set(derived) - tc
        rounds.append((len(derived), len(new)))
        if not new:
            return tc, rounds
        tc |= new


def semi_naive(edge):
    tc, rounds = set(), []
    delta = set(edge)                  # round 1: the base rule
    rounds.append((len(edge), len(delta)))
    while delta:
        tc |= delta
        derived = join(delta, edge)    # only last round's new facts
        delta = set(derived) - tc
        rounds.append((len(derived), len(delta)))
    return tc, rounds


# Non-linear rule tc(x,y) :- tc(x,z), tc(z,y): two IDB atoms, so two
# delta rules. Atoms before the delta use the new tc, atoms after it
# use the old one (the source's order), so no pair is joined twice.
def naive_nonlinear(edge):
    tc, rounds = set(), []
    while True:
        derived = list(edge) + join(tc, tc)
        new = set(derived) - tc
        rounds.append((len(derived), len(new)))
        if not new:
            return tc, rounds
        tc |= new


def semi_naive_nonlinear(edge):
    old, delta = set(), set(edge)
    rounds = [(len(edge), len(delta))]
    while delta:
        new_tc = old | delta
        derived = join(delta, old) + join(new_tc, delta)
        old = new_tc
        delta = set(derived) - old
        rounds.append((len(derived), len(delta)))
    return old, rounds


def sqlite_closure(edge):             # WITH RECURSIVE, same answer
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE edge(a, b)")
    db.executemany("INSERT INTO edge VALUES (?, ?)", sorted(edge))
    q = """WITH RECURSIVE tc(a, b) AS (
             SELECT a, b FROM edge
             UNION
             SELECT tc.a, edge.b FROM tc JOIN edge ON tc.b = edge.a)
           SELECT a, b FROM tc"""
    return set(db.execute(q).fetchall())


def show(name, rounds):
    print(f"{name}:")
    for i, (d, n) in enumerate(rounds, 1):
        print(f"  round {i}: {d:3} derivations, {n:2} new")
    total = sum(d for d, _ in rounds)
    print(f"  total {total} derivations in {len(rounds)} rounds")
    return total


if __name__ == "__main__":
    E = chain(6)
    tc1, r1 = naive(E)
    tc2, r2 = semi_naive(E)
    assert tc1 == tc2 == sqlite_closure(E)
    print(f"closure of the 6-node chain: {len(tc1)} facts")
    a = show("naive, linear rule", r1)
    b = show("semi-naive, linear rule", r2)
    print(f"  wasted by naive: {a} - {b} = {a - b}")

    tc3, r3 = naive_nonlinear(E)
    tc4, r4 = semi_naive_nonlinear(E)
    assert tc3 == tc4 == tc1
    show("naive, non-linear rule", r3)
    show("semi-naive, non-linear rule", r4)

    print("growth on longer chains (linear rule):")
    for n in (6, 12, 24, 48):
        e = chain(n)
        nv = sum(d for d, _ in naive(e)[1])
        sn = sum(d for d, _ in semi_naive(e)[1])
        print(f"  n={n:3}: naive {nv:6}, semi-naive {sn:5},"
              f" ratio {nv / sn:.1f}x")
