"""Invariant confluence (I-confluence), checked on two replicas.

A database state is a frozenset of versions. Merge is set union,
which is commutative, associative and idempotent. Each replica
starts from the same state Ds, runs one transaction that is valid
on its own replica, and then the two replicas merge.
If the merge breaks the invariant, that (invariant, operation)
pair is not I-confluent: it needs coordination.
"""


def merge(a, b):
    return a | b                      # set union of versions


def check(inv, ds, t1, t2):
    di, dj = ds | {t1}, ds | {t2}     # each replica commits locally
    assert inv(ds) and inv(di) and inv(dj), "must be valid locally"
    return inv(merge(di, dj)), merge(di, dj)


# --- invariants over a set of tagged rows -------------------------
def unique_ids(d):                    # ("user", id, name)
    ids = [r[1] for r in d if r[0] == "user"]
    return len(ids) == len(set(ids))


def balance(d):                       # ("op", op_id, +/- amount)
    return sum(r[2] for r in d if r[0] == "op")


def non_negative(d):
    return balance(d) >= 0


def no_overlap(d):                    # ("book", who, start, end)
    bs = sorted(r[2:] for r in d if r[0] == "book")
    return all(e1 <= s2 for (_, e1), (s2, _) in zip(bs, bs[1:]))


def live_users(d):
    gone = {r[1] for r in d if r[0] == "del"}
    return {r[1] for r in d if r[0] == "user"} - gone


def foreign_key(d):                   # ("order", id, user)
    live = live_users(d)
    return all(r[2] in live for r in d if r[0] == "order")


def foreign_key_cascade(d):
    live = live_users(d)              # orders of deleted users vanish
    kept = {r for r in d if r[0] != "order" or r[2] in live}
    return foreign_key(kept)


# (name, invariant, Ds, replica 1 txn, replica 2 txn); times in min
CASES = [
    ("room: Alice 10-11, Bob 11-12 (source)", no_overlap, set(),
     ("book", "Alice", 600, 660), ("book", "Bob", 660, 720)),
    ("room: Alice 10-11, Carol 10:30-11:30", no_overlap, set(),
     ("book", "Alice", 600, 660), ("book", "Carol", 630, 690)),
    ("unique id: insert Stan:5, Mary:5", unique_ids, set(),
     ("user", 5, "Stan"), ("user", 5, "Mary")),
    ("unique id: ids r1-1, r2-1 (replica prefix)", unique_ids, set(),
     ("user", "r1-1", "Stan"), ("user", "r2-1", "Mary")),
    ("balance>=0: 100, withdraw 60 / 70", non_negative,
     {("op", "d0", 100)}, ("op", "w1", -60), ("op", "w2", -70)),
    ("balance>=0: 100, deposit 50 / 20", non_negative,
     {("op", "d0", 100)}, ("op", "d1", 50), ("op", "d2", 20)),
    ("FK insert: orders o1, o2 -> alice", foreign_key,
     {("user", "alice", "")}, ("order", "o1", "alice"),
     ("order", "o2", "alice")),
    ("FK delete: del bob / order o3 -> bob", foreign_key,
     {("user", "bob", "")}, ("del", "bob"), ("order", "o3", "bob")),
    ("FK cascading delete: same txns", foreign_key_cascade,
     {("user", "bob", "")}, ("del", "bob"), ("order", "o3", "bob")),
]

if __name__ == "__main__":
    # the source's merge example: Rx = {v}, Ry = {w}
    print("merge({v}, {w}) =", sorted(merge({"v"}, {"w"})))
    for name, inv, ds, t1, t2 in CASES:
        ok, merged = check(inv, frozenset(ds), t1, t2)
        extra = ""
        if inv is non_negative:
            extra = f"  balance {balance(merged)}"
        print(f"{'ok  ' if ok else 'FAIL'} {name}{extra}")
    # merge laws, on the three states of the source's room example
    a = frozenset({("book", "Alice", 600, 660)})
    b = frozenset({("book", "Bob", 660, 720)})
    c = frozenset()
    print("commutative", merge(a, b) == merge(b, a),
          "associative", merge(merge(a, b), c) == merge(a, merge(b, c)),
          "idempotent", merge(a, a) == a)
