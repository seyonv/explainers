"""What eventual consistency can't promise: four failures, simulated.

Two (or three) replicas that accept writes locally and merge later,
Dynamo-style. Each demo breaks one guarantee from the cheat-sheet.

Run: python3 eventual-consistency-limits.py   (Python 3.10, stdlib)
"""


class Replica:
    """Stores versions: name -> set of owners. Merge is union."""
    def __init__(self, name):
        self.name, self.users = name, {}

    def register(self, username, owner):
        if self.users.get(username):  # sees local data only
            return "taken"
        self.users[username] = {owner}
        return "ok"

    def merge(self, other):  # commutative, associative, idempotent
        for u, owners in other.users.items():
            self.users.setdefault(u, set()).update(owners)


def unique_ok(r):
    return all(len(o) <= 1 for o in r.users.values())


def demo_uniqueness():
    a, b = Replica("A"), Replica("B")
    print("1. key uniqueness")
    print(f"  t1 Ann -> A register sam: {a.register('sam', 'ann')}")
    print(f"  t2 Bo  -> B register sam: {b.register('sam', 'bo')}")
    print(f"     each replica valid? A {unique_ok(a)},"
          f" B {unique_ok(b)}")
    a.merge(b)
    b.merge(a)
    owners = sorted(a.users["sam"])
    same = a.users == b.users
    print(f"  t3 merge: sam -> {owners}, replicas equal: {same}")
    print(f"     invariant 'one owner per name' holds? {unique_ok(a)}")


def demo_check_and_set():
    # Both writers read lock=free, then write their own name.
    # Each node applies the two writes in the order they arrive.
    print("2. check-and-set by read-then-write")
    arrivals = {"node 1": ["ann", "bo"], "node 2": ["bo", "ann"]}
    final = {}
    for node, order in arrivals.items():
        lock = "free"
        for writer in order:
            lock = writer             # write wins until the next one
        final[node] = lock
        print(f"  {node} applies {' then '.join(order)} -> {lock}")
    print(f"  nodes agree? {len(set(final.values())) == 1}")


def demo_all_or_nothing(n=3, w=2):
    print("3. all-or-nothing updates")
    nodes = [{"plan": "free"} for _ in range(n)]
    reachable = [True, False, False]  # partition: 1 of 3 answers
    acks = 0
    for node, up in zip(nodes, reachable):
        if up:
            node["plan"] = "pro"
            acks += 1
    status = "ok" if acks >= w else "partial"
    print(f"  write plan=pro: {acks} of {n} acked, needed {w}"
          f" -> {status}")
    print(f"  stored now: {[d['plan'] for d in nodes]}")


def demo_read_your_write():
    print("4. read-your-write")
    zones = {"us": {"bio": "old"}, "eu": {"bio": "old"}}
    zones["us"]["bio"] = "new"        # accepted in the local zone
    print(f"  write bio=new in us; read in us -> {zones['us']['bio']}")
    print(f"  same user, next request routed to eu -> "
          f"{zones['eu']['bio']}")


if __name__ == "__main__":
    demo_uniqueness()
    demo_check_and_set()
    demo_all_or_nothing()
    demo_read_your_write()
