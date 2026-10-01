"""Toy memory store: propose -> decide -> commit -> activate.

A pending write only becomes durable after a decision AND a
successful replay, and a running session only sees it after a
fresh session loads it. Not Hermes code; a model of its stages.
"""
import hashlib
import json

SKILL_DESC_MAX = 60  # Hermes routing-description budget (Pt 3)


class Store:
    def __init__(self):
        self.durable = {"MEMORY.md": [], "skills": {}}
        self.pending = []

    def tree_hash(self):
        blob = json.dumps(self.durable, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def propose(self, kind, body, source):
        self.pending.append(
            {"kind": kind, "body": body, "source": source}
        )
        return len(self.pending) - 1

    def decide(self, pid, approve):
        self.pending[pid]["decision"] = approve

    def replay(self, pid):
        p = self.pending[pid]
        if not p.get("decision"):
            return "rejected: not written"
        if p["kind"] == "skill":
            name, desc = p["body"]
            if len(desc) > SKILL_DESC_MAX:
                return f"replay failed: desc {len(desc)} > 60"
            self.durable["skills"][name] = desc
        else:
            self.durable["MEMORY.md"].append(p["body"])
        return "committed"

    def new_session(self):
        # A session snapshots durable state when it starts.
        return json.loads(json.dumps(self.durable))


def demo():
    s = Store()
    tuesday = s.new_session()
    print("start          ", s.tree_hash())

    pid = s.propose("fact", "pytest is flaky in this repo",
                    "agent inference after the Tuesday run")
    s.decide(pid, approve=False)
    r = s.replay(pid)
    print("reject flaky   ", s.tree_hash(), r)

    long_desc = ("Verify before done: run pytest, read the exit "
                 "code, and only claim success on exit 0")
    pid = s.propose("skill", ("verify-before-done", long_desc),
                    "operator, after the Tuesday run")
    s.decide(pid, approve=True)
    r = s.replay(pid)
    print("approve long   ", s.tree_hash(), r)

    short = "Claim done only after pytest exits 0."
    pid = s.propose("skill", ("verify-before-done", short),
                    "operator, after the Tuesday run")
    s.decide(pid, approve=True)
    r = s.replay(pid)
    print("approve short  ", s.tree_hash(), r,
          f"(desc {len(short)} chars)")

    print("Tuesday session sees skill?",
          "verify-before-done" in tuesday["skills"])
    print("fresh session sees skill?  ",
          "verify-before-done" in s.new_session()["skills"])
    print("fresh session sees flaky?  ",
          any("flaky" in m for m in s.new_session()["MEMORY.md"]))


if __name__ == "__main__":
    demo()
