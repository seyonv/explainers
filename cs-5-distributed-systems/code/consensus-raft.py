"""Consensus with Raft: a 5-node election and log repair, simulated.

Five queue nodes (the image-resize service's queue layer) replicate
a log. S1 leads term 1 and crashes. The followers' randomized
election timeouts (150-300 ms, the Raft paper's example range) decide
who runs first; the vote rule decides who can win. Then the new
leader repairs the lagging logs with AppendEntries.

Time is simulated in ms with a fixed one-way message delay, so every
run is exact and repeatable. Standard library only; Python 3.10.
"""
import heapq
import random

N, MAJORITY = 5, 5 // 2 + 1          # 5 nodes, 3 votes to win
DELAY = 5                            # one-way message delay, ms
LO, HI = 150, 300                    # election timeout range, ms


def up_to_date(c_last_term, c_last_idx, my_log):
    my_term = my_log[-1] if my_log else 0
    if c_last_term != my_term:          # later last term wins
        return c_last_term > my_term
    return c_last_idx >= len(my_log)     # same term: longer wins

def on_request_vote(node, cand, term, last_term, last_idx):
    if term > node["term"]:              # newer term: follow it
        node.update(term=term, voted=None, state="follower")
    grant = (term == node["term"]
             and node["voted"] in (None, cand)   # one vote a term
             and up_to_date(last_term, last_idx, node["log"]))
    if grant:
        node["voted"] = cand
    return grant


def elect(logs, rng, fixed=None, trace=None, crashed=("S1",)):
    """Run elections after the leader crashed at t = 0.
    logs: name -> list of entry terms. Returns (leader, term, t)."""
    nodes = {s: dict(term=1, voted=None, state="follower", log=l,
                     deadline=0, votes=set())
             for s, l in logs.items()}
    q, seq = [], 0

    def push(t, *ev):
        nonlocal seq
        seq += 1
        heapq.heappush(q, (t, seq) + ev)

    def arm(s, now):
        to = fixed if fixed is not None else rng.uniform(LO, HI)
        nodes[s]["deadline"] = now + to
        push(now + to, "timeout", s, now + to)

    def log(t, msg):
        if trace is not None:
            trace.append((t, msg))

    for s in nodes:                      # last heartbeat at t = 0
        if s not in crashed:
            arm(s, 0)
    while q:
        t, _, kind, *a = heapq.heappop(q)
        if t > 5000:
            return None, None, None      # gave up: no leader
        if kind == "timeout":
            s, when = a
            n = nodes[s]
            if n["deadline"] != when or n["state"] == "leader":
                continue                 # timer was reset
            n.update(term=n["term"] + 1, voted=s, state="candidate",
                     votes={s})
            log(t, f"{s} times out -> candidate, term {n['term']}")
            arm(s, t)
            for p in nodes:
                if p != s and p not in crashed:
                    push(t + DELAY, "rv", p, s, n["term"],
                         n["log"][-1], len(n["log"]))
        elif kind == "rv":
            p, cand, term, lt, li = a
            ok = on_request_vote(nodes[p], cand, term, lt, li)
            if ok:
                arm(p, t)                # granting resets my timer
            log(t, f"  {p} {'grants' if ok else 'refuses'} {cand}"
                   f" (term {term})")
            push(t + DELAY, "reply", cand, p, nodes[p]["term"], ok)
        elif kind == "reply":
            cand, voter, term, ok = a
            n = nodes[cand]
            if term > n["term"]:
                n.update(term=term, voted=None, state="follower")
            elif ok and n["state"] == "candidate" \
                    and term == n["term"]:
                n["votes"].add(voter)
                if len(n["votes"]) >= MAJORITY:
                    n["state"] = "leader"
                    log(t, f"{cand} has {len(n['votes'])} of {N}"
                           f" votes -> LEADER, term {term}")
                    return cand, term, t
    return None, None, None


def repair(leader_log, follower_log):
    """AppendEntries consistency check: walk nextIndex back until
    the entry before it matches, then overwrite from there."""
    nxt = len(leader_log)                # 1-based index to send next
    tries = []
    while True:
        prev = nxt - 1                   # prevLogIndex
        ok = prev == 0 or (prev <= len(follower_log)
                           and follower_log[prev - 1]
                           == leader_log[prev - 1])
        tries.append((prev, ok))
        if ok:
            return tries, follower_log[:prev] + leader_log[prev:]
        nxt -= 1


def commit_index(match, leader_log, term):
    for i in range(len(leader_log), 0, -1):
        if sum(m >= i for m in match) >= MAJORITY \
                and leader_log[i - 1] == term:  # own term only
            return i
    return 0


LOGS = {"S1": [1, 1, 1, 1], "S2": [1, 1, 1, 1], "S3": [1, 1, 1, 1],
        "S4": [1, 1, 1], "S5": [1, 1]}
SEED = 0


def scenario():
    tr = []
    logs = {k: list(v) for k, v in LOGS.items()}
    leader, term, t = elect(logs, random.Random(SEED), trace=tr)
    for when, msg in tr:
        print(f"t={when:6.1f} ms  {msg}")
    new = logs[leader] + [term]          # the new leader's no-op
    print(f"\n{leader} appends a no-op at index {len(new)},"
          f" term {term}")
    match = {leader: len(new), "S1": 0}
    for s in ("S2", "S3", "S4", "S5"):
        if s == leader:
            continue
        tries, fixed = repair(new, logs[s])
        steps = ", ".join(f"prev={p} {'ok' if ok else 'no'}"
                          for p, ok in tries)
        print(f"  {s} {logs[s]}: {steps} -> {fixed}")
        match[s] = len(fixed)
    ci = commit_index(list(match.values()), new, term)
    print(f"commitIndex = {ci}: entries 1..{ci} are committed")
    return leader, term, t


def monte_carlo(runs=10_000):
    """Time from the crash to a new leader, over many seeds."""
    even = {k: [1, 1, 1, 1] for k in LOGS}      # nobody lagging
    cases = (("lagging, 150-300", LOGS, None, runs),
             ("all equal, 150-300", even, None, runs),
             ("all equal, fixed 150", even, 150, 1))
    for label, logs, fixed, n in cases:
        times, first = [], 0
        for seed in range(n):
            tr = []
            leader, _, t = elect({k: list(v) for k, v in logs.items()},
                                 random.Random(seed), fixed, tr)
            if leader is None:
                continue
            times.append(t)
            first += sum("candidate" in m for _, m in tr) == 1
        if not times:
            print(f"{label:21} no leader after 5 s: every round"
                  f" is a split vote")
            continue
        times.sort()
        mean = sum(times) / len(times)
        p99 = times[int(0.99 * len(times)) - 1]
        print(f"{label:21} 1st candidate wins {first / n:.1%},"
              f" mean {mean:.0f} ms, p99 {p99:.0f} ms")


def commit_latency():
    # Azure median RTTs from East US (shared course facts), ms
    rtt = {"East US 2": 8, "West US": 69, "West Europe": 83,
           "Japan East": 162}
    acks = sorted(rtt.values())
    need = MAJORITY - 1                  # leader counts itself
    print(f"leader in East US, followers RTT {acks}:"
          f" commit waits for ack #{need} = {acks[need - 1]} ms")


if __name__ == "__main__":
    print(f"majority of {N} = {MAJORITY};"
          f" two majorities share >= {2 * MAJORITY - N} node\n")
    scenario()
    print()
    monte_carlo()
    print()
    commit_latency()
