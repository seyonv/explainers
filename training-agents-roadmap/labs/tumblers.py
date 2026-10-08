"""Tumblers: a lock with N dials (0-9). Turning a dial also drags the dials it is linked to by the
same amount. Reach the target code within the move budget. Original game for the roadmap capstone.
Superseded by safecracker.py (hidden wiring, pin links); this file stays for older links."""
import random
from collections import deque

def gen(seed, n=3, n_links=2, min_opt=3):
    rng = random.Random(seed)
    while True:
        links = {i: [] for i in range(n)}
        for _ in range(n_links):
            i, j = rng.sample(range(n), 2)
            if j not in links[i]: links[i].append(j)
        start = tuple(rng.randrange(10) for _ in range(n))
        target = tuple(rng.randrange(10) for _ in range(n))
        d = bfs(start, target, links, n)
        if d is not None and d >= min_opt:
            return {"n": n, "links": links, "start": start, "target": target, "optimal": d}

def turn(code, links, i, delta):
    c = list(code)
    for k in [i] + links[i]:
        c[k] = (c[k] + delta) % 10
    return tuple(c)

def actions(n):
    return [(i, d) for i in range(n) for d in (1, -1)]

def bfs(start, target, links, n):
    seen = {start: 0}; q = deque([start])
    while q:
        s = q.popleft()
        if s == target: return seen[s]
        for i, d in actions(n):
            t = turn(s, links, i, d)
            if t not in seen: seen[t] = seen[s] + 1; q.append(t)
    return None

def dist(a, b):
    return sum(min((x - y) % 10, (y - x) % 10) for x, y in zip(a, b))

if __name__ == "__main__":
    import statistics as st
    for n, nl in ((3, 2), (4, 3)):
        ps = [gen(s, n, nl) for s in range(300)]
        opt = [p["optimal"] for p in ps]
        rng = random.Random(0); wins = 0; greedy_wins = 0
        for p in ps:
            budget = p["optimal"] * 2
            s = p["start"]
            for _ in range(budget):
                s = turn(s, p["links"], *rng.choice(actions(n)))
                if s == p["target"]: wins += 1; break
            s = p["start"]  # greedy: pick the move that most reduces digit distance
            for _ in range(budget):
                s = min((turn(s, p["links"], i, d) for i, d in actions(n)), key=lambda t: dist(t, p["target"]))
                if s == p["target"]: greedy_wins += 1; break
        print(f"dials={n} links={nl}: optimal moves median {st.median(opt)} (range {min(opt)}-{max(opt)}); "
              f"random wins {wins/len(ps):.3f}; greedy-by-distance wins {greedy_wins/len(ps):.3f}")
    print(gen(1, 3, 2))
