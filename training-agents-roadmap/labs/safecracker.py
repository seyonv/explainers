"""Safecracker (Tumblers v2). A safe with N dials (0-9). Each dial may be secretly linked to others:
a gear link turns the linked dial by the same amount; a pin link turns it only when the driving dial
wraps past 0 (9->0 or 0->9). The wiring is hidden: you see the code, the target and the moves left.
Reach the target within the move budget. python3 safecracker.py prints the baselines."""
import itertools, json, random, re
from collections import deque

def actions(n):
    return [(i, d) for i in range(n) for d in (1, -1)]

def turn(code, links, i, d):
    c = list(code)
    wrapped = (code[i] == 9 and d == 1) or (code[i] == 0 and d == -1)
    c[i] = (c[i] + d) % 10
    for j, kind in links[i]:
        if kind == "gear" or wrapped:
            c[j] = (c[j] + d) % 10
    return tuple(c)

def bfs(start, target, links, n):
    seen = {start: 0}; q = deque([start])
    while q:
        s = q.popleft()
        if s == target: return seen[s]
        for i, d in actions(n):
            t = turn(s, links, i, d)
            if t not in seen: seen[t] = seen[s] + 1; q.append(t)
    return None

def _wiring(rng, n, n_links, tier):
    links = {i: [] for i in range(n)}
    pairs = rng.sample([(i, j) for i in range(n) for j in range(n) if i != j], n_links)
    for k, (i, j) in enumerate(pairs):
        kind = "pin" if tier == "hard" and k % 2 == 0 else "gear"   # hard: at least one pin
        links[i].append((j, kind))
    return links

def gen(seed, n, n_links, tier="linear", min_opt=3):
    rng = random.Random(f"{seed}:{n}:{n_links}:{tier}")
    while True:
        links = _wiring(rng, n, n_links, tier)
        start = tuple(rng.randrange(10) for _ in range(n))
        target = tuple(rng.randrange(10) for _ in range(n))
        d = bfs(start, target, links, n)
        if d is not None and d >= min_opt:
            return {"n": n, "links": links, "start": start, "target": target, "optimal": d, "tier": tier}

def canon(n, links):
    best = None
    for p in itertools.permutations(range(n)):
        edges = tuple(sorted((p[i], p[j], k) for i in links for j, k in links[i]))
        best = edges if best is None or edges < best else best
    return (n, best)

def render(obs):
    code = " ".join(str(x) for x in obs["code"]); tgt = " ".join(str(x) for x in obs["target"])
    return (f"dials: {code}\ntarget: {tgt}\nmoves left: {obs['moves_left']}\n"
            f"tool: turn(dial, + or -), dials numbered 1-{len(obs['code'])}")

def parse_action(text, n):
    m = re.search(r'turn\(\s*(\d+)\s*,\s*([+-])', text) or \
        re.search(r'"dial"\s*:\s*(\d+)\s*,\s*"direction"\s*:\s*"([+-])"', text)
    if not m: return None
    i = int(m.group(1)) - 1
    if not 0 <= i < n: return None
    return (i, 1 if m.group(2) == "+" else -1)
