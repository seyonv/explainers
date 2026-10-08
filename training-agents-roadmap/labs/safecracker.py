"""Safecracker (Tumblers v2). A safe with N dials (0-9). Each dial may be secretly linked to others:
a gear link turns the linked dial by the same amount; a pin link turns it only when the driving dial
wraps past 0 (9->0 or 0->9). The wiring is hidden: you see the code, the target and the moves left.
Reach the target within the move budget. python3 safecracker.py prints the baselines."""
import argparse, hashlib, itertools, json, math, os, random, re

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

def shortest_path(start, target, links, n):
    """Shortest move list from start to target. Every move has an inverse (turn i by -d undoes +d,
    pins included), so a search from both ends meets in the middle."""
    if start == target: return []
    par = ({start: None}, {target: None}); front = ([start], [target])
    while front[0] and front[1]:
        side = 0 if len(front[0]) <= len(front[1]) else 1
        mine, other, nxt, best = par[side], par[1 - side], [], None
        for s in front[side]:
            for i, d in actions(n):
                t = turn(s, links, i, d)
                if t in mine: continue
                mine[t] = (s, (i, d)); nxt.append(t)
                if t in other:
                    path = _join(par, t, side)
                    if best is None or len(path) < len(best): best = path
        if best is not None: return best
        front = (nxt, front[1]) if side == 0 else (front[0], nxt)
    return None

def _join(par, meet, side):
    fwd, bwd = par
    head, s = [], meet
    while fwd[s] is not None: s, m = fwd[s]; head.append(m)
    tail, s = [], meet
    while bwd[s] is not None: u, (i, d) = bwd[s]; tail.append((i, -d)); s = u
    return head[::-1] + tail

def bfs(start, target, links, n):
    p = shortest_path(start, target, links, n)
    return None if p is None else len(p)

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

RULES = """You are cracking a safe. It has {n} dials, each showing a digit 0-9. Turning a dial by + or - changes it by one (9 wraps to 0 and 0 wraps to 9).
Some dials are secretly linked. A gear link makes another dial turn the same way whenever the linked-from dial turns. A pin link makes another dial turn the same way only when the linked-from dial wraps past 0 (9 to 0, or 0 to 9). You cannot see the links; you can only watch the dials change.
Reach the target code before you run out of moves. Every reply must end with exactly one move, written turn(dial, +) or turn(dial, -), with dials numbered from 1. You may reason first. A reply without a valid move still uses up a move."""

def move_text(i, d):
    return f"turn({i + 1}, {'+' if d == 1 else '-'})"

def parse_action(text, n):
    m = re.search(r'turn\(\s*(\d+)\s*,\s*([+-])', text) or \
        re.search(r'"dial"\s*:\s*(\d+)\s*,\s*"direction"\s*:\s*"([+-])"', text)
    if not m: return None
    i = int(m.group(1)) - 1
    if not 0 <= i < n: return None
    return (i, 1 if m.group(2) == "+" else -1)

LINKS = {3: 2, 4: 3, 5: 5, 6: 6}   # links per dial count: enough coupling that no dial is free

def _h(c):
    return int(hashlib.sha1(repr(c).encode()).hexdigest(), 16)

def _bucket(c):
    return _h(c) % 4

_SMALL = {}
def _is_test(c, n, tier):
    """About a quarter of wiring shapes are test-only. Small safes have only a handful of shapes, so
    there the lowest-hash shape is test-only if no shape landed in the test quarter."""
    if _bucket(c) == 0: return True
    if n > 4: return False
    if (n, tier) not in _SMALL:
        rng = random.Random(0)
        shapes = {canon(n, _wiring(rng, n, LINKS[n], tier)) for _ in range(3000)}
        _SMALL[(n, tier)] = None if any(_bucket(s) == 0 for s in shapes) else min(shapes, key=_h)
    return c == _SMALL[(n, tier)]

def split(kind, n, tier, count):
    """Held-out safes. Each wiring shape (up to renaming dials) belongs to exactly one side: about a
    quarter of shapes are test-only. Test seeds start at 10,000, train seeds at 0."""
    want_test = kind == "test"; seed = 10_000 if want_test else 0; out = []
    while len(out) < count:
        p = gen(seed, n, LINKS[n], tier); seed += 1
        if _is_test(canon(n, p["links"]), n, tier) == want_test: out.append(p)
    return out

def dist(a, b):
    return sum(min((x - y) % 10, (y - x) % 10) for x, y in zip(a, b))

def play(policy, p, budget):
    code, hist = tuple(p["start"]), []
    for m in range(budget):
        obs = {"code": list(code), "target": tuple(p["target"]), "moves_left": budget - m}
        i, d = policy(obs, hist)
        nxt = turn(code, p["links"], i, d); hist.append((i, d, code, nxt)); code = nxt
        if code == tuple(p["target"]): return True, m + 1
    return False, budget

def random_policy(rng):
    return lambda obs, hist: rng.choice(actions(len(obs["code"])))

def try_undo_policy(seed=0):
    """Try a move; if it didn't bring the dials closer, undo it and try another."""
    rng = random.Random(seed); undone = set()
    def pol(obs, hist):
        n = len(obs["code"])
        if hist:
            i, d, before, after = hist[-1]
            if (i, -d) not in undone and dist(after, obs["target"]) >= dist(before, obs["target"]) \
                    and not (len(hist) > 1 and hist[-2][:2] == (i, -d)):
                undone.add((i, d)); return (i, -d)
            if dist(after, obs["target"]) < dist(before, obs["target"]): undone.clear()
        options = [a for a in actions(n) if a not in undone] or actions(n)
        return rng.choice(options)
    return pol

def prober_policy():
    """A script that knows the trick: turn each dial once to see which dials move with it, plan the
    shortest route on that guess, and re-probe whenever a move does something the guess didn't predict."""
    st = {"model": None, "probe_from": 0, "plan": []}
    def pol(obs, hist):
        n = len(obs["code"])
        if st["model"] is not None and hist:
            i, d, before, after = hist[-1]
            if turn(before, st["model"], i, d) != after:
                st["model"] = None; st["probe_from"] = len(hist); st["plan"] = []
        if st["model"] is None:
            done = len(hist) - st["probe_from"]
            if done < n: return (done, 1)
            st["model"] = {i: [] for i in range(n)}
            for (i, d, before, after) in hist[st["probe_from"]:st["probe_from"] + n]:
                st["model"][i] = [(j, "gear") for j in range(n) if j != i and (after[j] - before[j]) % 10 == 1]
        if not st["plan"]:
            st["plan"] = shortest_path(tuple(obs["code"]), tuple(obs["target"]), st["model"], n) or []
            if not st["plan"]:
                st["model"] = None; st["probe_from"] = len(hist); return (0, 1)
        return st["plan"].pop(0)
    return pol

def traces(count, n=None, tier="linear", mult=2.0):
    """Winning games of the probing script on training safes, as chat transcripts for SFT."""
    sizes = [n] if n else [3, 4]; out = []; k = 0; pools = {m: split("train", m, tier, 4 * count) for m in sizes}
    while len(out) < count and k < 4 * count:
        m = sizes[k % len(sizes)]; p = pools[m][k // len(sizes)]; k += 1
        budget = math.ceil(mult * p["optimal"]); pol = prober_policy(); code = tuple(p["start"]); hist = []
        msgs = [{"role": "system", "content": RULES.format(n=m)},
                {"role": "user", "content": render({"code": code, "target": p["target"], "moves_left": budget})}]
        won = False
        for step in range(budget):
            i, d = pol({"code": list(code), "target": tuple(p["target"]), "moves_left": budget - step}, hist)
            nxt = turn(code, p["links"], i, d); hist.append((i, d, code, nxt)); code = nxt
            msgs.append({"role": "assistant", "content": move_text(i, d)})
            if code == tuple(p["target"]): won = True; break
            msgs.append({"role": "user", "content": render({"code": code, "target": p["target"], "moves_left": budget - step - 1})})
        if won: out.append({"n": m, "tier": tier, "won": True, "messages": msgs})
    return out

FIXED = {0: [(2, "gear")], 1: [(0, "gear")], 2: []}

def _q_puzzle(rng, links, n=3):
    while True:
        s = tuple(rng.randrange(10) for _ in range(n)); t = tuple(rng.randrange(10) for _ in range(n))
        d = bfs(s, t, links, n)
        if d and d >= 3: return {"n": n, "links": links, "start": s, "target": t, "optimal": d}

def _key(code, target):
    return tuple((a - b) % 10 for a, b in zip(code, target))   # the safe looks the same from any target

def q_fixed(episodes=10_000, seed=0, links=FIXED, n=3):
    rng = random.Random(seed); Q = {}; A = actions(n); alpha, gamma = 0.5, 0.95
    for ep in range(episodes):
        eps = max(0.05, 1 - ep / (0.7 * episodes))
        p = _q_puzzle(rng, links, n); code = p["start"]
        for _ in range(4 * p["optimal"]):
            k = _key(code, p["target"]); q = Q.setdefault(k, [0.0] * len(A))
            a = rng.randrange(len(A)) if rng.random() < eps else max(range(len(A)), key=q.__getitem__)
            nxt = turn(code, links, *A[a]); win = nxt == p["target"]
            q2 = Q.setdefault(_key(nxt, p["target"]), [0.0] * len(A))
            q[a] += alpha * ((1.0 if win else -0.01) + (0 if win else gamma * max(q2)) - q[a])
            code = nxt
            if win: break
    return Q

def q_eval(Q, puzzles, budget_mult=2.0):
    A = actions(3); wins = 0
    pol = lambda obs, hist: A[max(range(len(A)), key=Q.get(_key(obs["code"], obs["target"]), [0.0] * len(A)).__getitem__)]
    for p in puzzles: wins += play(pol, p, math.ceil(budget_mult * p["optimal"]))[0]
    return wins / len(puzzles)

def ci95(k, n):
    if n == 0: return [0.0, 0.0]
    z, ph = 1.96, k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]

def _rate(policy_fn, ps, mult):
    k = sum(play(policy_fn(), p, math.ceil(mult * p["optimal"]))[0] for p in ps)
    return k, len(ps)

POLICIES = {"random": lambda: random_policy(random.Random(0)), "try_undo": try_undo_policy, "prober": prober_policy}
MULTS = (1.5, 1.75, 2.0)

def facts(outdir, sizes=(3, 4, 5, 6)):
    rows, cal = [], {}
    for tier in ("linear", "hard"):
        for n in sizes:
            ps = split("test", n, tier, 200 if n < 6 else 100)
            opts = sorted(p["optimal"] for p in ps)
            # Policies ignore the budget, so one run at the largest budget gives every smaller one too.
            moves = {name: [] for name in POLICIES}
            for name, f in POLICIES.items():
                pol_moves = moves[name]
                for p in ps:
                    won, m = play(f(), p, math.ceil(max(MULTS) * p["optimal"]))
                    pol_moves.append((m, p["optimal"]) if won else None)
            for mult in MULTS:
                row = {"tier": tier, "n": n, "split": "test", "count": len(ps), "mult": mult,
                       "optimal_median": opts[len(opts) // 2], "ci95": {}}
                for name in POLICIES:
                    k = sum(1 for x in moves[name] if x and x[0] <= math.ceil(mult * x[1]))
                    row[name] = round(k / len(ps), 3); row["ci95"][name] = ci95(k, len(ps))
                rows.append(row); print(json.dumps(row), flush=True)
        five = [r for r in rows if r["tier"] == tier and r["n"] == 5]
        ok = [r["mult"] for r in five if r["prober"] >= 0.85]
        cal[tier] = min(ok) if ok else 2.0
        if not ok: cal[tier + "_ceiling_note"] = True
    Q = q_fixed()
    rng = random.Random(1)
    same = [_q_puzzle(rng, FIXED) for _ in range(300)]
    other = [p for p in split("test", 3, "linear", 300)]
    q = {"n": 3, "episodes": 10_000, "fixed_wiring": round(q_eval(Q, same), 3), "new_wirings": round(q_eval(Q, other), 3)}
    overlap = 0
    for tier in ("linear", "hard"):
        for n in (3, 4):
            tc = {canon(n, p["links"]) for p in split("test", n, tier, 60)}
            overlap += sum(canon(n, p["links"]) in tc for p in split("train", n, tier, 120))
    out = {"generated": "2026-10-08", "budget_mult": {k: v for k, v in cal.items() if not k.endswith("note")},
           "notes": {k: v for k, v in cal.items() if k.endswith("note")}, "rows": rows, "q": q, "split_overlap": overlap}
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, "safecracker.json"), "w"), indent=1)
    _turn_cases(os.path.join(outdir, "turn-cases.json"))
    _playable(os.path.join(os.path.dirname(os.path.abspath(__file__)), "safes-play.json"), cal)
    return out

def _jsonable(links):
    return {str(i): [[j, k] for j, k in v] for i, v in links.items()}

def _turn_cases(path):
    rng, cases = random.Random(5), []
    while len(cases) < 300:
        n = rng.choice((3, 4, 5)); p = gen(rng.randrange(10**6), n, LINKS[n], rng.choice(("linear", "hard")))
        code = [rng.randrange(10) for _ in range(n)]; i, d = rng.choice(actions(n))
        if len(cases) % 2 == 0:                    # half the cases sit on a 9/0 boundary of the turned dial
            code[i] = 9 if d == 1 else 0
        cases.append({"code": code, "links": _jsonable(p["links"]), "i": i, "d": d,
                      "next": list(turn(tuple(code), p["links"], i, d))})
    json.dump(cases, open(path, "w"))

def _playable(path, cal):
    spec = [("linear", 3), ("linear", 3), ("linear", 4), ("linear", 4), ("linear", 5), ("hard", 3), ("hard", 4), ("hard", 4)]
    safes, used = [], {}
    for tier, n in spec:
        k = used.get((tier, n), 0); used[(tier, n)] = k + 1
        p = split("train", n, tier, k + 1)[k]
        safes.append({"tier": tier, "n": n, "start": list(p["start"]), "target": list(p["target"]),
                      "optimal": p["optimal"], "budget": math.ceil(cal.get(tier, 2.0) * p["optimal"]),
                      "links": _jsonable(p["links"])})
    json.dump(safes, open(path, "w"), indent=1)

def quick_table(sizes, count):
    """Percent of `count` test safes each baseline cracks at a 2x budget, for each tier and dial count."""
    rows = []
    for tier in ("linear", "hard"):
        for n in sizes:
            ps = split("test", n, tier, count)
            r = {k: round(100 * _rate(f, ps, 2.0)[0] / count) for k, f in POLICIES.items()}
            rows.append({"tier": tier, "n": n, **r})
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facts", metavar="OUTDIR", help="measure everything and write safecracker.json")
    ap.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--q", action="store_true", help="tabular Q-learning on one fixed safe, then on new ones")
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--traces", type=int, metavar="N", help="print N winning probing-script games as JSON lines (for SFT)")
    a = ap.parse_args()
    if a.traces:
        for g in traces(a.traces): print(json.dumps(g))
        return
    if a.facts: facts(a.facts, tuple(a.sizes)); return
    if a.q:
        Q = q_fixed(a.episodes); rng = random.Random(1)
        same = [_q_puzzle(rng, FIXED) for _ in range(300)]
        print(f"Q-table after {a.episodes} episodes: {q_eval(Q, same):.0%} of fresh targets on the safe it trained on, "
              f"{q_eval(Q, split('test', 3, 'linear', 300)):.0%} on safes with new wiring")
        return
    sizes = [s for s in a.sizes if s in LINKS] if a.sizes != [3, 4, 5, 6] else [3, 4, 5]
    print("50 test safes per row, move budget 2x the full-information optimum (bigger safes take a few minutes)")
    for r in quick_table(sizes, 50):
        print(f"{r['tier']:6} dials={r['n']}: random {r['random']}%, try-and-undo {r['try_undo']}%, prober {r['prober']}%")

if __name__ == "__main__":
    main()
