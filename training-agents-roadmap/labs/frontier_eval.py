"""Play Safecracker against any model behind an OpenAI-compatible chat endpoint: one move per message,
the wiring never shown. Prints the crack rate with a 95% interval and dollars per safe cracked.

  export OPENAI_API_KEY=...            # or any compatible provider, with --base-url
  python3 frontier_eval.py --models <model> --episodes 2 --max-usd 3            # dry run: read the transcripts
  python3 frontier_eval.py --models <strong> <cheaper> --episodes 50 --max-usd 30

Fill in PRICES from your provider's pricing page first: (input, output) dollars per 1M tokens."""
import argparse, json, math, os, sys, time, urllib.request
from safecracker import split, turn, render, parse_action, ci95, RULES

PRICES = {
    # "model-id": (input $ per 1M tokens, output $ per 1M tokens),
}

def make_chat(base_url, key):
    def chat(model, messages):
        body = json.dumps({"model": model, "messages": messages, "max_completion_tokens": 4000}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as r: d = json.load(r)
                return d["choices"][0]["message"].get("content") or "", d.get("usage", {})
            except Exception:
                if attempt == 3: raise
                time.sleep(5 * (attempt + 1))
    return chat

def cost(prices, model, usage):
    pi, po = prices.get(model, (0.0, 0.0))
    return (usage.get("prompt_tokens", 0) * pi + usage.get("completion_tokens", 0) * po) / 1e6

def episode(chat, model, p, budget, prices):
    code, usd, log = tuple(p["start"]), 0.0, []
    msgs = [{"role": "system", "content": RULES.format(n=p["n"])},
            {"role": "user", "content": render({"code": code, "target": p["target"], "moves_left": budget})}]
    for m in range(budget):
        out, usage = chat(model, msgs); usd += cost(prices, model, usage)
        lines = out.strip().splitlines()
        a = (parse_action(lines[-1], p["n"]) if lines else None) or parse_action(out[-200:], p["n"])
        log.append({"move": m, "reply_tail": out[-300:], "action": a})
        if a: code = turn(code, p["links"], *a)
        if code == tuple(p["target"]): return True, m + 1, usd, log
        note = "" if a else "That reply had no valid move, so it used a move and changed nothing.\n"
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": note + render({"code": code, "target": p["target"], "moves_left": budget - m - 1})}]
    return False, budget, usd, log

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-usd", type=float, default=30)
    ap.add_argument("--tier", default="linear", choices=["linear", "hard"])
    ap.add_argument("--n", type=int, default=5, help="dials")
    ap.add_argument("--mult", type=float, default=1.75, help="move budget as a multiple of the shortest route")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--out", default="frontier_results.json")
    a = ap.parse_args()
    missing = [m for m in a.models if m not in PRICES]
    if missing: print(f"Add prices for {missing} to PRICES at the top of this file first.", file=sys.stderr); sys.exit(2)
    chat = make_chat(a.base_url, os.environ.get("OPENAI_API_KEY", ""))
    safes = split("test", a.n, a.tier, a.episodes); spent = 0.0; results = []
    for model in a.models:
        won = done = 0; usd = 0.0; games = []
        for p in safes:
            if spent >= a.max_usd: print("spend cap reached", file=sys.stderr); break
            w, mv, c, log = episode(chat, model, p, math.ceil(a.mult * p["optimal"]), PRICES)
            won += w; usd += c; spent += c; done += 1
            games.append({"start": p["start"], "target": p["target"], "optimal": p["optimal"], "won": w, "usd": c, "log": log})
            print(f"{model} {done}/{len(safes)} {'cracked' if w else 'failed'} in {mv} moves, ${c:.3f}", flush=True)
        lo, hi = ci95(won, done)
        results.append({"model": model, "tier": a.tier, "n": a.n, "mult": a.mult, "episodes": done, "won": won,
                        "rate": won / max(done, 1), "ci95": [lo, hi], "usd": usd, "usd_per_crack": usd / won if won else None, "games": games})
        print(f"{model}: {won}/{done} cracked ({won / max(done, 1):.0%}, 95% CI {lo:.0%}-{hi:.0%}), "
              f"${usd / won:.3f} per safe cracked" if won else f"{model}: 0/{done} cracked")
    json.dump(results, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
