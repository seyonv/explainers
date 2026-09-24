"""Speculative decoding on your Mac: is the draft cheap enough?  (course 5, spec-decoding-practice card)

Measures, with llama.cpp and the Llama-3.2 GGUF files Ollama already downloaded:
  (a) c, the draft's cost per step as a fraction of the target's   (llama-bench, token generation)
  (b) acceptance = accepted / drafted for draft length n = 2, 4, 8  (llama-speculative, temp 0)
      on three prompts: code, chat, repeat-a-list
  (c) tokens/s of the target alone vs target + draft (n = 4)        (two llama-servers, runs alternated)
Then it plugs (a) and (b) into Leviathan et al.'s Theorem 3.8,
      speedup = (1 - a^(g+1)) / ((1 - a) (g c + 1)),
and prints the prediction next to the measurement.

Setup (once):
  brew install llama.cpp                 # gives llama-server, llama-speculative, llama-bench
  ollama pull llama3.2:3b && ollama pull llama3.2:1b
The lab finds the GGUF files with `ollama show --modelfile llama3.2:3b` (the FROM line). Or pass them:
  python3 labs/spec-mac.py --target /path/to/target.gguf --draft /path/to/draft.gguf

Run:
  python3 labs/spec-mac.py              # full run: roughly 10-20 minutes on an M3; close other apps first
  python3 labs/spec-mac.py --quick      # smoke test: 8 tokens, one prompt, one run (~2 minutes)
  python3 labs/spec-mac.py --recorded   # no measuring: reprint the author's numbers (M3, 2026-09-22/23)

Timings are load-sensitive: the lab prints your load average and warns if the machine is busy.
Acceptance rates are not (they count tokens, not seconds).
"""
import argparse, json, os, re, shutil, statistics, subprocess, sys, time, urllib.request

PROMPTS = {
    "code": "Write a Python function that returns the n-th Fibonacci number using iteration. Include a docstring and three doctest examples.",
    "chat": "Explain to a new employee, in plain English, why it is important to back up their work, and give five practical tips.",
    "repeat": "Repeat the following list exactly, one item per line: apple, banana, cherry, date, elderberry, fig, grape, honeydew, kiwi, lemon, mango, nectarine, orange, papaya, quince, raspberry, strawberry, tangerine.",
}

# The author's run: Apple M3 8-core, 24 GB, llama.cpp 0.4.1 Metal, target Llama-3.2-3B-Instruct Q4_K_M,
# draft Llama-3.2-1B-Instruct Q8_0. Load average was ~33 on 8 cores during (c), so treat (c) as rough.
RECORDED = {
    "tg": {"target": 28.8, "draft": 43.3},                      # llama-bench -p 512 -n 128 -r 3, tok/s
    "accept": {                                                 # (accepted, drafted), 200 tokens, temp 0
        "code":   {2: (130, 142), 4: (156, 180), 8: (172, 232)},
        "chat":   {2: (118, 168), 4: (136, 268), 8: (152, 400)},
        "repeat": {2: (115, 174), 4: (136, 260), 8: (146, 448)},
    },
    "server": {                                                 # medians of 5 alternated runs, n = 4, tok/s
        "code":   (15.48, 16.65),
        "chat":   (16.40, 36.40),
        "repeat": (16.34, 15.51),
    },
}


def thm38(a, g, c):
    """Leviathan Thm 3.8: expected walltime improvement for acceptance a, draft length g, cost ratio c."""
    return expected_tokens(a, g) / (g * c + 1)


def expected_tokens(a, g):
    """Leviathan Eq. 1: expected tokens per target pass (i.i.d. acceptances)."""
    return g + 1 if a >= 1 else (1 - a ** (g + 1)) / (1 - a)


def need_tools():
    missing = [b for b in ("llama-server", "llama-speculative", "llama-bench") if not shutil.which(b)]
    if missing:
        print("Missing:", ", ".join(missing))
        print("Install llama.cpp:  brew install llama.cpp")
        print("Or see the author's numbers without measuring:  python3 labs/spec-mac.py --recorded")
        sys.exit(1)


def ollama_gguf(tag):
    """Return the GGUF blob path Ollama uses for `tag` (the FROM line of its modelfile)."""
    if not shutil.which("ollama"):
        return None
    out = subprocess.run(["ollama", "show", "--modelfile", tag], capture_output=True, text=True).stdout
    m = re.search(r"^FROM\s+(\S+\.gguf|\S*/blobs/\S+)", out, re.M)
    return m.group(1) if m and os.path.exists(m.group(1)) else None


def load_warning():
    l1 = os.getloadavg()[0]
    cores = os.cpu_count() or 8
    print(f"load average (1 min) = {l1:.1f} on {cores} cores", end="")
    if l1 > cores / 2:
        print("  <- BUSY: timings in (a) and (c) will be low and noisy. Close other apps and re-run.")
    else:
        print("  (fine)")


def bench_tg(model, n, reps):
    """Token-generation speed (tok/s) from llama-bench."""
    out = subprocess.run(["llama-bench", "-m", model, "-p", "0", "-n", str(n), "-r", str(reps), "-o", "json"],
                         capture_output=True, text=True).stdout
    rows = json.loads(out)
    return rows[-1]["avg_ts"]


def spec_accept(target, draft, prompt, n_draft, n_predict):
    err = subprocess.run(["llama-speculative", "-m", target, "-md", draft, "-p", prompt, "-n", str(n_predict),
                          "--temp", "0", "--spec-draft-n-max", str(n_draft), "-ngl", "99", "-ngld", "99", "--seed", "1"],
                         capture_output=True, text=True).stderr
    dr = int(re.search(r"n_drafted\s*=\s*(\d+)", err).group(1))
    ac = int(re.search(r"n_accept\s*=\s*(\d+)", err).group(1))
    return ac, dr


def start_server(target, port, extra):
    p = subprocess.Popen(["llama-server", "-m", target, "--port", str(port), "-ngl", "99", "-c", "4096", "-np", "1"] + extra,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return p
        except Exception:
            time.sleep(0.5)
    p.kill()
    sys.exit(f"llama-server on port {port} did not start")


def complete(port, prompt, n_predict):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["timings"]["predicted_per_second"]


def report(tg, accept, server, n_list):
    c = tg["target"] / tg["draft"]
    print("\n== (a) cost ratio c ==")
    print(f"target {tg['target']:.1f} tok/s, draft {tg['draft']:.1f} tok/s")
    print(f"c = draft step / target step = {tg['target']:.1f} / {tg['draft']:.1f} = {c:.3f}")
    print("   (Leviathan App A.3: T5-small/base/large drafts c = 0.02 / 0.04 / 0.11; Chen: 1.8 / 14.1 = 0.128)")

    print("\n== (b) acceptance and the Thm 3.8 prediction (using acceptance as alpha, and this c) ==")
    print(f"{'prompt':>7} {'n':>3} {'accepted/drafted':>17} {'alpha':>6} {'E tokens/pass':>14} {'cost/pass':>10} {'predicted':>10}")
    for k, row in accept.items():
        for n in n_list:
            if n not in row:
                continue
            ac, dr = row[n]
            a = ac / dr if dr else 0.0
            print(f"{k:>7} {n:>3} {ac:>8}/{dr:<8} {a:>6.1%} {expected_tokens(a, n):>14.2f} {n * c + 1:>10.2f} {thm38(a, n, c):>9.2f}x")
    print("cost/pass = n*c + 1 target-steps; predicted = E / cost.  Break-even c for code n=2:", end=" ")
    if "code" in accept and 2 in accept["code"]:
        a = accept["code"][2][0] / accept["code"][2][1]
        print(f"(E-1)/n = ({expected_tokens(a, 2):.2f}-1)/2 = {(expected_tokens(a, 2) - 1) / 2:.2f};"
              f" for 2x you'd need c <= {(expected_tokens(a, 2) / 2 - 1) / 2:.2f}")
    else:
        print("(needs the code prompt at n=2)")

    if server:
        print("\n== (c) measured, llama-server, target alone vs target + draft (n = 4) ==")
        print(f"{'prompt':>7} {'target tok/s':>13} {'+draft tok/s':>13} {'measured':>9} {'predicted n=4':>14}")
        for k, (base, spec) in server.items():
            pred = thm38(accept[k][4][0] / accept[k][4][1], 4, c) if accept.get(k, {}).get(4, (0, 0))[1] else float("nan")
            print(f"{k:>7} {base:>13.1f} {spec:>13.1f} {spec / base:>8.2f}x {pred:>13.2f}x")
        print("If a speedup is far above the prediction (e.g. chat 2.2x in the author's busy run), look at the")
        print("individual runs: on a loaded machine each run lands 'slow' or 'fast' at random.")


def papers_and_batch():
    print("\n== (d) the papers through the same formula (our recomputation) ==")
    print("Leviathan App A.3, T5-XXL 11B on TPU-v4, batch 1:  task draft temp gamma alpha c -> Thm 3.8 | paper Exp | paper Emp")
    for task, m, temp, g, a, c, exp, emp in [
            ("EnDe", "T5-small", 0, 7, .75, .02, 3.2, 3.4), ("EnDe", "T5-base", 0, 7, .80, .04, 3.3, 2.8),
            ("EnDe", "T5-large", 0, 7, .82, .11, 2.5, 1.7), ("EnDe", "T5-small", 1, 7, .62, .02, 2.3, 2.6),
            ("EnDe", "T5-base", 1, 5, .68, .04, 2.4, 2.4), ("EnDe", "T5-large", 1, 3, .71, .11, 2.0, 1.4),
            ("CNNDM", "T5-small", 0, 5, .65, .02, 2.4, 3.1), ("CNNDM", "T5-base", 0, 5, .73, .04, 2.6, 3.0),
            ("CNNDM", "T5-large", 0, 3, .74, .11, 2.0, 2.2), ("CNNDM", "T5-small", 1, 5, .53, .02, 1.9, 2.3),
            ("CNNDM", "T5-base", 1, 3, .55, .04, 1.8, 2.2), ("CNNDM", "T5-large", 1, 3, .56, .11, 1.6, 1.7)]:
        print(f"  {task:>5} {m:>8} t{temp} g{g} a{a:.2f} c{c:.2f} -> {thm38(a, g, c):.2f}x | {exp}x | {emp}x")
    c = 1.8 / 14.1
    print(f"Chen, Chinchilla 70B + 4B draft, 16 TPU v4, K = 4: c = 1.8 / 14.1 = {c:.3f}")
    for name, ms, paper in [("XSum nucleus", 7.52, 1.92), ("XSum greedy", 7.00, 2.01), ("HumanEval", 5.73, 2.46)]:
        lo, hi = 0.0, 0.9999                     # back-solve the alpha that Thm 3.8 would need
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if thm38(mid, 4, c) < paper else (lo, mid)
        print(f"  {name:>12}: 14.1 / {ms} = {14.1 / ms:.2f}x (paper: {paper}x); implied alpha ~ {lo:.2f}")

    print("\n== (e) why it fades with batch: course 2's model (Llama-3.1-8B + 1.0B draft, H100, alpha 0.8, K 4) ==")
    P_t, P_d, BW, C, a, k = 8.03e9, 1.0e9, 3.35e12, 9.89e14, 0.8, 4
    E = expected_tokens(a, k)
    print(f"{'batch':>6} {'normal ms':>10} {'verify tokens':>14} {'verify ms':>10} {'draft ms':>9} {'speedup':>8}")
    for B in (1, 32, 64, 128, 256):
        t_norm = max(2 * P_t / BW, 2 * P_t * B / C)
        tv = max(2 * P_t / BW, 2 * P_t * B * (k + 1) / C)
        td = k * max(2 * P_d / BW, 2 * P_d * B / C)
        print(f"{B:6d} {t_norm * 1e3:10.2f} {B * (k + 1):14d} {tv * 1e3:10.2f} {td * 1e3:9.2f} {E * t_norm / (tv + td):7.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help="target GGUF (default: Ollama's llama3.2:3b)")
    ap.add_argument("--draft", help="draft GGUF (default: Ollama's llama3.2:1b)")
    ap.add_argument("--n-predict", type=int, default=200)
    ap.add_argument("--runs", type=int, default=5, help="alternated server runs per prompt")
    ap.add_argument("--prompts", default="code,chat,repeat")
    ap.add_argument("--draft-n", default="2,4,8")
    ap.add_argument("--quick", action="store_true", help="smoke test: 8 tokens, code prompt, n=2 and 4, 1 run")
    ap.add_argument("--recorded", action="store_true", help="print the author's recorded numbers, measure nothing")
    args = ap.parse_args()

    if args.recorded:
        print("Recorded run: Apple M3, llama.cpp 0.4.1 Metal, 3B Q4_K_M target / 1B Q8_0 draft, 2026-09-22/23.")
        print("(c) was measured with load average ~33 on 8 cores: target-only fell from 28.8 to ~16 tok/s.")
        report(RECORDED["tg"], RECORDED["accept"], RECORDED["server"], [2, 4, 8])
        papers_and_batch()
        return

    need_tools()
    target = args.target or ollama_gguf("llama3.2:3b")
    draft = args.draft or ollama_gguf("llama3.2:1b")
    if not (target and draft):
        sys.exit("Couldn't find the GGUF files. Run `ollama pull llama3.2:3b` and `ollama pull llama3.2:1b`,\n"
                 "or pass --target and --draft (see `ollama show --modelfile llama3.2:3b`, FROM line).")
    if args.quick:
        args.n_predict, args.runs, args.prompts, args.draft_n = 8, 1, "code", "2,4"
    prompts = {k: PROMPTS[k] for k in args.prompts.split(",")}
    n_list = [int(x) for x in args.draft_n.split(",")]
    print("target:", target)
    print("draft: ", draft)
    load_warning()

    bn, br = (8, 1) if args.quick else (128, 3)
    print(f"\n(a) llama-bench -n {bn} -r {br} on each model ...", flush=True)
    tg = {"target": bench_tg(target, bn, br), "draft": bench_tg(draft, bn, br)}

    print(f"(b) llama-speculative, {args.n_predict} tokens, temp 0 ...", flush=True)
    accept = {}
    for k, pr in prompts.items():
        accept[k] = {}
        for n in n_list:
            accept[k][n] = spec_accept(target, draft, pr, n, args.n_predict)
            print(f"    {k} n={n}: {accept[k][n][0]}/{accept[k][n][1]} accepted", flush=True)

    print(f"(c) llama-server alone vs with draft (n=4), {args.runs} alternated runs each ...", flush=True)
    a = start_server(target, 18771, [])
    b = start_server(target, 18772, ["-md", draft, "-ngld", "99", "--spec-draft-n-max", "4"])
    server = {}
    try:
        complete(18771, "Hello", 4); complete(18772, "Hello", 4)        # warm-up
        for k, pr in prompts.items():
            base, spec = [], []
            for _ in range(args.runs):
                base.append(complete(18771, pr, args.n_predict))
                spec.append(complete(18772, pr, args.n_predict))
            server[k] = (statistics.median(base), statistics.median(spec))
            print(f"    {k}: alone {[round(x, 1) for x in base]}  +draft {[round(x, 1) for x in spec]}", flush=True)
    finally:
        a.terminate(); b.terminate(); a.wait(); b.wait()
    load_warning()
    report(tg, accept, server, n_list)
    papers_and_batch()
    if args.quick:
        print("\n(--quick: 8 tokens is too few for meaningful acceptance or speed; this only checks the plumbing.)")


if __name__ == "__main__":
    main()

# Try this:
# 1. A cheaper draft: pass --draft with a much smaller GGUF of the same tokenizer family (or a more
#    quantized 1B, e.g. Q4_K_M). c falls; watch whether the Thm 3.8 prediction for code n=2 crosses 1.2x.
# 2. Temperature: set "temperature": 0.8 in complete() and --temp 0.8 in spec_accept(). Acceptance drops
#    for chat much more than for code (Leviathan Table 3: t=1 alphas are lower than t=0).
# 3. Run it twice, once with a video call open and once with everything closed, and compare (c):
#    that gap is why the card trusts acceptance rates more than timings.
