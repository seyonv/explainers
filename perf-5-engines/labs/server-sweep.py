"""Benchmark a server on your Mac: the throughput-latency curve.  (course 5, benchmarking card)

Starts llama-server (llama.cpp's OpenAI-style server, continuous batching) with Llama-3.2-3B,
and for each concurrency C = 1, 2, 4, 8, 16:
  - restarts the server with C parallel slots,
  - sends one warm-up request (not timed),
  - fires C requests at once (a closed-loop burst: ~150-token prompts, 128 output tokens,
    temperature 0, ignore EOS, prompt cache off, a different ticket number in every prompt),
  - streams each reply and records TTFT, TPOT = (E2E - TTFT) / (n - 1) and total output tokens/s,
  - repeats 3 times and reports the median.
Then it prints the derived numbers the card uses (throughput gain, goodput under an illustrative
SLO, a prefill back-of-envelope for TTFT) and, with numpy, what vLLM's --burstiness does to arrivals.

Setup (once):
  brew install llama.cpp          # gives llama-server
  ollama pull llama3.2:3b         # or any GGUF; the lab reads the FROM line of
                                  #   `ollama show --modelfile llama3.2:3b` to find the file
  python3 labs/server-sweep.py --model /path/to/model.gguf     # to use a GGUF directly

Run:
  python3 labs/server-sweep.py              # full sweep: ~5-10 minutes on an M3. CLOSE OTHER APPS FIRST.
  python3 labs/server-sweep.py --quick      # smoke test: C = 1, 8 output tokens, one run (~30 s)
  python3 labs/server-sweep.py --recorded   # no measuring: reprint the author's sweep (M3, 2026-09-23)

Timings are load-sensitive. The author's run was taken while the Mac was busy (load average ~33
on 8 cores from other processes), so treat those numbers as rough and compare with your own.
"""
import argparse, json, os, re, shutil, statistics, subprocess, sys, threading, time, urllib.request

import numpy as np

PORT = 18766
LEVELS = (1, 2, 4, 8, 16)
BASE = "You are a helpful assistant. Summarise the following note in three bullet points. Note: "
FILLER = "The quarterly planning meeting covered hiring, the office move, and the new release schedule. "

# The author's run: Apple M3 8-core, 24 GB, llama.cpp 0.4.1 Metal, Llama-3.2-3B-Instruct Q4_K_M,
# 128 output tokens, median of 3 runs (sweep2_results.json). Machine busy (load ~33).
RECORDED = {
    1:  {"agg": 28.913,  "ttft": 0.421, "ttft_max": 0.421, "tpot_ms": 31.518,  "user": 31.728, "runs": [28.91, 31.84, 24.74]},
    2:  {"agg": 48.287,  "ttft": 0.773, "ttft_max": 0.774, "tpot_ms": 35.671,  "user": 28.034, "runs": [48.67, 48.29, 42.92]},
    4:  {"agg": 36.064,  "ttft": 1.731, "ttft_max": 1.733, "tpot_ms": 93.961,  "user": 10.643, "runs": [36.06, 35.06, 37.76]},
    8:  {"agg": 36.427,  "ttft": 4.183, "ttft_max": 4.185, "tpot_ms": 188.397, "user": 5.308,  "runs": [36.43, 35.23, 39.44]},
    16: {"agg": 100.182, "ttft": 5.155, "ttft_max": 6.335, "tpot_ms": 118.154, "user": 8.464,  "runs": [100.18, 92.69, 102.03]},
}
PP512 = 407.9          # llama-bench prompt processing, 3B, tok/s (course facts)
PROMPT_TOKENS = 150    # approximate prompt length
SLO_TTFT_S, SLO_TPOT_MS = 2.0, 100.0   # illustrative SLO for the goodput example


def ollama_gguf(tag):
    """Return the GGUF blob path Ollama uses for `tag` (the FROM line of its modelfile)."""
    if not shutil.which("ollama"):
        return None
    out = subprocess.run(["ollama", "show", "--modelfile", tag], capture_output=True, text=True).stdout
    m = re.search(r"^FROM\s+(\S+)", out, re.M)
    return m.group(1) if m and os.path.exists(m.group(1)) else None


def start(model, slots):
    p = subprocess.Popen(["llama-server", "-m", model, "--port", str(PORT), "-ngl", "99",
                          "-c", str(1024 * slots), "-np", str(slots), "--cont-batching"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return p
        except Exception:
            time.sleep(0.5)
    p.kill()
    sys.exit("llama-server did not start (is port %d free?)" % PORT)


def one(i, n_predict, out):
    prompt = BASE + f"(ticket {i}) " + FILLER * 8          # unique text per request: no shared-cache wins
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "stream": True,
                       "cache_prompt": False, "ignore_eos": True}).encode()
    t0 = time.perf_counter(); first = None; n = 0
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            d = json.loads(line[6:])
            if d.get("content"):
                n += 1
                if first is None:
                    first = time.perf_counter()
            if d.get("stop"):
                break
    t1 = time.perf_counter()
    out[i] = {"ttft": first - t0, "e2e": t1 - t0, "tokens": n, "tpot": (t1 - first) / max(n - 1, 1)}


def sweep(model, levels, n_predict, reps):
    results = {}
    for C in levels:
        p = start(model, C)
        try:
            one(10_000, n_predict, {})                      # warm-up, not timed
            runs = []
            for rep in range(reps):
                out = {}
                ths = [threading.Thread(target=one, args=(rep * 100 + i, n_predict, out)) for i in range(C)]
                t0 = time.perf_counter()
                for t in ths: t.start()
                for t in ths: t.join()
                wall = time.perf_counter() - t0
                v = out.values()
                runs.append({"agg": sum(x["tokens"] for x in v) / wall,
                             "ttft": statistics.median(x["ttft"] for x in v),
                             "ttft_max": max(x["ttft"] for x in v),
                             "tpot_ms": 1000 * statistics.median(x["tpot"] for x in v),
                             "user": statistics.median(1 / x["tpot"] for x in v)})
            results[C] = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
            results[C]["runs"] = [round(r["agg"], 2) for r in runs]
            print(f"  C = {C:2d} done: {results[C]['agg']:.1f} tok/s", flush=True)
        finally:
            p.terminate(); p.wait()
    return results


def report(res, n_out):
    print("\n== The throughput-latency curve (median of runs) ==")
    print(f"{'C':>3} {'total tok/s':>12} {'per-user tok/s':>15} {'median TTFT':>12} {'max TTFT':>9} {'median TPOT':>12}   runs (total tok/s)")
    for C, r in res.items():
        print(f"{C:>3} {r['agg']:>12.1f} {r['user']:>15.1f} {r['ttft']:>10.2f} s {r['ttft_max']:>7.2f} s "
              f"{r['tpot_ms']:>9.1f} ms   {r['runs']}")
    lo, hi = min(res), max(res)
    if lo == hi:
        return
    a, b = res[lo], res[hi]
    print(f"\nC = {lo} -> {hi}: total {a['agg']:.1f} -> {b['agg']:.1f} tok/s ({b['agg'] / a['agg']:.2f}x), "
          f"per user {a['user']:.1f} -> {b['user']:.1f} tok/s ({a['user'] / b['user']:.1f}x slower), "
          f"TTFT {a['ttft']:.2f} -> {b['ttft']:.2f} s ({b['ttft'] / a['ttft']:.1f}x)")

    print("\n== Check: total tok/s ~ C x n_out / (slowest TTFT + (n_out - 1) x TPOT) ==")
    for C, r in res.items():
        wall = r["ttft_max"] + (n_out - 1) * r["tpot_ms"] / 1000
        print(f"  C = {C:2d}: {C} x {n_out} / ({r['ttft_max']:.2f} + {n_out - 1} x {r['tpot_ms'] / 1000:.4f}) "
              f"= {C * n_out} / {wall:.2f} s = {C * n_out / wall:6.1f} tok/s   (measured {r['agg']:.1f})")

    print(f"\n== Why TTFT grows: all C prompts are prefilled before anyone's first token ==")
    print(f"  back-of-envelope: C x {PROMPT_TOKENS} prompt tokens / {PP512} tok/s (llama-bench pp512)")
    for C, r in res.items():
        print(f"  C = {C:2d}: {C * PROMPT_TOKENS:5d} tokens / {PP512} = {C * PROMPT_TOKENS / PP512:5.2f} s   "
              f"(measured median {r['ttft']:.2f} s, max {r['ttft_max']:.2f} s)")

    print(f"\n== Goodput under an illustrative SLO: TTFT <= {SLO_TTFT_S} s and TPOT <= {SLO_TPOT_MS:.0f} ms ==")
    print("  (judged on max TTFT and median TPOT, because per-request TPOT is not kept)")
    best = None
    for C, r in res.items():
        ok = r["ttft_max"] <= SLO_TTFT_S and r["tpot_ms"] <= SLO_TPOT_MS
        good = r["agg"] if ok else 0.0
        if ok and (best is None or good > best[1]):
            best = (C, good)
        print(f"  C = {C:2d}: TTFT {r['ttft_max']:.2f} s, TPOT {r['tpot_ms']:.1f} ms -> "
              f"{'meets' if ok else 'MISSES'}  goodput {good:6.1f} tok/s   (raw {r['agg']:.1f})")
    peak = max(res, key=lambda c: res[c]["agg"])
    if best:
        print(f"  best goodput: {best[1]:.1f} tok/s at C = {best[0]}; raw peak {res[peak]['agg']:.1f} tok/s at "
              f"C = {peak} misses the SLO")


def arrivals(seed=0):
    """What vLLM's --request-rate / --burstiness do: gamma inter-arrival gaps, shape = burstiness,
    mean gap = 1 / rate, so CV = 1 / sqrt(burstiness). burstiness = 1 is a Poisson process."""
    print("\n== Arrival shapes at 1 request/s (numpy, seed 0; what vllm bench serve --burstiness generates) ==")
    rng = np.random.default_rng(seed)
    print(f"{'burstiness':>10} {'CV formula':>11} {'CV measured':>12} {'busiest 1 s':>12} {'empty seconds':>14}   first 20 s (# per second)")
    for b in (5.0, 1.0, 0.25, 0.1):
        gaps = rng.gamma(shape=b, scale=1.0 / b, size=10_000)       # mean 1 s
        t = np.cumsum(gaps)
        t = t[t < 10_000]
        counts = np.bincount(t.astype(int), minlength=10_000)[:10_000]
        print(f"{b:>10} {1 / np.sqrt(b):>11.2f} {gaps.std() / gaps.mean():>12.2f} {counts.max():>12d} "
              f"{(counts == 0).mean():>13.0%}   {''.join(str(min(c, 9)) for c in counts[:20])}")
    print("  same average load (1 req/s) in every row; lower burstiness = the same requests in clumps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="path to a GGUF file (default: Ollama's llama3.2:3b)")
    ap.add_argument("--quick", action="store_true", help="smoke test: C = 1, 8 tokens, one run")
    ap.add_argument("--recorded", action="store_true", help="reprint the author's numbers, no measuring")
    args = ap.parse_args()

    if args.recorded:
        print("Author's sweep: Apple M3, llama.cpp 0.4.1 Metal, Llama-3.2-3B Q4_K_M, 128 output tokens,")
        print("median of 3 runs, 2026-09-23. The Mac was busy (load ~33 on 8 cores): rough numbers.")
        report(RECORDED, 128)
        arrivals()
        return

    if not shutil.which("llama-server"):
        print("llama-server not found. Install llama.cpp:  brew install llama.cpp")
        print("Or see the author's numbers without measuring:  python3 labs/server-sweep.py --recorded")
        sys.exit(1)
    model = args.model or ollama_gguf("llama3.2:3b")
    if not model:
        print("No model found. Either `ollama pull llama3.2:3b` (the lab reads the FROM line of")
        print("`ollama show --modelfile llama3.2:3b`) or pass --model /path/to/model.gguf")
        sys.exit(1)

    load = os.getloadavg()[0]
    cores = os.cpu_count() or 8
    print(f"Model: {model}")
    print(f"Load average {load:.1f} on {cores} cores." +
          ("  WARNING: the machine is busy; close other apps or the timings will be skewed." if load > cores / 2 else ""))
    print("Close other apps first: this benchmark measures the whole machine, not just the model.")

    levels, n_out, reps = ((1,), 8, 1) if args.quick else (LEVELS, 128, 3)
    res = sweep(model, levels, n_out, reps)
    report(res, n_out)
    arrivals()


if __name__ == "__main__":
    main()

# Try this:
#   1. Set cache_prompt True and use the same prompt for every request (drop the ticket number):
#      TTFT falls at high C because the prefill is shared. That is the prefix-cache trap.
#   2. Change LEVELS to (1, 2, 3, 4, 6, 8, 12, 16) to see exactly where the Metal dip starts and ends.
#   3. Replace the closed-loop burst with open-loop arrivals: start each request at np.cumsum of
#      rng.exponential(1 / rate) and watch TTFT explode once rate exceeds the C = 16 throughput / 128.
