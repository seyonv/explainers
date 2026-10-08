"""Coding-agent traffic vs the chat benchmark, for the Inference mission.

  python3 trace_lab.py --reuse                    # rung 0: how much of each workload's prompt a prefix cache could reuse
  python3 trace_lab.py --kv Qwen/Qwen3-8B         # how many tokens of KV cache fit on one GPU
  python3 trace_lab.py --slice --minutes 30       # a replayable slice of the agent trace for `vllm bench serve`
  python3 trace_lab.py --commands --day 4         # the exact server and benchmark commands for a day's configs
  python3 trace_lab.py --rank results/            # goodput per config on both workloads, both rankings, Kendall's tau
  python3 trace_lab.py --rank results/ --chart ranks.svg    # ... and the slope chart for the post

Data (downloaded on first use into ./data, or --data DIR):
- TraceLab (Zhu et al. 2026, arXiv 2606.30560): 357K LLM calls from real Claude Code and Codex sessions,
  token counts and timings, no prompt text. CC BY 4.0. github.com/uw-syfi/TraceLab
- ShareGPT_V3_unfiltered_cleaned_split.json (Hugging Face anon8231489123/ShareGPT_Vicuna_unfiltered, 673 MB):
  the file `vllm bench serve --dataset-name sharegpt` samples from.
Standard library only, except --reuse's ShareGPT half, which tokenizes with the `tokenizers` package:
  uv run --python 3.12 --with tokenizers python trace_lab.py --reuse
"""
import argparse, gzip, json, math, random, sys, urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACE_URL = "https://github.com/uw-syfi/TraceLab/releases/download/v0.0.1/syfi_coding_trace.jsonl.gz"
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
HF = "https://huggingface.co/{model}/resolve/main/{file}"
BLOCK = 16                      # vLLM's prefix-cache block size, in tokens
CTX = 40960                     # Qwen3-8B's native context length
SLO = {"ttft_ms": 2000, "tpot_ms": 50}   # decided before any run; every config is judged against it


def fetch(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url.rsplit('/', 1)[-1]} ...", file=sys.stderr)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "trace-lab/1.0"}), timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return path


# ---------- the agent trace ----------
def load_trace(path):
    """Sessions -> rounds in order. Each round keeps only what the replay needs."""
    sessions = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            ev = r.get("timing_events") or []
            sessions[r["session_id"]].append({
                "round": r["round_index"], "provider": r.get("provider"),
                "input": r.get("input_tokens_total") or 0, "prefix": r.get("prefix_tokens") or 0,
                "output": r.get("output_tokens") or 0,
                "start": ts(ev[0]["timestamp"]) if ev else None,
            })
    for rs in sessions.values():
        rs.sort(key=lambda x: x["round"])
    return dict(sessions)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def trace_reuse(sessions):
    """Two measures of how much of each prompt a cache already holds: what the providers reported serving from
    their prompt caches, and an upper bound from the previous prompt in the same session."""
    tot = pre = ub = 0
    for rs in sessions.values():
        prev = None
        for r in rs:
            tot += r["input"]; pre += r["prefix"]
            if prev is not None:
                ub += min(r["input"], prev["input"])
            prev = r
    return {"requests": sum(len(v) for v in sessions.values()), "sessions": len(sessions), "prompt_tokens": tot,
            "provider_cached_share": pre / tot, "previous_prompt_upper_bound": ub / tot}


# ---------- the chat benchmark, sampled the way vLLM samples it ----------
def sharegpt_prompts(path, n, encode, seed=0):
    """vLLM's ShareGPTDataset (v0.31.0): keep conversations with at least two turns, shuffle with random.seed(seed),
    use the first turn as the prompt and the second's length as the output, and skip a pair if the prompt or
    output is under 4 tokens, the prompt is over 1024, or both together are over 2048."""
    data = [e for e in json.loads(Path(path).read_text(encoding="utf-8")) if len(e.get("conversations", [])) >= 2]
    random.seed(seed)
    random.shuffle(data)
    out = []
    for e in data:
        if len(out) >= n:
            break
        p = encode(e["conversations"][0]["value"])
        o = len(encode(e["conversations"][1]["value"]))
        if len(p) < 4 or o < 4 or len(p) > 1024 or len(p) + o > 2048:
            continue
        out.append(p)
    return out


def block_reuse(prompts, block=BLOCK):
    """Share of full blocks already seen earlier in the replay, with each block's hash chained to the blocks
    before it in the same prompt, as vLLM's prefix cache keys them. An infinite cache: an upper bound."""
    seen, blocks, hits = set(), 0, 0
    for ids in prompts:
        h = None
        for i in range(0, len(ids) - block + 1, block):
            h = hash((h, tuple(ids[i:i + block])))
            blocks += 1
            hits += h in seen
            seen.add(h)
    return hits / blocks if blocks else 0.0


# ---------- KV maths ----------
def kv_capacity(cfg, weight_bytes, gpu_bytes=80e9, util=0.9, dtype_bytes=2):
    per_token = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]) * dtype_bytes
    free = gpu_bytes * util - weight_bytes
    return {"kv_bytes_per_token": per_token, "weight_bytes": weight_bytes, "gpu_bytes": gpu_bytes, "util": util,
            "kv_tokens": int(free // per_token), "kv_blocks": int(free // per_token // BLOCK)}


def hub_json(model, file, data):
    return json.loads(fetch(HF.format(model=model, file=file), Path(data) / "hub" / model.replace("/", "--") / file).read_text())


# ---------- the replay slice ----------
def make_slice(sessions, minutes, scale=8, seed=0, gap_mult=0.1, max_tokens=CTX, n_sessions=64):
    """A fixed, replayable slice: `n_sessions` sessions picked at random (seeded), each starting at a random
    moment in the first half of the slice and keeping its own pacing (gaps multiplied by gap_mult, so a user's
    minutes-long pause fits). More sessions = more load. Token counts are divided by `scale` to fit one GPU's
    context. Each session is one growing chain of block ids: a round reuses the first prefix/16/scale ids of
    the previous round, then appends new ones."""
    rng = random.Random(seed)
    order = sorted(sessions); rng.shuffle(order)
    out, dropped, total, nxt, used = [], 0, 0, 0, 0
    horizon = minutes * 60
    for sid in order:
        if used >= n_sessions:
            break
        rs = [r for r in sessions[sid] if r["start"] is not None]
        if len(rs) < 2:
            continue
        used += 1
        t0, offset = rs[0]["start"], rng.uniform(0, horizon * 0.5)
        chain = []
        for r in rs:
            t = offset + (r["start"] - t0) * gap_mult
            if t > horizon:
                break
            n_in, n_out = max(1, round(r["input"] / scale)), max(1, round(r["output"] / scale))
            reuse = min(len(chain), (r["prefix"] // scale) // BLOCK)
            need = math.ceil(n_in / BLOCK)
            chain = chain[:reuse] + list(range(nxt, nxt + max(0, need - reuse)))
            nxt += max(0, need - reuse)
            total += 1
            if n_in + n_out > max_tokens:
                dropped += 1
                continue
            out.append({"timestamp": round(t * 1000), "input_length": n_in, "output_length": n_out, "hash_ids": chain[:need], "session": sid})
    out.sort(key=lambda r: r["timestamp"])
    return out, {"requests": len(out), "sessions": used, "dropped_too_long": dropped, "considered": total, "scale": scale, "gap_mult": gap_mult, "seed": seed,
                 "requests_per_s": len(out) / horizon, "mean_input": sum(r["input_length"] for r in out) / max(1, len(out))}


def slice_reuse(rows, block=BLOCK):
    """Share of block ids in the slice already sent earlier: what an unbounded prefix cache could reuse."""
    seen, n, hit = set(), 0, 0
    for r in rows:
        for h in r["hash_ids"]:
            n += 1; hit += h in seen; seen.add(h)
    return hit / n if n else 0.0


# ---------- configs, commands and ranking ----------
CONFIGS = {
    "stock":        {"day": 4, "flags": ""},
    "prefix-off":   {"day": 4, "flags": "--no-enable-prefix-caching"},
    "batch-tokens": {"day": 5, "flags": "--max-num-batched-tokens 16384"},
    "long-prefill": {"day": 5, "flags": "--long-prefill-token-threshold 2048"},
    "max-seqs":     {"day": 5, "flags": "--max-num-seqs 64"},
    "kv-fp8":       {"day": 6, "flags": "--kv-cache-dtype fp8"},
    "util-95":      {"day": 6, "flags": "--gpu-memory-utilization 0.95"},
    "cpu-offload":  {"day": 6, "flags": "--kv-offloading-size 64 --kv-offloading-backend native"},
    "ngram-spec":   {"day": 7, "flags": "--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4}'"},
    "no-chunked":   {"day": 7, "flags": "--no-enable-chunked-prefill"},
}


def commands(day, model="Qwen/Qwen3-8B", rate=4, n=1000, seed=0, slice_path="agent-slice.jsonl"):
    """A bash script for one day's configs, to run on the GPU machine next to agent-slice.jsonl. For each config it
    starts vLLM, waits until it answers, runs both benchmarks, then stops it. Same model, seeds and SLO for all.
    Flags checked against vLLM v0.31.0's engine and benchmark arguments. PYTHONHASHSEED is pinned because vLLM's
    timed-trace replay seeds its synthetic prompt tokens with Python's string hash, which is salted per process."""
    if not Path(slice_path).exists():
        raise SystemExit(f"{slice_path} not found: make it first with  python3 trace_lab.py --slice")
    n_agent = sum(1 for _ in open(slice_path))
    good = f"--goodput ttft:{SLO['ttft_ms']} tpot:{SLO['tpot_ms']}"
    lines = ["#!/usr/bin/env bash",
             f"# Day {day}: run on the GPU machine, in a folder holding this script and {slice_path}.",
             f"# SLO (fixed before any run): time to first token <= {SLO['ttft_ms']} ms, time per output token <= {SLO['tpot_ms']} ms.",
             "set -euo pipefail", "mkdir -p results logs",
             f"[ -f ShareGPT_V3_unfiltered_cleaned_split.json ] || curl -L -o ShareGPT_V3_unfiltered_cleaned_split.json {SHAREGPT_URL}",
             "serve() {   # start vLLM in the background and wait until it answers",
             "  vllm serve \"$@\" > logs/server.log 2>&1 & SERVER=$!",
             "  until curl -sf localhost:8000/health > /dev/null; do sleep 5; kill -0 $SERVER || { tail -20 logs/server.log; exit 1; }; done",
             "}",
             "stop() { kill $SERVER; wait $SERVER 2>/dev/null || true; }"]
    for name, c in CONFIGS.items():
        if c["day"] != day:
            continue
        lines += [f"\n# {name}",
                  f"serve {model} --max-model-len {CTX} --seed {seed} {c['flags']}".rstrip(),
                  f"vllm bench serve --model {model} --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts {n} --request-rate {rate} --seed {seed} {good} --save-result --result-dir results --result-filename {name}.sharegpt.json",
                  f"curl -s localhost:8000/metrics > logs/{name}.before-agent.metrics.txt",
                  f"PYTHONHASHSEED=0 vllm bench serve --model {model} --dataset-name timed_trace --dataset-path {slice_path} --timed-trace-sec-multiplier 0.001 --num-prompts {n_agent} --seed {seed} {good} --save-result --result-dir results --result-filename {name}.agent.json",
                  f"curl -s localhost:8000/metrics > logs/{name}.after-agent.metrics.txt   # the difference is the agent run's preemptions and prefix-cache hits",
                  f"stop; cp logs/server.log logs/{name}.server.log"]
    return "\n".join(lines)


def kendall_tau(a, b):
    """Rank agreement between two orderings of the same items: 1 = identical, -1 = reversed."""
    items = [x for x in a if x in b]
    pa, pb = {x: i for i, x in enumerate(a)}, {x: i for i, x in enumerate(b)}
    c = d = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = (pa[items[i]] - pa[items[j]]) * (pb[items[i]] - pb[items[j]])
            c += s > 0; d += s < 0
    return (c - d) / (c + d) if c + d else 1.0


COUNTERS = {"preemptions": "vllm:num_preemptions", "prefix_hits": "vllm:prefix_cache_hits", "prefix_queries": "vllm:prefix_cache_queries"}


def counters(path):
    """Sum of each vLLM counter in a Prometheus /metrics dump (v0.31.0 names; exported with a _total suffix)."""
    out = dict.fromkeys(COUNTERS, 0.0)
    for line in Path(path).read_text().splitlines():
        for k, name in COUNTERS.items():
            if line.startswith((name + "{", name + " ", name + "_total{", name + "_total ")):
                out[k] += float(line.rsplit(" ", 1)[1])
    return out


def agent_cache(logs_dir):
    """Per config, the agent run's preemptions and prefix-cache hit rate: after-agent counters minus before-agent."""
    res = {}
    for f in sorted(Path(logs_dir).glob("*.before-agent.metrics.txt")):
        name = f.name.split(".")[0]
        after = f.with_name(f"{name}.after-agent.metrics.txt")
        if not after.exists():
            continue
        a, b = counters(f), counters(after)
        d = {k: b[k] - a[k] for k in COUNTERS}
        res[name] = {"preemptions": int(d["preemptions"]), "prefix_hit_rate": round(d["prefix_hits"] / d["prefix_queries"], 4) if d["prefix_queries"] else None}
    return res


def rank(results_dir, tie=0.02):
    """Goodput (requests per second meeting the SLO) per config on each workload, from vLLM's saved results.
    Configs within `tie` (2%) of the ShareGPT winner are listed as tied with it: a winner the benchmark can't
    separate from the runner-up is no basis for a rank flip."""
    good = defaultdict(dict)
    for f in sorted(Path(results_dir).glob("*.json")):
        name, work, _ = f.name.rsplit(".", 2)
        r = json.loads(f.read_text())
        good[work][name] = r.get("request_goodput", r.get("goodput"))
    order = {w: sorted(g, key=lambda k: -(g[k] or 0)) for w, g in good.items()}
    out = {"goodput": good, "ranking": order}
    if "sharegpt" in order and "agent" in order:
        top = order["sharegpt"][0]
        out["sharegpt_winner"] = top
        best_chat = good["sharegpt"][top] or 0
        out["sharegpt_tied_with_winner"] = [k for k in order["sharegpt"][1:] if (good["sharegpt"][k] or 0) >= best_chat * (1 - tie)]
        out["sharegpt_winner_rank_on_agent"] = order["agent"].index(top) + 1 if top in order["agent"] else None
        out["kendall_tau"] = kendall_tau(order["sharegpt"], order["agent"])
        stock = good["agent"].get("stock")
        best = good["agent"][order["agent"][0]]
        out["best_over_stock_on_agent"] = best / stock if stock else None
    logs = Path(results_dir).resolve().parent / "logs"
    if logs.is_dir():
        out["agent_cache"] = agent_cache(logs)
    return out


def slope_chart(r, path, W=560):
    """Two columns of configs, ranked by goodput on each workload, with a line joining each config's places."""
    a, b = r["ranking"].get("sharegpt", []), r["ranking"].get("agent", [])
    names = [x for x in a if x in b]
    H, top, gap = 90 + 30 * len(names), 70, 30
    xa, xb = 170, W - 170
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="13">', f'<rect width="{W}" height="{H}" fill="#fbfbf9"/>',
           f'<text x="{xa}" y="40" text-anchor="end" font-weight="700">ranked on ShareGPT</text><text x="{xb}" y="40" font-weight="700">ranked on agent traffic</text>']
    for name in names:
        ya, yb = top + gap * a.index(name), top + gap * b.index(name)
        moved = abs(a.index(name) - b.index(name)) >= 3
        c = "#c0492b" if moved else "#9aa1a9"
        out.append(f'<line x1="{xa + 8}" y1="{ya - 4}" x2="{xb - 8}" y2="{yb - 4}" stroke="{c}" stroke-width="{2.5 if moved else 1.5}"/>')
        out.append(f'<text x="{xa}" y="{ya}" text-anchor="end">{a.index(name) + 1}. {name}</text><text x="{xb}" y="{yb}">{b.index(name) + 1}. {name}</text>')
    out.append(f'<text x="{W / 2}" y="{H - 14}" text-anchor="middle" fill="#6b7068">Kendall\'s tau between the rankings: {r.get("kendall_tau", float("nan")):.2f}</text></svg>')
    Path(path).write_text("\n".join(out) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reuse", action="store_true"); g.add_argument("--kv", metavar="MODEL"); g.add_argument("--slice", action="store_true")
    g.add_argument("--commands", action="store_true"); g.add_argument("--rank", metavar="DIR"); g.add_argument("--facts", metavar="OUT")
    ap.add_argument("--data", default=HERE / "data")
    ap.add_argument("--sharegpt-sample", type=int, default=1000, help="as many first-turn prompts as the benchmark sends")
    ap.add_argument("--gpu-gb", type=float, default=80); ap.add_argument("--util", type=float, default=0.9)
    ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--scale", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sessions", type=int, default=64, help="sessions in the slice; more sessions, more load")
    ap.add_argument("--turns", action="store_true", help="with --slice: print each request's length and how much it shares with the one before")
    ap.add_argument("--day", type=int, default=4)
    ap.add_argument("--rate", type=float, default=4, help="with --commands: ShareGPT requests per second; raise it if every config meets the SLO")
    ap.add_argument("--chart", default="", help="with --rank: write the slope chart to this SVG")
    a = ap.parse_args(argv)
    data = Path(a.data)
    if a.commands:
        print(commands(a.day, rate=a.rate)); return 0
    if a.rank:
        r = rank(a.rank)
        if not r["goodput"]:
            print(f"no vLLM results in {Path(a.rank)} yet: copy the results folder back from the GPU machine", file=sys.stderr)
        if r.get("sharegpt_tied_with_winner"):
            print(f"warning: on ShareGPT, {', '.join(r['sharegpt_tied_with_winner'])} came within 2% of {r['sharegpt_winner']}. The chat benchmark can't separate them, "
                  "so don't claim a rank flip from this; rerun with a higher --rate", file=sys.stderr)
        print(json.dumps(r, indent=1))
        if a.chart:
            slope_chart(r, a.chart); print("wrote", a.chart, file=sys.stderr)
        return 0
    if a.kv:
        cfg = hub_json(a.kv, "config.json", data)
        idx = hub_json(a.kv, "model.safetensors.index.json", data)
        k = kv_capacity(cfg, idx["metadata"]["total_size"], a.gpu_gb * 1e9, a.util)
        print(f"{a.kv}: {k['kv_bytes_per_token']:,} bytes of KV per token; weights {k['weight_bytes'] / 1e9:.2f} GB")
        print(f"room for {k['kv_tokens']:,} tokens of KV ({k['kv_blocks']:,} blocks of {BLOCK}) on a {a.gpu_gb:g} GB GPU at {a.util:g} utilisation")
        print(f"(GB here means 10^9 bytes; an H100 80GB has 80 GiB, so rerun with --gpu-gb {80 * 2**30 / 1e9:.1f}. This also ignores activations and CUDA graphs; vLLM prints its own number at startup: compare them)")
        return 0
    trace = load_trace(fetch(TRACE_URL, data / "syfi_coding_trace.jsonl.gz"))
    if a.slice:
        rows, info = make_slice(trace, a.minutes, a.scale, a.seed, n_sessions=a.sessions)
        if not a.turns:   # --turns is for looking; it leaves your slice file alone
            with open("agent-slice.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps({k: r[k] for k in ("timestamp", "input_length", "output_length", "hash_ids", "session")}) + "\n")
        print(f"{'slice' if a.turns else 'wrote agent-slice.jsonl'}: {info['requests']:,} requests from {info['sessions']} session{'s' if info['sessions'] != 1 else ''} over {a.minutes:g} minutes "
              f"({info['requests_per_s']:.2f} per second, {info['mean_input']:,.0f} prompt tokens each on average); "
              f"{info['dropped_too_long']:,} of {info['considered']:,} dropped as longer than {CTX:,} tokens after dividing by {a.scale}; "
              f"{slice_reuse(rows):.1%} of its blocks repeat an earlier one")
        print(f"time inside each session runs {1 / info['gap_mult']:g}x faster than in the trace (gaps x{info['gap_mult']:g}), so tool calls and user pauses fit in {a.minutes:g} minutes")
        if a.turns:
            print(f"\nturn  prompt tokens (÷{a.scale})  blocks  blocks shared with the turn before")
            prev = []
            for k, r in enumerate(rows):
                h = r["hash_ids"]; same = 0
                while same < min(len(h), len(prev)) and h[same] == prev[same]:
                    same += 1
                print(f"{k + 1:>4}  {r['input_length']:>13,}  {len(h):>6,}  {same:>6,}  ({same / len(h):.0%})" if h else f"{k + 1:>4}")
                prev = h
            print("(TraceLab has token counts, not text: these block ids are made up, one chain per session, sharing exactly the prefix the trace reports as cached)")
        return 0
    t = trace_reuse(trace)
    res = {"trace": t}
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(fetch(HF.format(model="Qwen/Qwen3-8B", file="tokenizer.json"), data / "hub" / "Qwen--Qwen3-8B" / "tokenizer.json")))
        prompts = sharegpt_prompts(fetch(SHAREGPT_URL, data / "ShareGPT_V3_unfiltered_cleaned_split.json"), a.sharegpt_sample, lambda s: tok.encode(s).ids, a.seed)
        res["sharegpt"] = {"prompts": len(prompts), "prompt_tokens": sum(map(len, prompts)), "block_reuse": block_reuse(prompts)}
    except ImportError:
        print("(install `tokenizers` for the ShareGPT half: uv run --python 3.12 --with tokenizers python trace_lab.py --reuse)", file=sys.stderr)
    print(f"agent trace: {t['sessions']:,} sessions, {t['requests']:,} requests, {t['prompt_tokens'] / t['requests']:,.0f} prompt tokens per request on average")
    print(f"  served from the providers' prompt caches: {t['provider_cached_share']:.1%} of prompt tokens")
    print(f"  upper bound from the previous prompt in the session: {t['previous_prompt_upper_bound']:.1%}")
    if "sharegpt" in res:
        s = res["sharegpt"]
        print(f"chat benchmark (ShareGPT as vllm bench samples it): {s['prompts']:,} prompts, {s['prompt_tokens'] / s['prompts']:.0f} tokens each on average")
        print(f"  {BLOCK}-token blocks already seen earlier in the replay: {s['block_reuse']:.1%}")
    if a.facts:
        cfg = hub_json("Qwen/Qwen3-8B", "config.json", data)
        idx = hub_json("Qwen/Qwen3-8B", "model.safetensors.index.json", data)
        res["kv"] = kv_capacity(cfg, idx["metadata"]["total_size"])
        res["kv_80gib"] = kv_capacity(cfg, idx["metadata"]["total_size"], 80 * 2**30)
        res["ctx"] = cfg.get("max_position_embeddings")
        res["download_bytes"] = {"trace": (data / "syfi_coding_trace.jsonl.gz").resolve().stat().st_size,
                                 "sharegpt": (data / "ShareGPT_V3_unfiltered_cleaned_split.json").resolve().stat().st_size}
        res["bench"] = {"sharegpt_prompts": 1000, "rate": 4, "slice_minutes": 30, "configs": len(CONFIGS)}
        rows, info = make_slice(trace, 30, 8, 0)
        res["slice"] = {**info, "block_reuse": slice_reuse(rows)}
        res["fits_ctx_at_scale"] = {str(s): sum(1 for rs in trace.values() for r in rs if (r["input"] + r["output"]) / s <= CTX) / t["requests"] for s in (1, 4, 8, 16)}
        res["slo"] = SLO; res["generated"] = datetime.now().strftime("%Y-%m-%d")
        Path(a.facts).write_text(json.dumps(res, indent=1) + "\n")
        print("wrote", a.facts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
