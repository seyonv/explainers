"""The size x data sweep, and where (if anywhere) training from scratch overtakes fine-tuned GPT-2.

  python3 sweep.py --plan                 # the grid, its FLOPs, and how long it takes at your measured speed
  python3 sweep.py --commands             # the exact command for every run not yet in results.jsonl
  python3 sweep.py --crossover            # best from-scratch vs GPT-2 fine-tuned, per corpus size
  python3 sweep.py --chart curves.svg     # the chart for your post: one line per model, BPB against corpus size

Every from-scratch run trains on 20 x its parameter count in bytes (the compute-optimal rule of thumb from
Hoffmann et al. 2022), capped at 16 passes over the corpus (Muennighoff et al. 2023 found repeated data stops
helping well before that). Each GPT-2 fine-tune may use up to the FLOPs of the largest from-scratch run on the
same corpus, so the baseline is never short of compute. Every run stops early once validation stops improving.
"""
import argparse, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIZES = {"1M": 0.98e6, "3M": 3.2e6, "10M": 10.0e6, "30M": 31.6e6}     # approximate; tiny_gpt.py prints the exact count
CORPUS_MB = [1, 3, 10, 30, 100]
GPT2 = 124.4e6
UV = "uv run --python 3.12 --with numpy --with torch"


def grid():
    rows = []
    for mb in CORPUS_MB:
        for s, n in SIZES.items():
            tokens = min(20 * n, 16 * mb * 1e6)
            rows.append({"kind": "scratch", "size": s, "mb": mb, "tokens": tokens, "flops": 6 * n * tokens, "epochs": tokens / (mb * 1e6)})
        top = max(r["flops"] for r in rows if r["mb"] == mb and r["kind"] == "scratch")
        rows.append({"kind": "gpt2", "size": "124M", "mb": mb, "tokens": top / (6 * GPT2), "flops": top, "epochs": None})
    return rows


def results(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def speed(res):
    """Measured FLOPs per second on this machine, per model kind, from earlier runs."""
    out = {}
    for r in res:
        if r.get("seconds"):
            k = r.get("size", "124M")
            out[k] = max(out.get(k, 0), r["train_flops"] / r["seconds"])
    return out


def matches(r, row):
    """A logged run counts for a grid cell only if it used that cell's budget, so a quick or timed run you did
    by hand (Day 3, say) is never mistaken for the sweep's run."""
    if r.get("train_mb") != row["mb"]:
        return False
    if row["kind"] == "scratch":
        return r.get("size") == row["size"] and abs((r.get("tokens_budget") or 0) - row["tokens"]) <= 0.01 * row["tokens"]
    return r["model"] == "GPT-2 fine-tuned" and abs((r.get("flops_budget") or 0) - row["flops"]) <= 0.01 * row["flops"]


def done(res, row):
    return any(matches(r, row) for r in res)


def sweep_rows(res):
    return [r for r in res if any(matches(r, row) for row in grid())]


def command(row):
    if row["kind"] == "scratch":
        return f"{UV} python tiny_gpt.py --size {row['size']} --mb {row['mb']} --tokens {row['tokens']:.6g} --save models/{row['size']}-{row['mb']}MB.pt"
    return f"{UV} --with transformers python finetune_gpt2.py --mb {row['mb']} --flops {row['flops']:.6g}"


def crossover(res):
    """Per corpus size: the test BPB of the from-scratch model and the GPT-2 fine-tune with the best validation BPB
    (never chosen on test), and the smallest size where from scratch wins. Sizes without both numbers are skipped."""
    table = []
    for mb in CORPUS_MB:
        sc = [r for r in res if r.get("train_mb") == mb and r["model"].startswith("from scratch")]
        ft = [r for r in res if r.get("train_mb") == mb and r["model"] == "GPT-2 fine-tuned"]
        if sc and ft:
            b, f = min(sc, key=lambda r: r["val_bpb"]), min(ft, key=lambda r: r["val_bpb"])
            table.append({"mb": mb, "scratch_bpb": b["test_bpb"], "scratch_size": b["size"], "scratch_val": b["val_bpb"], "gpt2_ft_bpb": f["test_bpb"],
                          "model_file": f"models/{b['size']}-{mb}MB.pt"})
    wins = [t["mb"] for t in table if t["scratch_bpb"] < t["gpt2_ft_bpb"]]
    return table, (min(wins) if wins else None)


def chart(res, path, W=640, H=380):
    """Test bits per byte against corpus size (log scale), one line per model size plus GPT-2 fine-tuned. Where a
    cell has several runs, the one with the best validation BPB is drawn."""
    series, chosen = {}, {}
    for r in res:
        if "train_mb" in r and "test_bpb" in r:
            name = "GPT-2 fine-tuned" if r["model"] == "GPT-2 fine-tuned" else r["model"].replace("from scratch ", "from scratch, ")
            k = (name, r["train_mb"])
            if k not in chosen or r["val_bpb"] < chosen[k]["val_bpb"]:
                chosen[k] = r
                series.setdefault(name, {})[r["train_mb"]] = r["test_bpb"]
    ys = [v for s_ in series.values() for v in s_.values()] or [1, 2]
    lo, hi = math.floor(min(ys) * 10) / 10, math.ceil(max(ys) * 10) / 10 + 0.05
    L, R, T, B = 60, W - 150, 30, H - 50
    X = lambda mb: L + (math.log10(mb) - 0) / 2 * (R - L)
    Y = lambda v: T + (hi - v) / (hi - lo) * (B - T)
    colors = ["#4a9d77", "#3b6fb0", "#a66a00", "#7a5bb5", "#c0492b"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="#fbfbf9"/>', f'<text x="{L}" y="18" font-weight="700" font-size="14">Bits per byte on held-out forecast discussions (lower is better)</text>']
    for mb in CORPUS_MB:
        out.append(f'<line x1="{X(mb):.1f}" x2="{X(mb):.1f}" y1="{T}" y2="{B}" stroke="#ecece8"/><text x="{X(mb):.1f}" y="{B + 18}" text-anchor="middle" fill="#6b7068">{mb} MB</text>')
    for k in range(int(lo * 10), int(hi * 10) + 1, 2):
        v = k / 10
        out.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#6b7068">{v:.1f}</text>')
    for i, (name, pts) in enumerate(sorted(series.items(), key=lambda kv: kv[0] != "GPT-2 fine-tuned")):
        c = "#1c1f23" if name == "GPT-2 fine-tuned" else colors[i % len(colors)]
        p = " ".join(f"{X(mb):.1f},{Y(v):.1f}" for mb, v in sorted(pts.items()))
        dash = ' stroke-dasharray="6 4"' if name == "GPT-2 fine-tuned" else ""
        out.append(f'<polyline points="{p}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
        last = max(pts)
        out.append(f'<text x="{X(last) + 8:.1f}" y="{Y(pts[last]) + 4:.1f}" fill="{c}">{name}</text>')
    _, x = crossover(res)
    if x:
        out.append(f'<line x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{T}" y2="{B}" stroke="#c0492b" stroke-dasharray="3 3"/><text x="{X(x) + 4:.1f}" y="{T + 12}" fill="#c0492b">from scratch wins from {x} MB</text>')
    out.append(f'<text x="{(L + R) / 2:.0f}" y="{H - 10}" text-anchor="middle" fill="#6b7068">training text (log scale)</text></svg>')
    Path(path).write_text("\n".join(out) + "\n")
    return len(series)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true"); g.add_argument("--commands", action="store_true"); g.add_argument("--crossover", action="store_true"); g.add_argument("--chart", metavar="SVG")
    ap.add_argument("--results", default=HERE / "results.jsonl")
    a = ap.parse_args(argv)
    res = results(a.results)
    if a.plan:
        sp = speed(res)
        total = 0.0
        print(f"{'run':<24}{'corpus':>8}{'train FLOPs':>14}{'epochs':>9}{'your time':>15}")
        for r in grid():
            fps = sp.get(r["size"])
            hrs = r["flops"] / fps / 3600 if fps else None
            total += hrs or 0
            name = f"from scratch {r['size']}" if r["kind"] == "scratch" else "GPT-2 fine-tuned"
            ep = f"{r['epochs']:.1f}" if r["epochs"] else "up to"
            print(f"{name:<24}{r['mb']:>6} MB{r['flops']:>14.2e}{ep:>9}{(f'{hrs:.1f} h' if hrs else 'time one first'):>15}")
        print(f"\n{len(grid())} runs. Measured so far on this machine: {total:.1f} h for the sizes you've timed." if sp else f"\n{len(grid())} runs. Time one run of each size first; the plan then fills in your hours.")
    elif a.commands:
        for r in grid():
            if not done(res, r):
                print(command(r))
    elif a.chart:
        print(f"wrote {a.chart}: {chart(sweep_rows(res), a.chart)} lines")
    else:
        table, x = crossover(sweep_rows(res))
        for t in table:
            print(f"{t['mb']:>4} MB  from scratch {t['scratch_bpb']:.3f} ({t['scratch_size']})  GPT-2 fine-tuned {t['gpt2_ft_bpb']:.3f}  {'from scratch wins' if t['scratch_bpb'] < t['gpt2_ft_bpb'] else 'fine-tune wins'}")
        print(f"\ncrossover: {x} MB" if x else "\nno crossover in the sizes you've run: fine-tune GPT-2")
        if table:
            best = min(table, key=lambda t: t["scratch_val"])
            print(f"best from-scratch model: {best['model_file']} (validation {best['scratch_val']:.3f} bits per byte, the lowest); use it for sample.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
