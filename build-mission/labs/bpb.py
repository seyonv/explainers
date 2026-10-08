"""Bits per byte (BPB) on the forecast-discussion test file, for the baselines every model has to beat.

BPB = total negative log-likelihood of the test text, in bits, divided by its size in UTF-8 bytes. It doesn't
care how a model splits text into tokens, so a byte-level model, GPT-2 and a compressor are on one scale.
Lower is better; 8.0 is "no idea" (every byte equally likely).

  python3 bpb.py --gzip                 # compressors: no learning at all
  python3 bpb.py --bigram --mb 1        # a byte bigram trained on your 1 MB file
  uv run --python 3.12 --with torch --with transformers python bpb.py --gpt2      # GPT-2, zero-shot
  uv run ... python bpb.py --gpt2 --model path/to/finetuned                        # any saved causal LM

Reads data/corpus/ written by afd_corpus.py (or --data DIR). Prints the test file's SHA-256 so you can check
you're scoring the same bytes as everyone else.
"""
import argparse, gzip, hashlib, json, lzma, math, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def corpus(data):
    return Path(data) / "corpus"


def read(path):
    return Path(path).read_bytes()


def gzip_bpb(test: bytes):
    return {"gzip": 8 * len(gzip.compress(test, 9)) / len(test), "xz": 8 * len(lzma.compress(test, preset=9)) / len(test)}


def bigram_counts(train: bytes):
    return Counter(zip(train, train[1:])), Counter(train[:-1])


def bigram_bpb(pairs, firsts, text: bytes, k: float):
    """Add-k smoothed P(next byte | this byte). The first byte is scored as uniform (8 bits)."""
    bits = 8.0
    for a, b in zip(text, text[1:]):
        bits -= math.log2((pairs[(a, b)] + k) / (firsts[a] + 256 * k))
    return bits / len(text)


def best_k(pairs, firsts, val: bytes):
    """Pick the smoothing on validation text, never on test."""
    return min([0.001, 0.01, 0.03, 0.1, 0.3, 1.0], key=lambda k: bigram_bpb(pairs, firsts, val[:300_000], k))


def lm_bpb(model, tok, text: str, ctx=1024, stride=512, device="cpu", log=print):
    """Sliding-window NLL of a causal LM over the whole text: each token is scored once, with up to
    ctx - stride tokens of context it was not itself predicted in."""
    import torch
    ids = tok(text, return_tensors="pt").input_ids[0]
    nll, scored, prev_end, t0 = 0.0, 0, 0, time.time()
    for begin in range(0, len(ids), stride):
        end = min(begin + ctx, len(ids))
        x = ids[begin:end].unsqueeze(0).to(device)
        new = end - prev_end                      # tokens not scored by an earlier window
        with torch.no_grad():
            logits = model(x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        tgt = x[0, 1:]
        tok_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        take = tok_lp[-new:] if begin else tok_lp
        nll -= take.sum().item(); scored += len(take)
        prev_end = end
        if end == len(ids):
            break
    nll += math.log(len(tok.get_vocab()))         # the very first token has no context: charge it uniform
    n_bytes = len(text.encode("utf-8"))
    log(f"  {len(ids):,} tokens over {n_bytes:,} bytes ({len(ids) / n_bytes:.3f} tokens per byte), {time.time() - t0:.0f}s")
    return nll / math.log(2) / n_bytes, len(ids)


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def gpt2_bpb(test_text, name="gpt2", log=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = device()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(dev).eval()
    log(f"{name} on {dev}, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")
    return lm_bpb(model, tok, test_text, device=dev, log=log)


def write_facts(a, c, test):
    """Every baseline the mission page shows, measured in one go, plus the training runs in results.jsonl."""
    man = json.loads((c / "MANIFEST.json").read_text())
    val = read(c / "val.txt")
    F = {"generated": time.strftime("%Y-%m-%d"), "test_sha256": hashlib.sha256(test).hexdigest(), "test_bytes": len(test),
         "test_dates": man["test_dates"], "val_dates": man.get("val_dates"), "train_dates_quick": man["train_dates"], "quick_seconds": man.get("seconds"), "files": man["files"], "rows": []}
    F["rows"] += [{"model": k, "bpb": round(v, 4)} for k, v in gzip_bpb(test).items()]
    for mb in (1, 3):
        pairs, firsts = bigram_counts(read(c / f"train_{mb}MB.txt"))
        k = best_k(pairs, firsts, val)
        F["rows"].append({"model": "byte bigram", "train_mb": mb, "k": k, "bpb": round(bigram_bpb(pairs, firsts, test, k), 4)})
    t0 = time.time()
    bpb, n = gpt2_bpb(test.decode("utf-8"))
    F["rows"].append({"model": "gpt2 zero-shot", "bpb": round(bpb, 4), "tokens": n, "tokens_per_byte": round(n / len(test), 4), "seconds": round(time.time() - t0)})
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tr = read(c / "train_1MB.txt")
    F["train_1mb_gpt2_tokens_per_byte"] = round(len(tok(tr.decode("utf-8")).input_ids) / len(tr), 4)
    F["embedding_params_at_width_144"] = {"byte": 256 * 144, "gpt2_bpe": len(tok) * 144}
    from sweep import grid
    g = grid()
    F["sweep"] = {"runs": len(g), "scratch_flops": sum(r["flops"] for r in g if r["kind"] == "scratch"), "gpt2_flops_cap": sum(r["flops"] for r in g if r["kind"] == "gpt2"),
                  "sizes": list(dict.fromkeys(r["size"] for r in g if r["kind"] == "scratch")),
                  "epochs_10M": {str(r["mb"]): round(r["epochs"], 2) for r in g if r["size"] == "10M" and r["mb"] in (30, 100)}}
    if a.full_data:
        fm = json.loads((Path(a.full_data) / "corpus" / "MANIFEST.json").read_text())
        F["full"] = {"seconds": fm.get("seconds"), "train_days": len(fm["train_dates"]), "files": fm["files"]}
    if Path(a.results).exists():
        F["runs"] = [json.loads(l) for l in Path(a.results).read_text().splitlines() if l.strip()]
    Path(a.facts).write_text(json.dumps(F, indent=1) + "\n")
    print("wrote", a.facts)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bits per byte on the held-out forecast discussions")
    ap.add_argument("--gzip", action="store_true")
    ap.add_argument("--bigram", action="store_true")
    ap.add_argument("--gpt2", action="store_true")
    ap.add_argument("--model", default="gpt2", help="a Hub id or a local folder saved by finetune_gpt2.py")
    ap.add_argument("--mb", type=int, default=1, help="training size for --bigram")
    ap.add_argument("--data", default=HERE / "data")
    ap.add_argument("--json", action="store_true", help="print one JSON line per result")
    ap.add_argument("--facts", default="", help="measure every baseline and write them, plus results.jsonl rows, to this JSON file")
    ap.add_argument("--results", default=HERE / "results.jsonl")
    ap.add_argument("--full-data", default="", help="with --facts: a data folder made by afd_corpus.py --full, to record its sizes and download time")
    a = ap.parse_args(argv)
    c = corpus(a.data)
    test = read(c / "test.txt")
    print(f"test.txt: {len(test):,} bytes, sha256 {hashlib.sha256(test).hexdigest()[:12]}")
    if a.facts:
        return write_facts(a, c, test)
    out = []
    if a.gzip:
        for k, v in gzip_bpb(test).items():
            out.append({"model": k, "bpb": v})
    if a.bigram:
        train, val = read(c / f"train_{a.mb}MB.txt"), read(c / "val.txt")
        pairs, firsts = bigram_counts(train)
        k = best_k(pairs, firsts, val)
        out.append({"model": "byte bigram", "train_mb": a.mb, "k": k, "bpb": bigram_bpb(pairs, firsts, test, k)})
    if a.gpt2:
        bpb, n = gpt2_bpb(test.decode("utf-8"), a.model)
        out.append({"model": a.model, "bpb": bpb, "tokens": n})
    for r in out:
        print(json.dumps(r) if a.json else f"{r['model']:>14}  {r['bpb']:.3f} bits per byte" + (f"  (trained on {r['train_mb']} MB, k={r['k']})" if "k" in r else ""))
    if not out:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
