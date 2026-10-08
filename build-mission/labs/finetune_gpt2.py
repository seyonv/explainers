"""The baseline your from-scratch model has to beat: GPT-2 (124M) fine-tuned on the same forecast text.

  uv run --python 3.12 --with torch --with transformers python finetune_gpt2.py --mb 1 --minutes 20
  uv run --python 3.12 --with torch --with transformers python finetune_gpt2.py --mb 10 --flops 3e15

Matched FLOPs: pass --flops with the training FLOPs of the from-scratch run you're comparing against (tiny_gpt.py
prints them), and fine-tuning stops at the same budget (6 x 124M parameters x tokens). Early stopping on the
validation file, then bits per byte on the test file. Appends one JSON line to results.jsonl.
"""
import argparse, json, math, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bpb import lm_bpb

HERE = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--flops", type=float, default=0, help="stop at this many training FLOPs (overrides --minutes)")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=3, help="stop after this many evaluations without a better validation score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="folder to save the best model in")
    ap.add_argument("--data", default=HERE / "data")
    ap.add_argument("--out", default=HERE / "results.jsonl")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    c = Path(a.data) / "corpus"
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(dev)
    N = sum(p.numel() for p in model.parameters())
    ids = tok((c / f"train_{a.mb}MB.txt").read_text(encoding="utf-8"), return_tensors="pt").input_ids[0]
    val = (c / "val.txt").read_text(encoding="utf-8")[:200_000]
    print(f"GPT-2: {N / 1e6:.0f}M parameters; {len(ids):,} training tokens from {a.mb} MB; on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(a.seed)
    quiet = lambda *_: None
    model.eval(); best = lm_bpb(model, tok, val, device=dev, log=quiet)[0]; model.train()
    best_state, step, seen, t0, stale = None, 0, 0, time.time(), 0
    print(f"step      0  val {best:.3f} bits per byte (zero-shot); it prints again every {a.eval_every} steps", flush=True)
    done = lambda: (6 * N * seen >= a.flops) if a.flops else (time.time() - t0 >= a.minutes * 60)
    while not done():
        ix = torch.randint(len(ids) - a.ctx - 1, (a.batch,), generator=g)
        x = torch.stack([ids[i:i + a.ctx + 1] for i in ix]).to(dev)
        logits = model(x[:, :-1]).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; seen += a.batch * a.ctx
        if step % a.eval_every == 0 or done():
            model.eval(); v = lm_bpb(model, tok, val, device=dev, log=quiet)[0]; model.train()
            print(f"step {step:>6}  val {v:.3f} bits per byte  ({time.time() - t0:.0f}s, {seen / len(ids):.2f} epochs)", flush=True)
            if v < best:
                best, best_state, stale = v, {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= a.patience:
                    print("validation stopped improving: stopping early"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    t, _ = lm_bpb(model, tok, (c / "test.txt").read_text(encoding="utf-8"), device=dev, log=quiet)
    if a.save:
        model.save_pretrained(a.save); tok.save_pretrained(a.save)
    row = {"model": "GPT-2 fine-tuned", "params": N, "train_mb": a.mb, "steps": step, "tokens_seen": seen, "epochs": round(seen / len(ids), 2), "flops_budget": a.flops or None,
           "train_flops": 6 * N * seen, "val_bpb": round(best, 4), "test_bpb": round(t, 4),
           "seconds": round(time.time() - t0), "seed": a.seed, "device": dev}
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\ntest: {t:.3f} bits per byte (best val {best:.3f}), {6 * N * seen:.2e} training FLOPs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
