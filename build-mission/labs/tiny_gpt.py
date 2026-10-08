"""A small GPT trained from scratch on your forecast discussions, scored in bits per byte on the test file.

It reads bytes, not BPE tokens: 256 symbols, so a 1M-parameter model isn't mostly embedding table. Swap in
your own chapter-4 model if you like; keep the data, the evaluation and the FLOP count the same.

  uv run --python 3.12 --with torch python tiny_gpt.py --size 1M --mb 1 --minutes 10
  uv run --python 3.12 --with torch python tiny_gpt.py --size 1M --mb 1 --steps 200     # a short sanity run

Each run appends one JSON line to results.jsonl: size, parameters, corpus MB, steps, tokens seen, training
FLOPs (6 x parameters x tokens), best validation BPB and test BPB. sweep.py reads that file.
"""
import argparse, json, math, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SIZES = {"1M": (4, 144, 4), "3M": (6, 208, 4), "10M": (8, 320, 8), "30M": (10, 512, 8)}   # layers, width, heads


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, layers, d, heads, ctx=256, vocab=256):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight            # tie input and output embeddings
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def bpb(model, data, dev, stride=128, batch=32):
    """Bits per byte over a whole byte tensor. Windows of ctx bytes start every `stride` bytes (the last one is
    pulled back to end at the text's end); each window scores only the bytes past where the previous one
    ended, so every byte is scored once, with at least ctx - stride bytes of context after the first window."""
    model.eval()
    ctx, n = model.ctx, len(data)
    starts = list(range(0, max(1, n - ctx), stride)) + [max(0, n - ctx)]
    jobs, prev_end = [], 1                               # byte 0 has no context; it is charged 8 bits below
    for s in starts:
        end = min(s + ctx, n)
        if end > prev_end:
            jobs.append((s, end - prev_end)); prev_end = end
    nll = 0.0
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        x = torch.stack([data[s:s + ctx] for s, _ in chunk]).to(dev)
        lp = F.log_softmax(model(x[:, :-1]).float(), -1).gather(2, x[:, 1:].unsqueeze(2)).squeeze(2)
        for row, (_, new) in zip(lp, chunk):
            nll -= row[-new:].sum().item()
    model.train()
    return (nll / math.log(2) + 8.0) / n


def load(path):
    return torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.uint8).long()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", choices=SIZES, default="1M")
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=10, help="wall-clock budget (ignored if --steps)")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--tokens", type=float, default=0, help="stop after this many training bytes (sweep.py uses 20 x parameters)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5, help="stop after this many evaluations without a better validation score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=HERE / "data")
    ap.add_argument("--out", default=HERE / "results.jsonl")
    ap.add_argument("--save", default="", help="save the best model here (sample.py reads it)")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    c = Path(a.data) / "corpus"
    train, val, test = load(c / f"train_{a.mb}MB.txt"), load(c / "val.txt")[:200_000], load(c / "test.txt")
    L, d, h = SIZES[a.size]
    model = TinyGPT(L, d, h).to(dev)
    N = params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    print(f"{a.size}: {N / 1e6:.2f}M parameters, {len(train) / 1e6:.1f} MB of training text, on {dev}")
    g = torch.Generator().manual_seed(a.seed)
    t0, step, seen, best, best_state, evals, stale = time.time(), 0, 0, float("inf"), None, 0, 0
    if a.tokens:
        a.steps = math.ceil(a.tokens / (a.batch * model.ctx))
    budget = a.steps or 10**9
    while step < budget and (a.steps or time.time() - t0 < a.minutes * 60):
        ix = torch.randint(len(train) - model.ctx - 1, (a.batch,), generator=g)
        x = torch.stack([train[i:i + model.ctx + 1] for i in ix]).to(dev)
        frac = (step / budget) if a.steps else (time.time() - t0) / (a.minutes * 60)
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))
        loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; seen += a.batch * model.ctx
        if step % 200 == 0 or step == budget:
            v = bpb(model, val, dev); evals += 1
            print(f"step {step:>6}  train loss {loss.item() / math.log(2):.3f} bits/byte  val {v:.3f}  ({time.time() - t0:.0f}s)")
            if v < best:                               # early stopping on validation, never on test
                best, best_state, stale = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= a.patience:
                    print("validation stopped improving: stopping early"); break
    if best_state is None or evals == 0:
        best = bpb(model, val, dev)
    else:
        model.load_state_dict(best_state)
    t = bpb(model, test, dev)
    if a.save:
        torch.save({"size": a.size, "state": model.state_dict()}, a.save)
    row = {"model": f"from scratch {a.size}", "size": a.size, "params": N, "train_mb": a.mb, "steps": step,
           "tokens_seen": seen, "epochs": round(seen / len(train), 2), "tokens_budget": a.tokens or None, "train_flops": 6 * N * seen, "val_bpb": round(best, 4), "test_bpb": round(t, 4),
           "seconds": round(time.time() - t0), "seed": a.seed, "device": dev}
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\ntest: {t:.3f} bits per byte (best val {best:.3f}), {6 * N * seen:.2e} training FLOPs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
