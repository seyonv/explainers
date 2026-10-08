"""The blind test: can friends tell your model's forecast discussions from real ones?

  uv run --python 3.12 --with torch python tiny_gpt.py --size 10M --mb 30 --tokens 2e8 --save best.pt
  uv run --python 3.12 --with torch python sample.py best.pt --pairs 10

Writes quiz.txt (numbered excerpts, real and generated, shuffled) and answers.txt. Real excerpts come from the
test file, so neither your model nor GPT-2 trained on them. Each excerpt is cut to the same length.
"""
import argparse, random, re, sys
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_gpt import SIZES, TinyGPT

HERE = Path(__file__).resolve().parent
PROMPT = "Area Forecast Discussion\nNational Weather Service "


@torch.no_grad()
def generate(model, prompt: bytes, n: int, temp: float, gen: torch.Generator):
    x = torch.tensor(list(prompt), dtype=torch.long).unsqueeze(0)
    for _ in range(n):
        logits = model(x[:, -model.ctx:])[0, -1] / temp
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=gen)
        x = torch.cat([x, nxt.unsqueeze(0)], 1)
    return bytes(x[0].tolist()).decode("utf-8", errors="replace")


YEAR = re.compile(r"\b(19|20)\d\d\b")
HEADER = re.compile(r"^(Area Forecast Discussion|National Weather Service)\b")


def excerpt(text, length):
    """The discussion without its title, office and issue-time lines, and without any line that names a year:
    real excerpts are from 2025-26 and the model learned on 2018-24, so a year would give the answer away. A
    generated excerpt can run into a second discussion, so header lines are dropped wherever they appear."""
    lines = text.split("\n")[3:]
    body = ("\n".join(l for l in lines if not YEAR.search(l) and not HEADER.match(l)).strip("\n") + "\n")[:length]
    return body[: body.rfind("\n") + 1 or length].rstrip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--length", type=int, default=600, help="bytes per excerpt")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=HERE / "data")
    a = ap.parse_args(argv)
    ck = torch.load(a.model, map_location="cpu")
    model = TinyGPT(*SIZES[ck["size"]]); model.load_state_dict(ck["state"]); model.eval()
    gen, rng = torch.Generator().manual_seed(a.seed), random.Random(a.seed)
    real = (Path(a.data) / "corpus" / "test.txt").read_text(encoding="utf-8").split("Area Forecast Discussion\n")[1:]
    items = [("real", excerpt("Area Forecast Discussion\n" + t, a.length)) for t in rng.sample(real, a.pairs)]
    items += [("model", excerpt(generate(model, PROMPT.encode(), a.length + 400, a.temp, gen), a.length)) for _ in range(a.pairs)]
    rng.shuffle(items)
    Path("quiz.txt").write_text("".join(f"=== {i + 1} ===\n{t}\n" for i, (_, t) in enumerate(items)))
    Path("answers.txt").write_text("".join(f"{i + 1}: {k}\n" for i, (k, _) in enumerate(items)))
    print(f"wrote quiz.txt ({len(items)} excerpts) and answers.txt. Share only the quiz.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
