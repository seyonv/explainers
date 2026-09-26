# Card: results.html · Lecture 9 · SECTION 4: results in the morning! GPT-2, GPT-3 repro · https://www.youtube.com/watch?v=l8pRSuU81PU&t=13385s
# Run from the course folder: python labs/train_gpt2.py --quick, then python labs/results.py   (--log PATH, --plot saves labs/results.png)
import os
import sys
from common import DATA, HERE, flag, arg

sz = "124M"
loss_baseline = {"124M": 3.2924}[sz]  # OpenAI GPT-2 124M on the FineWeb-Edu val shard
hella2_baseline = {"124M": 0.294463, "350M": 0.375224, "774M": 0.431986, "1558M": 0.488946}[sz]  # GPT-2
hella3_baseline = {"124M": 0.337, "350M": 0.436, "774M": 0.510, "1558M": 0.547}[sz]  # GPT-3

path = arg('--log', os.path.join(DATA, "log", "log.txt"))
if not os.path.exists(path):
    print(f"no log at {path}: run `python labs/train_gpt2.py --quick` first")
    sys.exit(1)

# play.ipynb's parser: "step stream value" lines, grouped by stream (train, val, hella)
streams = {}
with open(path, "r") as f:
    for line in f:
        step, stream, val = line.strip().split()
        streams.setdefault(stream, {})[int(step)] = float(val)
streams_xy = {k: list(zip(*sorted(v.items()))) for k, v in streams.items()}

xs, ys = streams_xy["train"]
print(f"log: {path} ({len(xs)} train steps)")
print(f"Min Train Loss: {min(ys):.4f} (first {ys[0]:.4f}, last {ys[-1]:.4f})")
xs, ys = streams_xy["val"]
print(f"Min Validation Loss: {min(ys):.4f} | OpenAI GPT-2 ({sz}) checkpoint val loss: {loss_baseline} | beaten: {min(ys) < loss_baseline}")
xs, ys = streams_xy["hella"]
print(f"Max Hellaswag eval: {max(ys):.4f} | GPT-2 ({sz}): {hella2_baseline} | GPT-3 ({sz}): {hella3_baseline}")

if flag('--plot'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.plot(*streams_xy["train"], label=f'nanogpt ({sz}) train loss')
    plt.plot(*streams_xy["val"], label=f'nanogpt ({sz}) val loss')
    plt.axhline(y=loss_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint val loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.legend()
    plt.title("Loss")
    plt.subplot(122)
    plt.plot(*streams_xy["hella"], label=f"nanogpt ({sz})")
    plt.axhline(y=hella2_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint")
    plt.axhline(y=hella3_baseline, color='g', linestyle='--', label=f"OpenAI GPT-3 ({sz}) checkpoint")
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("HellaSwag eval")
    plt.savefig(os.path.join(HERE, "results.png"), dpi=80, bbox_inches='tight')
    print("saved labs/results.png")

# how long the real run takes
tokens = 19073 * 2**19
print(f"the real run: 19,073 steps x 2**19 = {tokens / 1e9:.2f}B tokens")
for name, tok_s in (("8xA100 at ~1.5M tok/s", 1.5e6), ("M3 at ~1,300 tok/s", 1300)):
    secs = tokens / tok_s
    print(f"  {name}: {secs / 3600:.1f} h = {secs / 86400:.1f} days")
