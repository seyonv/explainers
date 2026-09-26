# Card: overfit-then-load.html · Lecture 9 · optimization loop: overfit a single batch + data loader lite · https://www.youtube.com/watch?v=l8pRSuU81PU&t=3402s
# Run from the course folder: python labs/overfit-then-load.py   (--steps N, default 50)
import time
import torch
from common import GPT, GPTConfig, DataLoaderLite, pick_device, sync, arg

device = pick_device()
print(f"using device: {device}")
steps = arg('--steps', 50)
B, T = 4, 32


def train(get_batch, label):
    torch.manual_seed(1337)
    model = GPT(GPTConfig())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"--- {label} ---")
    t0 = time.time()
    for i in range(steps):
        x, y = get_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if i < 3 or i % 10 == 9 or i == steps - 1:
            print(f"step {i}, loss: {loss.item():.4f}")
    sync(device)
    print(f"{label}: {(time.time() - t0) / steps * 1000:.0f} ms/step")


# 1) commit 7822fce: the same batch 50 times -> memorise it
loader = DataLoaderLite(B, T)
x0, y0 = loader.next_batch()
train(lambda: (x0, y0), "overfit one batch")

# 2) commit 631f7d6: a fresh batch every step
loader = DataLoaderLite(B, T, verbose=False)
train(loader.next_batch, "DataLoaderLite, new batch every step")
