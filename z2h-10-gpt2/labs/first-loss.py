# Card: first-loss.html · Lecture 9 · let's train: data batches (B,T) → logits (B,T,C) + cross entropy loss · https://www.youtube.com/watch?v=l8pRSuU81PU&t=2750s
# Run from the course folder: python labs/first-loss.py
import math
import tiktoken
import torch
from common import GPT, GPTConfig, load_text, pick_device

device = pick_device()
print(f"using device: {device}")
enc = tiktoken.get_encoding('gpt2')

# play.ipynb: a 4x6 batch from the first 1,000 characters
text = load_text()
data = text[:1000]
print(repr(data[:100]))
tokens = enc.encode(data)
print("tokens[:24]:", tokens[:24])
buf = torch.tensor(tokens[:24 + 1])
x = buf[:-1].view(4, 6)
y = buf[1:].view(4, 6)
print("x =", x.tolist())
print("y =", y.tolist())

# commit 92b5bf9 / 41078d1: B=4, T=32 and the loss at init
B, T = 4, 32
buf = torch.tensor(tokens[:B * T + 1])
x = buf[:-1].view(B, T).to(device)
y = buf[1:].view(B, T).to(device)
torch.manual_seed(1337)
model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x, y)
print("logits.shape:", tuple(logits.shape))
print("flattened for cross_entropy:", tuple(logits.view(-1, logits.size(-1)).shape), tuple(y.view(-1).shape))
print(f"expected -ln(1/50257) = {-math.log(1 / 50257):.4f}")
print(f"loss at init: {loss.item():.4f}")
probs = torch.softmax(logits.float(), dim=-1)
print(f"prob of the right token at init: mean {probs.gather(-1, y.unsqueeze(-1)).mean().item():.2e} (uniform 1/50257 = {1 / 50257:.2e})")
