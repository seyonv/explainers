# Card: train-and-script.html · Lecture 7 · training the bigram model / port our code to a script · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2093s
# Run from the course folder: python labs/train-and-script.py   (part 1 = notebook loop, part 2 = bigram.py)
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from common import Shakespeare, train

ds = Shakespeare()

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        return logits, F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx

# ---- part 1: the notebook loop (AdamW lr 1e-3, batch 32, block 8) ----
print("== part 1: notebook loop, AdamW lr 1e-3, batch_size 32 ==")
torch.manual_seed(1337)
ds.get_batch('train', 4, 8)            # the notebook drew one 4x8 batch here
torch.manual_seed(1337)
m = BigramLanguageModel(ds.vocab_size)
m.generate(torch.zeros((1, 1), dtype=torch.long), 100)   # ...and sampled 100 chars (uses the RNG)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
t0 = time.time()
for steps in range(10000):
    xb, yb = ds.get_batch('train', batch_size, 8)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps + 1 in (1, 100, 1000, 10000):
        print(f"after {steps+1:5d} steps: loss {loss.item():.4f}")
print(f"({time.time() - t0:.1f} s)")

# ---- part 2: bigram.py, the script version ----
print("== part 2: bigram.py (batch 32, block 8, lr 1e-2, 3000 iters, eval_iters 200) ==")
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
torch.manual_seed(1337)
model = BigramLanguageModel(ds.vocab_size).to(device)
t0 = time.time()
train(model, ds, batch_size, block_size, max_iters, eval_interval, learning_rate, eval_iters, device)
print(f"training took {time.time() - t0:.1f} s on {device}")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(ds.decode(model.generate(context, max_new_tokens=300)[0].tolist()))
