# Card: heads-in-the-model.html · Lecture 7 · inserting a single self-attention block / multi-headed self-attention · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4751s
# Run from the course folder: python labs/heads-in-the-model.py   (trains two small models on CPU, ~1 min)
import torch
import torch.nn as nn
from torch.nn import functional as F
from common import Shakespeare, train, nparams

# hyperparameters at this point in the video
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3   # lowered from 1e-2: self-attention can't take very high learning rates
eval_iters = 200
n_embd = 32
ds = Shakespeare()
vocab_size = ds.vocab_size

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x) # (B,T,hs)
        return wei @ v    # (B,T,hs)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class BigramLanguageModel(nn.Module):

    def __init__(self, sa_heads):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = sa_heads
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T))
        x = self.sa_heads(x) # apply self-attention. (B,T,C)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        return logits, F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx

m = BigramLanguageModel(Head(n_embd))
print("register_buffer: 'tril' in state_dict:", any('tril' in k for k in m.state_dict()),
      "  'tril' in parameters:", any('tril' in n for n, _ in m.named_parameters()))
try:
    m(torch.zeros((1, block_size + 1), dtype=torch.long))
except IndexError as err:
    print(f"forward with T = {block_size+1} (no cropping): IndexError: {err}")

for name, make in [("one head, head_size 32", lambda: Head(n_embd)),
                   ("4 heads x 8 dims, concatenated", lambda: MultiHeadAttention(4, n_embd//4))]:
    torch.manual_seed(1337)
    model = BigramLanguageModel(make())
    print(f"== {name}: {nparams(model)} params ==")
    train(model, ds, batch_size, block_size, max_iters, eval_interval, learning_rate, eval_iters)
    context = torch.zeros((1, 1), dtype=torch.long)
    print(ds.decode(model.generate(context, max_new_tokens=200)[0].tolist()))
