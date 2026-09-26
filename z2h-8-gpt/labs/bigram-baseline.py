# Card: bigram-baseline.html · Lecture 7 · simplest baseline: bigram language model, loss, generation · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1331s
# Run from the course folder: python labs/bigram-baseline.py
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from common import Shakespeare

ds = Shakespeare()
vocab_size, decode = ds.vocab_size, ds.decode
torch.manual_seed(1337)
xb, yb = ds.get_batch('train', batch_size=4, block_size=8)   # the notebook's batch
print("xb =", xb.tolist())

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print("logits.shape:", logits.shape)
print("loss:", loss)
print(f"expected at uniform init: -ln(1/{vocab_size}) = {-math.log(1/vocab_size):.4f}")
print("params:", sum(p.numel() for p in m.parameters()), f"= {vocab_size}*{vocab_size}")

idx = torch.zeros((1, 1), dtype=torch.long)   # kick off with token 0 = '\n'
print("100 sampled characters from the untrained model:")
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
