# Shared helpers for the z2h-8-gpt labs: tiny shakespeare, the char tokenizer, get_batch, estimate_loss, and gpt.py's modules.
# Imported by the other labs; run a lab from the course folder, e.g. `python labs/bigram-baseline.py`.
import os
import sys
import time
import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def flag(name):
    return name in sys.argv[1:]


def load_text():
    path = os.path.join(DATA, 'input.txt')
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        print(f'downloading input.txt to {path}')
        urllib.request.urlretrieve(URL, path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class Shakespeare:
    """text -> stoi/itos -> 90/10 split, exactly as in the lecture."""

    def __init__(self):
        self.text = load_text()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split, batch_size, block_size, device='cpu'):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, ds, batch_size, block_size, eval_iters, device='cpu'):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = ds.get_batch(split, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(model, ds, batch_size, block_size, max_iters, eval_interval, learning_rate,
          eval_iters=200, device='cpu'):
    """The bigram.py / gpt.py loop. Prints the step lines and returns the last losses."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    t0 = time.time()
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, ds, batch_size, block_size, eval_iters, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                  f"   ({time.time() - t0:.1f} s)", flush=True)
        xb, yb = ds.get_batch('train', batch_size, block_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return losses


def nparams(model):
    return sum(p.numel() for p in model.parameters())


# ---- gpt.py's modules, with the globals turned into constructor arguments ----

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.0, proj=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) if proj else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, mult=4, dropout=0.0, proj=True):
        super().__init__()
        if proj:
            self.net = nn.Sequential(nn.Linear(n_embd, mult * n_embd), nn.ReLU(),
                                     nn.Linear(mult * n_embd, n_embd), nn.Dropout(dropout))
        else:  # the first version in the video: Linear + ReLU, no projection
            self.net = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout=0.0, residual=True, layernorm=True):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout, proj=residual)
        self.ffwd = FeedFoward(n_embd, dropout=dropout, proj=residual)
        self.ln1 = nn.LayerNorm(n_embd) if layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(n_embd) if layernorm else nn.Identity()
        self.residual = residual

    def forward(self, x):
        if self.residual:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.sa(x)
            x = self.ffwd(x)
        return x


class GPTLanguageModel(nn.Module):
    """gpt.py's model. init_weights=True is the post-video std-0.02 init that gpt.py ships with."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0,
                 residual=True, layernorm=True, init_weights=True):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout, residual, layernorm)
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) if layernorm else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)
        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        return logits, F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
