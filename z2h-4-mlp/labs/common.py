# Shared helpers for the z2h-4-mlp labs (makemore part 2: the MLP).
# Lecture: https://www.youtube.com/watch?v=TCH_1BHY58I · imported by the other labs, not run directly.
import os
import random
import urllib.request

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES_URL = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'


def load_words():
    path = os.path.join(HERE, 'data', 'names.txt')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'downloading {NAMES_URL}')
        urllib.request.urlretrieve(NAMES_URL, path)
    return open(path, 'r').read().splitlines()


words = load_words()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}


def build_dataset(words, block_size=3):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append
    return torch.tensor(X), torch.tensor(Y)


def splits(block_size=3):
    """80/10/10 split of the shuffled words (random.seed(42), as in the lecture)."""
    ws = list(words)  # shuffle a copy, so calling this twice gives the same split
    random.seed(42)
    random.shuffle(ws)
    n1 = int(0.8 * len(ws))
    n2 = int(0.9 * len(ws))
    return (build_dataset(ws[:n1], block_size),
            build_dataset(ws[n1:n2], block_size),
            build_dataset(ws[n2:], block_size))


def init_params(n_embd=2, n_hidden=100, block_size=3, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((27, n_embd), generator=g)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
    b1 = torch.randn(n_hidden, generator=g)
    W2 = torch.randn((n_hidden, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    return g, parameters


def forward(parameters, X):
    C, W1, b1, W2, b2 = parameters
    emb = C[X]
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    return h @ W2 + b2


@torch.no_grad()
def split_loss(parameters, X, Y):
    return torch.nn.functional.cross_entropy(forward(parameters, X), Y).item()


def train(parameters, X, Y, steps, lr_fn, g, batch_size=32, log_every=0):
    """Minibatch SGD. lr_fn(i) gives the learning rate at step i. Returns the per-step minibatch losses."""
    lossi = []
    for i in range(steps):
        ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
        logits = forward(parameters, X[ix])
        loss = torch.nn.functional.cross_entropy(logits, Y[ix])
        for p in parameters:
            p.grad = None
        loss.backward()
        lr = lr_fn(i)
        for p in parameters:
            p.data += -lr * p.grad
        lossi.append(loss.item())
        if log_every and (i % log_every == 0 or i == steps - 1):
            print(f'  step {i:7d}/{steps}  minibatch loss {loss.item():.4f}')
    return lossi
