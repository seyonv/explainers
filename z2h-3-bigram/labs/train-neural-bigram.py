# Card: train-neural-bigram.html · Lecture 2 · vectorized loss, backward/update, notes, sampling · https://www.youtube.com/watch?v=PaCmpygFfXo&t=5749s
# Run from the course folder: python labs/train-neural-bigram.py   (optional: --iters 200)
import sys
import torch
import torch.nn.functional as F
from common import setup

iters = int(sys.argv[sys.argv.index('--iters') + 1]) if '--iters' in sys.argv else 100
words, stoi, itos, N = setup()

def dataset(ws):
    xs, ys = [], []
    for w in ws:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)

def train(xs, ys, steps, lr, reg, show):
    num = xs.nelement()
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)
    for k in range(steps):
        xenc = F.one_hot(xs, num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(num), ys].log().mean() + reg * (W**2).mean()
        if k in show:
            print(f'  step {k}: loss {loss.item():.4f}')
        W.grad = None
        loss.backward()
        W.data += -lr * W.grad
    return W

print('emma only, lr 0.1, no reg:')
xs, ys = dataset(words[:1])
train(xs, ys, 3, 0.1, 0.0, {0, 1, 2})

xs, ys = dataset(words)
print('number of examples:', xs.nelement())
show = {0, 1, 9, 49, iters - 1}
print(f'all words, lr 50, no reg, {iters} steps:')
train(xs, ys, iters, 50, 0.0, show)
print(f'all words, lr 50, reg 0.01*(W**2).mean(), {iters} steps:')
W = train(xs, ys, iters, 50, 0.01, show)

# note 1: one_hot(i) @ W is just row i of W
i = 5
print('one_hot(5) @ W == W[5]:', torch.allclose(F.one_hot(torch.tensor(i), 27).float() @ W, W[i]))
# W is close to log-counts: compare its row-softmax to the smoothed counting P
P = (N + 1).float()
P /= P.sum(1, keepdim=True)
Pnet = torch.softmax(W.detach(), dim=1)
for a, b in [('.', 'a'), ('e', '.'), ('n', '.'), ('q', 'u')]:
    print(f'P[{a}{b}]  net {Pnet[stoi[a], stoi[b]]:.4f}  counting(N+1) {P[stoi[a], stoi[b]]:.4f}')

g = torch.Generator().manual_seed(2147483647)
net_names = []
for i in range(5):
    out, ix = [], 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    net_names.append(''.join(out))
print('neural net samples:', net_names)

g = torch.Generator().manual_seed(2147483647)
count_names = []
for i in range(5):
    out, ix = [], 0
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    count_names.append(''.join(out))
print('counting (N+1) samples:', count_names)
