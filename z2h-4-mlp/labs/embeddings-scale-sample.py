# Card: embeddings-scale-sample · Lecture 3 ch.15-18 https://www.youtube.com/watch?v=TCH_1BHY58I&t=3927s
# Run from the course folder: python labs/embeddings-scale-sample.py [--quick] [--plot]
import sys
import torch
import torch.nn.functional as F
from common import HERE, itos, splits, init_params, split_loss, train

quick = '--quick' in sys.argv
(Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = splits()

# 1) look at the 2-D embeddings of the 300-hidden model from the last card
g, parameters = init_params(n_embd=2, n_hidden=300)
for steps, lr in [(30000, 0.1), (20000, 0.01)]:
    train(parameters, Xtr, Ytr, steps, lambda i: lr, g)
C = parameters[0].data
print(f'2-D model (10,281 params): train {split_loss(parameters, Xtr, Ytr):.4f} | dev {split_loss(parameters, Xdev, Ydev):.4f}')
centre = C.mean(0)
far = (C - centre).norm(dim=1)
print('C, sorted by distance from the centre of all 27 points:')
for i in far.argsort(descending=True).tolist():
    print(f'  {itos[i]!r} ({C[i, 0].item():+.2f}, {C[i, 1].item():+.2f})  dist {far[i].item():.2f}')
d = torch.cdist(C, C)
print('3 nearest neighbours of each vowel:')
for v in 'aeiou':
    i = ord(v) - 96
    nn = d[i].argsort()[1:4].tolist()
    print(f'  {v}: ' + ', '.join(f'{itos[j]} ({d[i, j].item():.2f})' for j in nn))
if '--plot' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0], C[:, 1], s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha='center', va='center', color='white')
    plt.grid('minor')
    plt.savefig(f'{HERE}/embeddings-2d.png', dpi=100)
    print('saved labs/embeddings-2d.png')

# 2) scale up: 10-D embeddings, 200 hidden units, 200k steps (lr 0.1 then 0.01)
g, parameters = init_params(n_embd=10, n_hidden=200)
print(f'\n10-D model: {sum(p.nelement() for p in parameters)} params')
steps = 20000 if quick else 200000
train(parameters, Xtr, Ytr, steps, lambda i: 0.1 if i < steps // 2 else 0.01, g, log_every=steps // 10)
print(f'{"quick " if quick else ""}{steps} steps: train {split_loss(parameters, Xtr, Ytr):.4f} | '
      f'dev {split_loss(parameters, Xdev, Ydev):.4f}')

# 3) sample from the model
C, W1, b1, W2, b2 = parameters
block_size = 3
g = torch.Generator().manual_seed(2147483647 + 10)
with torch.no_grad():
    for _ in range(20):
        out = []
        context = [0] * block_size  # initialize with all ...
        while True:
            emb = C[torch.tensor([context])]  # (1, block_size, d)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out))
