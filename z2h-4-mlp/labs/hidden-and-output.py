# Card: hidden-and-output · Lecture 3 ch.5-6, 8 https://www.youtube.com/watch?v=TCH_1BHY58I&t=1115s
# Run from the course folder: python labs/hidden-and-output.py
import torch
from common import words, build_dataset

X, Y = build_dataset(words[:5])
g = torch.Generator().manual_seed(2147483647)  # for reproducibility
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

emb = C[X]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
print('emb.view(-1, 6) @ W1 :', tuple((emb.view(-1, 6) @ W1).shape), '+ b1', tuple(b1.shape),
      '-> broadcast as', tuple(b1.unsqueeze(0).shape), 'copied down 32 rows')
print('h.shape', tuple(h.shape), '| h range [%.4f, %.4f]' % (h.min().item(), h.max().item()))
print('fraction of |h| > 0.99:', round((h.abs() > 0.99).float().mean().item(), 3))
logits = h @ W2 + b2
print('logits.shape', tuple(logits.shape))
for name, p in zip(['C', 'W1', 'b1', 'W2', 'b2'], parameters):
    print(f'  {name:2s} {str(tuple(p.shape)):10s} {p.nelement():5d}')
print('total parameters:', sum(p.nelement() for p in parameters))
