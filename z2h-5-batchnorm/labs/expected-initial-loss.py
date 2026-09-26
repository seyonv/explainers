# Card: expected-initial-loss (z2h-5-batchnorm) · Lecture 4 "fixing the initial loss" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=259s
# Run from the course folder: python labs/expected-initial-loss.py
import torch
import torch.nn.functional as F
from common import mlp1, mlp1_forward, Xtr, Ytr, vocab_size, g

# 1. what a know-nothing model should score: uniform over 27 characters
print(f'expected initial loss -ln(1/27) = {-torch.tensor(1/27.).log().item():.4f}')

# 2. the 4-class toy: loss of the label-0 example for different logits
for name, logits in [('all zeros (uniform)', torch.tensor([0.0, 0.0, 0.0, 0.0])),
                     ('confident and right', torch.tensor([5.0, 0.0, 0.0, 0.0])),
                     ('confident and wrong', torch.tensor([0.0, 0.0, 5.0, 0.0]))]:
    loss = F.cross_entropy(logits.view(1, 4), torch.tensor([0]))
    print(f'4-class toy, {name:20s}: logits {[round(v, 2) for v in logits.tolist()]} -> loss {loss.item():.4f}')
gt = torch.Generator().manual_seed(2147483647)
for scale in [1, 10, 100]:
    logits = torch.randn(10000, 4, generator=gt) * scale  # 10,000 random 4-class examples
    print(f'4-class toy, randn * {scale:<3d}: mean loss over 10k draws {F.cross_entropy(logits, torch.zeros(10000, dtype=torch.long)).item():.4f}')
print(f'4-class uniform = ln 4 = {torch.tensor(4.).log().item():.4f}')

# 3. the lecture's starter MLP (all weights plain randn): the loss at step 0
P = mlp1('original')
ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]
_, h, logits = mlp1_forward(P, Xb)
loss = F.cross_entropy(logits, Yb)
print(f'starter MLP, step-0 loss: {loss.item():.4f}')
print(f'  logits of example 0 range from {logits[0].min().item():.1f} to {logits[0].max().item():.1f}')
p = F.softmax(logits, dim=1)[torch.arange(32), Yb]
print(f'  probability given to the correct next char: median {p.median().item():.2e} (uniform would be {1/27:.4f})')
