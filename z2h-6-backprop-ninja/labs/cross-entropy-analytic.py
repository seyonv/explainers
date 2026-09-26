# Card: cross-entropy-analytic · Lecture 5 ch.5 exercise 2 https://www.youtube.com/watch?v=q8SA3rM6ckI&t=5191s
# Run from the course folder: python labs/cross-entropy-analytic.py [--plot]   (--plot saves dlogits.png next to this file)
import os
import torch
import torch.nn.functional as F
from common import HERE, chunked_forward, cmp, flag

d = chunked_forward()
logits, Yb, n, loss = d['logits'], d['Yb'], d['n'], d['loss']
loss_fast = F.cross_entropy(logits, Yb)
print(loss_fast.item(), 'diff:', (loss_fast - loss).item())

# backward pass: softmax minus one-hot, divided by n
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
cmp('logits', dlogits, logits)

torch.set_printoptions(precision=4, linewidth=100)
print('F.softmax(logits, 1)[0] =', F.softmax(logits, 1)[0].detach())
print('dlogits[0] * n          =', (dlogits[0] * n).detach())
print(f'Yb[0] = {Yb[0].item()}: p = {F.softmax(logits, 1)[0, Yb[0]].item():.4f}, so that entry is p - 1 = {(dlogits[0, Yb[0]] * n).item():.4f}')
print('dlogits[0].sum() =', dlogits[0].sum().item())
print(f'max |row sum| over the batch = {dlogits.sum(1).abs().max().item():.2e}')

if flag('--plot'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(dlogits.detach(), cmap='gray')
    out = os.path.join(HERE, 'dlogits.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print('saved', out)
else:
    print(f'imshow(dlogits): the 32 darkest cells are the targets (min {dlogits.min().item():.4f}); '
          f'every other cell is small and positive (max {dlogits.max().item():.4f})')
    print('argmin per row == Yb for all rows:', bool((dlogits.argmin(1) == Yb).all()))
