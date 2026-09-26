# Card: squash-the-logits (z2h-5-batchnorm) · Lecture 4 "fixing the initial loss" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=259s
# Run from the course folder: python labs/squash-the-logits.py [--quick: 20k steps instead of 200k] [--plot: save loss curves PNG]
import os, torch
import torch.nn.functional as F
from common import mlp1, mlp1_forward, train_mlp1, split_loss, steps_from_argv, flag, Xtr, Ytr, g, HERE, Timer

def step0_loss(P):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)  # the first minibatch the loop would draw
    _, _, logits = mlp1_forward(P, Xtr[ix])
    return F.cross_entropy(logits, Ytr[ix]).item()

# 1. step-0 loss as the output layer gets humbler (b2 *= 0, W2 *= scale)
for scale in [1.0, 0.1, 0.01, 0.0]:
    P = mlp1('original')
    with torch.no_grad():
        P['W2'] *= scale
        if scale != 1.0:
            P['b2'] *= 0
    print(f'W2 * {scale:<4}{"" if scale == 1.0 else ", b2 * 0"}: step-0 loss {step0_loss(P):.4f}')

# 2. full training, before and after the fix
max_steps = steps_from_argv()
curves = {}
for stage in ['original', 'fix-logits']:
    P = mlp1(stage)
    with Timer() as t:
        lossi = train_mlp1(P, max_steps, verbose=False)
    curves[stage] = lossi
    first = [10**l for l in lossi[:1000]]
    print(f'{stage:10s}: mean loss steps 0-999 {sum(first)/1000:.4f} | '
          f'train {split_loss(P, "train"):.4f} val {split_loss(P, "val"):.4f}  ({max_steps} steps, {t.s:.0f}s)')

if flag('--plot'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage, lossi in curves.items():
        plt.plot(lossi, label=stage, alpha=0.6)
    plt.ylabel('log10 loss'); plt.xlabel('step'); plt.legend()
    plt.savefig(os.path.join(HERE, 'squash-the-logits.png'), dpi=120)
    print('saved labs/squash-the-logits.png')
