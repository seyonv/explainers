# Card: batchnorm-inference (z2h-5-batchnorm) · Lecture 4 "batch normalization" / "summary" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=2440s
# Run from the course folder: python labs/batchnorm-inference.py [--quick: 20k steps]
import warnings, torch
import torch.nn.functional as F
warnings.filterwarnings('ignore', message='std\\(\\): degrees of freedom')  # section 3 does this on purpose
from common import mlp1, mlp1_forward, train_mlp1, split_loss, calibrate_bn, steps_from_argv, Xtr, Ytr, Timer

P = mlp1('batchnorm')
print('trained by backprop (requires_grad):', [n for n in P if P[n].requires_grad])
print('buffers (EMA, no grad):            ', [n for n in P if not P[n].requires_grad])
max_steps = steps_from_argv()
with Timer() as t:
    train_mlp1(P, max_steps, verbose=False)
print(f'trained {max_steps} steps in {t.s:.0f}s')

# 1. option 1: calibrate after training; option 2: the running estimates kept during training
bnmean, bnstd = calibrate_bn(P)
print(f'calibrated vs running mean: max |diff| {(bnmean - P["bnmean_running"]).abs().max().item():.4f} '
      f'(mean values span {bnmean.min().item():.2f}..{bnmean.max().item():.2f})')
print(f'calibrated vs running std : max |diff| {(bnstd - P["bnstd_running"]).abs().max().item():.4f} '
      f'(std values span {bnstd.min().item():.2f}..{bnstd.max().item():.2f})')
for name, st in [('calibrated', (bnmean, bnstd)), ('running', None)]:
    print(f'{name:10s} stats: train {split_loss(P, "train", st):.4f} val {split_loss(P, "val", st):.4f}')

# 2. batch statistics couple the examples: one fixed example, five different batchmates
gj = torch.Generator().manual_seed(42)
x0 = Xtr[:1]
with torch.no_grad():
    outs = []
    for _ in range(5):
        batch = torch.cat([x0, Xtr[torch.randint(0, Xtr.shape[0], (31,), generator=gj)]])
        _, _, logits = mlp1_forward(dict(P), batch)  # batch stats; dict(P) so the buffers in P stay untouched
        outs.append(F.softmax(logits[0], 0)[Ytr[0]].item())
print(f'p(correct next char) for training example 0 in 5 batches: {[round(o, 4) for o in outs]}')

# 3. a single example: batch statistics fail, running statistics work
with torch.no_grad():
    _, _, lg = mlp1_forward(dict(P), x0)
    print(f'one example with batch stats : logits contain nan? {torch.isnan(lg).any().item()} (std of 1 value is nan)')
    _, _, lg = mlp1_forward(P, x0, (P['bnmean_running'], P['bnstd_running']))
    print(f'one example with running stats: p(correct) = {F.softmax(lg[0], 0)[Ytr[0]].item():.4f}')

# 4. why momentum 0.001, not PyTorch's 0.1, at batch 32: EMA of 10,000 batch means on the trained net
gm = torch.Generator().manual_seed(42)
with torch.no_grad():
    for momentum in [0.1, 0.001]:
        run = torch.zeros_like(bnmean)
        for _ in range(10000):
            ix = torch.randint(0, Xtr.shape[0], (32,), generator=gm)
            hpreact = P['C'][Xtr[ix]].view(32, -1) @ P['W1']
            run = (1 - momentum) * run + momentum * hpreact.mean(0, keepdim=True)
        print(f'momentum {momentum:<5}: running mean vs calibrated, mean |diff| {(run - bnmean).abs().mean().item():.4f}')
