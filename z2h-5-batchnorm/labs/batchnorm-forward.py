# Card: batchnorm-forward (z2h-5-batchnorm) · Lecture 4 "batch normalization" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=2440s
# Run from the course folder: python labs/batchnorm-forward.py [--quick: 20k steps]
import torch
import torch.nn.functional as F
from common import mlp1, mlp1_forward, params_of, train_mlp1, split_loss, steps_from_argv, Xtr, Ytr, g, Timer

P = mlp1('batchnorm')
print(f'parameters: {sum(p.nelement() for p in params_of(P))} (the Kaiming MLP had 11897: -200 for b1, +400 for bngain/bnbias)')

# 1. one minibatch: hpreact before and after standardising each of the 200 columns over the batch (dim 0)
state = g.get_state()
ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]
emb = P['C'][Xb]
hpreact = emb.view(32, -1) @ P['W1']
bnmeani = hpreact.mean(0, keepdim=True)
bnstdi = hpreact.std(0, keepdim=True)
hbn = P['bngain'] * (hpreact - bnmeani) / bnstdi + P['bnbias']
cm, cs = hpreact.mean(0), hpreact.std(0)
print(f'before BN: column means {cm.min().item():+.2f}..{cm.max().item():+.2f}, column stds {cs.min().item():.2f}..{cs.max().item():.2f}')
cm, cs = hbn.mean(0), hbn.std(0)
print(f'after  BN: column means {cm.abs().max().item():.1e} (max |.|), column stds {cs.min().item():.4f}..{cs.max().item():.4f}')
logits = torch.tanh(hbn) @ P['W2'] + P['b2']
print(f'step-0 loss: {F.cross_entropy(logits, Yb).item():.4f}')
g.set_state(state)

# 2. a bias before BN is useless: add b1 back and look at its gradient
b1 = torch.randn(200, generator=torch.Generator().manual_seed(1)).requires_grad_()
h1 = emb.view(32, -1) @ P['W1'] + b1
h1 = P['bngain'] * (h1 - h1.mean(0, keepdim=True)) / h1.std(0, keepdim=True) + P['bnbias']
F.cross_entropy(torch.tanh(h1) @ P['W2'] + P['b2'], Yb).backward()
print(f'b1.grad max |.|: {b1.grad.abs().max().item():.1e}  (zero up to float rounding: the mean subtraction removes b1)')
for p in params_of(P):
    p.grad = None

# 3. train the BatchNorm MLP (lr 0.1 -> 0.01), evaluate with the running statistics
max_steps = steps_from_argv()
with Timer() as t:
    train_mlp1(P, max_steps, verbose=False)
print(f'batchnorm ({max_steps} steps, {t.s:.0f}s): train {split_loss(P, "train"):.4f} val {split_loss(P, "val"):.4f}')
print(f'learned bngain: mean {P["bngain"].mean().item():.3f}, range {P["bngain"].min().item():.2f}..{P["bngain"].max().item():.2f}; '
      f'bnbias: range {P["bnbias"].min().item():+.2f}..{P["bnbias"].max().item():+.2f}')
