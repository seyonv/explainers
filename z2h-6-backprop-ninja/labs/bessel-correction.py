# Card: bessel-correction · Lecture 5 ch.4 Bessel's correction https://www.youtube.com/watch?v=q8SA3rM6ckI&t=3917s
# Run from the course folder: python labs/bessel-correction.py
import torch
from common import Xtr, init_params

# The data BatchNorm sees: hprebn = embcat @ W1 + b1 for the whole training set (the 4,137-param net at init)
g, (C, W1, b1, W2, b2, bngain, bnbias) = init_params()
with torch.no_grad():
    hprebn_all = C[Xtr].view(Xtr.shape[0], -1) @ W1 + b1     # (182625, 64)
x_all = hprebn_all[:, 0]                                        # one neuron
true_var = x_all.var(unbiased=False).item()                     # N = 182,625, so the 1/N vs 1/(N-1) gap is tiny
print(f'full-data variance of neuron 0: {true_var:.4f}  (N = {x_all.shape[0]:,})')

xb = x_all[torch.randint(0, x_all.shape[0], (32,), generator=g)]
print(f'one batch of 32:  var(unbiased=False) = {xb.var(unbiased=False).item():.4f}   var(unbiased=True) = {xb.var(unbiased=True).item():.4f}')

trials = 10000
ix = torch.randint(0, x_all.shape[0], (trials, 32), generator=g)
batches = x_all[ix]                                             # (10000, 32)
biased = batches.var(1, unbiased=False).mean().item()
unbiased = batches.var(1, unbiased=True).mean().item()
print(f'average over {trials:,} batches of 32:')
print(f'  1/n     (biased)  : {biased:.4f}  = {biased/true_var:.4f} x true   (expected 31/32 = {31/32:.4f})')
print(f'  1/(n-1) (unbiased): {unbiased:.4f}  = {unbiased/true_var:.4f} x true')

# PyTorch's BatchNorm1d normalizes with the biased variance but stores the unbiased one in running_var
bn = torch.nn.BatchNorm1d(1, momentum=1.0)                      # momentum 1.0: running_var = this batch's estimate
bn(xb.view(32, 1))
print(f'nn.BatchNorm1d on that batch: running_var = {bn.running_var.item():.4f}  (unbiased)')
print(f'  but the train-mode output uses the biased {xb.var(unbiased=False).item():.4f}: output std(unbiased=False) = '
      f'{bn(xb.view(32, 1)).std(unbiased=False).item():.4f}')
print(f'  (with the unbiased variance it would be {(xb.var(unbiased=False) / (xb.var(unbiased=True) + 1e-5)).sqrt().item():.4f})')
