# Card: sum-broadcast-duality · Lecture 5 ch.3 exercise 1 (0:22-0:41) https://www.youtube.com/watch?v=q8SA3rM6ckI&t=1349s
# Run from the course folder: python labs/sum-broadcast-duality.py
import torch
import torch.nn.functional as F
from common import chunked_forward, cmp

# The duality on a toy: b (2,1) is broadcast across 3 columns, then everything is summed.
a = torch.randn(2, 3, requires_grad=True); b = torch.randn(2, 1, requires_grad=True)
(a + b).sum().backward()                       # forward: broadcast b, then sum
print('broadcast forward -> sum backward: b.grad =', b.grad.flatten().tolist())   # each b used 3 times
x = torch.randn(2, 3, requires_grad=True)
s = x.sum(1, keepdim=True); (s * torch.tensor([[10.], [20.]])).sum().backward()  # forward: sum
print('sum forward -> broadcast backward: x.grad =', x.grad.tolist())

# In the net: logit_maxes (32,1) is broadcast in norm_logits = logits - logit_maxes
d = chunked_forward()
logits, logit_maxes, counts, counts_sum_inv, counts_sum = (
    d[k] for k in ['logits', 'logit_maxes', 'counts', 'counts_sum_inv', 'counts_sum'])
n, Yb, probs = d['n'], d['Yb'], d['probs']
# redo dnorm_logits quickly (see softmax-chain-backward.py)
dlogprobs = torch.zeros_like(d['logprobs']); dlogprobs[range(n), Yb] = -1.0/n
dprobs = (1.0 / probs) * dlogprobs
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
dcounts = counts_sum_inv * dprobs + torch.ones_like(counts) * (-counts_sum**-2) * dcounts_sum_inv
dnorm_logits = counts * dcounts

dlogits = dnorm_logits.clone()                           # branch 1: logits -> norm_logits
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)      # broadcast forward, so sum backward
cmp('logit_maxes', dlogit_maxes, logit_maxes)
print('  dlogit_maxes[:5] =', [f'{v:.2e}' for v in dlogit_maxes[:5, 0].tolist()])
print(f'  largest |dlogit_maxes| = {dlogit_maxes.abs().max().item():.2e}  (the shift cannot change the loss)')
cmp('logits (1 of 2)', dlogits, logits)
onehot = F.one_hot(logits.max(1).indices, num_classes=logits.shape[1])
print('  one_hot of the max positions: shape', tuple(onehot.shape), 'row sums', onehot.sum(1).unique().tolist())
dlogits += onehot * dlogit_maxes                        # branch 2: logits -> max -> logit_maxes
cmp('logits', dlogits, logits)
