# Card: chunked-forward-and-cmp · Lecture 5 ch.2 starter code https://www.youtube.com/watch?v=q8SA3rM6ckI&t=446s
# Run from the course folder: python labs/chunked-forward-and-cmp.py   [--all]   (--all: the whole Exercise 1 answer, 26 cmp lines)
import torch
import torch.nn.functional as F
from common import Xtr, Xdev, Xte, chunked_forward, cmp, flag

print('splits:', tuple(Xtr.shape), tuple(Xdev.shape), tuple(Xte.shape))
d = chunked_forward()
print('parameters:', sum(p.nelement() for p in d['parameters']))
print(f"loss: {d['loss'].item():.4f}")
for name in ['emb', 'embcat', 'hprebn', 'bnraw', 'h', 'logits', 'counts_sum', 'logprobs']:
    print(f'  {name:11s} {tuple(d[name].shape)}')

# cmp() on a correct gradient, a slightly-off one, and a wrong one
logprobs, Yb, n = d['logprobs'], d['Yb'], d['n']
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
cmp('logprobs', dlogprobs, logprobs)
cmp('logprobs+1e-9', dlogprobs + 1e-9, logprobs)
cmp('logprobs wrong', -dlogprobs, logprobs)

if flag('--all'):   # Exercise 1 solved (notebook cell 11): every intermediate, one by one
    globals().update(d)
    dlogprobs = torch.zeros_like(logprobs); dlogprobs[range(n), Yb] = -1.0/n
    dprobs = (1.0 / probs) * dlogprobs
    dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
    dcounts = counts_sum_inv * dprobs
    dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
    dcounts += torch.ones_like(counts) * dcounts_sum
    dnorm_logits = counts * dcounts
    dlogits = dnorm_logits.clone()
    dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
    dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
    dh = dlogits @ W2.T; dW2 = h.T @ dlogits; db2 = dlogits.sum(0)
    dhpreact = (1.0 - h**2) * dh
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True); dbnraw = bngain * dhpreact; dbnbias = dhpreact.sum(0, keepdim=True)
    dbndiff = bnvar_inv * dbnraw
    dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
    dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
    dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
    dbndiff += (2*bndiff) * dbndiff2
    dhprebn = dbndiff.clone()
    dbnmeani = (-dbndiff).sum(0)
    dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)
    dembcat = dhprebn @ W1.T; dW1 = embcat.T @ dhprebn; db1 = dhprebn.sum(0)
    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            dC[Xb[k, j]] += demb[k, j]
    names = ['logprobs', 'probs', 'counts_sum_inv', 'counts_sum', 'counts', 'norm_logits', 'logit_maxes',
             'logits', 'h', 'W2', 'b2', 'hpreact', 'bngain', 'bnbias', 'bnraw', 'bnvar_inv', 'bnvar',
             'bndiff2', 'bndiff', 'bnmeani', 'hprebn', 'embcat', 'W1', 'b1', 'emb', 'C']
    print('--- Exercise 1, all gradients ---')
    for s in names:
        cmp(s, globals()['d' + s], globals()[s])
    exact = sum(torch.all(globals()['d' + s] == globals()[s].grad).item() for s in names)
    print(f'{exact} of {len(names)} exact')
