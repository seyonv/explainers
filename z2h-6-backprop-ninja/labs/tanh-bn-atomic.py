# Card: tanh-bn-atomic · Lecture 5 ch.3 exercise 1 (0:53-1:18) https://www.youtube.com/watch?v=q8SA3rM6ckI&t=3197s
# Run from the course folder: python labs/tanh-bn-atomic.py
import torch
from common import chunked_forward, cmp

d = chunked_forward()
n = d['n']
h, hpreact, bngain, bnbias, bnraw = (d[k] for k in ['h', 'hpreact', 'bngain', 'bnbias', 'bnraw'])
bnvar_inv, bnvar, bndiff2, bndiff, bnmeani, hprebn = (
    d[k] for k in ['bnvar_inv', 'bnvar', 'bndiff2', 'bndiff', 'bnmeani', 'hprebn'])
dh = h.grad                                   # take dh as given (from matmul-backward-by-shapes)

dhpreact = (1.0 - h**2) * dh                  # tanh' = 1 - tanh**2, written with the output h
cmp('hpreact', dhpreact, hpreact)
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)   # bngain (1,64) was broadcast over 32 rows: sum
dbnraw = bngain * dhpreact
dbnbias = dhpreact.sum(0, keepdim=True)
cmp('bngain', dbngain, bngain)
cmp('bnraw', dbnraw, bnraw)
cmp('bnbias', dbnbias, bnbias)
dbndiff = bnvar_inv * dbnraw                  # branch 1 of 2: bndiff -> bnraw
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
cmp('bndiff (1 of 2)', dbndiff, bndiff)       # False: the bndiff2 branch is still missing
dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv   # power rule on (bnvar + eps)**-0.5
cmp('bnvar', dbnvar, bnvar)
dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar   # sum forward -> broadcast backward
cmp('bndiff2', dbndiff2, bndiff2)
dbndiff += (2*bndiff) * dbndiff2              # branch 2 of 2
cmp('bndiff', dbndiff, bndiff)
dhprebn = dbndiff.clone()                     # branch 1 of 2: hprebn -> bndiff
dbnmeani = (-dbndiff).sum(0)
cmp('bnmeani', dbnmeani, bnmeani)
cmp('hprebn (1 of 2)', dhprebn, hprebn)
dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)   # branch 2 of 2: hprebn -> bnmeani
cmp('hprebn', dhprebn, hprebn)
print('shapes: dbngain', tuple(dbngain.shape), 'dbnmeani', tuple(dbnmeani.shape), 'dhprebn', tuple(dhprebn.shape))
