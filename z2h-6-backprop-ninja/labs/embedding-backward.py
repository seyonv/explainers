# Card: embedding-backward · Lecture 5 ch.3 exercise 1 (1:18-1:26) https://www.youtube.com/watch?v=q8SA3rM6ckI&t=4712s
# Run from the course folder: python labs/embedding-backward.py
import torch
from common import chunked_forward, cmp, itos

d = chunked_forward()
Xb, C, emb, embcat, W1, b1, hprebn = (d[k] for k in ['Xb', 'C', 'emb', 'embcat', 'W1', 'b1', 'hprebn'])
dhprebn = hprebn.grad                          # take dhprebn as given (from tanh-bn-atomic)

# Linear layer 1: hprebn = embcat @ W1 + b1, same pattern as layer 2
dembcat = dhprebn @ W1.T                       # (32,64) @ (64,30) -> (32,30)
dW1 = embcat.T @ dhprebn                       # (30,32) @ (32,64) -> (30,64)
db1 = dhprebn.sum(0)                           # (64,)
cmp('embcat', dembcat, embcat)
cmp('W1', dW1, W1)
cmp('b1', db1, b1)
demb = dembcat.view(emb.shape)                 # undo the view: (32,30) -> (32,3,10)
cmp('emb', demb, emb)

# emb = C[Xb]: each of the 96 lookups sends its gradient row back to the row of C it came from
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]                   # += : a character used twice gets both gradients
cmp('C', dC, C)

dC_fast = torch.zeros_like(C).index_add_(0, Xb.view(-1), demb.view(-1, C.shape[1]))
cmp('C (index_add_)', dC_fast, C)
print('max |loop - index_add_| =', (dC - dC_fast).abs().max().item())

counts = torch.bincount(Xb.view(-1), minlength=C.shape[0])
print(f'Xb holds {Xb.nelement()} lookups of {int((counts > 0).sum())} distinct characters;'
      f' {int((counts == 0).sum())} rows of dC stay zero')
top = counts.argmax().item()
print(f"most used: '{itos[top]}' x{counts[top].item()} -> dC[{top}] is the sum of {counts[top].item()} rows of demb")
