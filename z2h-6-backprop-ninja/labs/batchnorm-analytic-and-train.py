# Card: batchnorm-analytic-and-train · Lecture 5 ch.6-7 exercises 3-4 https://www.youtube.com/watch?v=q8SA3rM6ckI&t=5797s
# Run from the course folder: python labs/batchnorm-analytic-and-train.py [--quick]   (full: 200k steps, about 3 min; --quick: 20k steps)
import time
import torch
import torch.nn.functional as F
from common import Xtr, Ytr, Xdev, Ydev, block_size, itos, chunked_forward, cmp, init_params, flag

# ---- Exercise 3: BatchNorm backward in one line ----
d = chunked_forward()
n, hprebn, hpreact, bngain, bnbias, bnvar_inv, bnraw = (
    d[k] for k in ['n', 'hprebn', 'hpreact', 'bngain', 'bnbias', 'bnvar_inv', 'bnraw'])
hpreact_fast = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias
print('forward max diff:', (hpreact_fast - hpreact).abs().max().item())
dhpreact = hpreact.grad
dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
cmp('hprebn', dhprebn, hprebn)

# ---- Exercise 4: train the MLP with our own backward pass ----
def manual_backward(Xb, Yb, emb, embcat, bnraw, bnvar_inv, h, logits, parameters):
    C, W1, b1, W2, b2, bngain, bnbias = parameters
    n = Xb.shape[0]
    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n
    dh = dlogits @ W2.T                       # 2nd layer
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    dhpreact = (1.0 - h**2) * dh              # tanh
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)   # batchnorm
    dbnbias = dhpreact.sum(0, keepdim=True)
    dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
    dembcat = dhprebn @ W1.T                  # 1st layer
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    demb = dembcat.view(emb.shape)            # embedding
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            ix = Xb[k, j]
            dC[ix] += demb[k, j]
    return [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

def forward(Xb, parameters):
    C, W1, b1, W2, b2, bngain, bnbias = parameters
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1
    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=True)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    return emb, embcat, bnraw, bnvar_inv, h, logits

g, parameters = init_params(n_embd=10, n_hidden=200)
print('parameters:', sum(p.nelement() for p in parameters))
max_steps = 20000 if flag('--quick') else 200000
batch_size = 32

# first, one batch checked against loss.backward() (its own generator, so g's stream matches the notebook)
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=torch.Generator().manual_seed(1))
Xb, Yb = Xtr[ix], Ytr[ix]
emb, embcat, bnraw, bnvar_inv, h, logits = forward(Xb, parameters)
F.cross_entropy(logits, Yb).backward()
grads = manual_backward(Xb, Yb, emb, embcat, bnraw, bnvar_inv, h, logits.detach(), [p.detach() for p in parameters])
for p, grad in zip(parameters, grads):
    cmp(str(tuple(p.shape)), grad, p)
for p in parameters:
    p.grad = None

t0 = time.time()
with torch.no_grad():                         # no autograd graph at all from here on
    for i in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        emb, embcat, bnraw, bnvar_inv, h, logits = forward(Xb, parameters)
        loss = F.cross_entropy(logits, Yb)
        grads = manual_backward(Xb, Yb, emb, embcat, bnraw, bnvar_inv, h, logits, parameters)
        lr = 0.1 if i < max_steps // 2 else 0.01 # step learning rate decay (the notebook: at 100,000)
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad
        if i % (max_steps // 20) == 0:
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
print(f'trained {max_steps} steps in {time.time() - t0:.0f} s')

C, W1, b1, W2, b2, bngain, bnbias = parameters
with torch.no_grad():                         # calibrate the batch norm at the end of training
    hpreact = C[Xtr].view(Xtr.shape[0], -1) @ W1 + b1
    bnmean = hpreact.mean(0, keepdim=True)
    bnvar = hpreact.var(0, keepdim=True, unbiased=True)

@torch.no_grad()
def split_loss(split, x, y):
    hpreact = C[x].view(x.shape[0], -1) @ W1 + b1
    hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
    logits = torch.tanh(hpreact) @ W2 + b2
    print(split, f'{F.cross_entropy(logits, y).item():.4f}')

split_loss('train', Xtr, Ytr)
split_loss('val', Xdev, Ydev)

g = torch.Generator().manual_seed(2147483647 + 10)
samples = []
with torch.no_grad():
    for _ in range(20):
        out, context = [], [0] * block_size
        while True:
            hpreact = C[torch.tensor([context])].view(1, -1) @ W1 + b1
            hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
            probs = F.softmax(torch.tanh(hpreact) @ W2 + b2, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        samples.append(''.join(itos[i] for i in out))
print('samples:', ' '.join(samples))
