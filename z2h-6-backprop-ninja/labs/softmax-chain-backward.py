# Card: softmax-chain-backward · Lecture 5 ch.3 exercise 1 (0:13-0:33) https://www.youtube.com/watch?v=q8SA3rM6ckI&t=781s
# Run from the course folder: python labs/softmax-chain-backward.py
import torch
from common import chunked_forward, cmp

d = chunked_forward()
n, Yb = d['n'], d['Yb']
logprobs, probs, counts, counts_sum, counts_sum_inv, norm_logits = (
    d[k] for k in ['logprobs', 'probs', 'counts', 'counts_sum', 'counts_sum_inv', 'norm_logits'])
print(f"loss {d['loss'].item():.4f}, n = {n}")

# loss = -logprobs[range(n), Yb].mean(): only the 32 picked entries matter, each with weight 1/n
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
print(f'dlogprobs: {int((dlogprobs != 0).sum())} non-zero of {dlogprobs.nelement()}, each = {dlogprobs[0, Yb[0]].item():.6f}')
cmp('logprobs', dlogprobs, logprobs)

dprobs = (1.0 / probs) * dlogprobs          # d/dx log(x) = 1/x: boosts low-probability correct chars
cmp('probs', dprobs, probs)
p0 = probs[0, Yb[0]].item()
print(f'  row 0: probs[0,Yb[0]] = {p0:.4f} -> dprobs = {dprobs[0, Yb[0]].item():.4f}')

dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)   # counts_sum_inv was broadcast (32,1)->(32,27): sum back
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

dcounts = counts_sum_inv * dprobs            # branch 1: counts -> probs
cmp('counts (1 of 2)', dcounts, counts)
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)
dcounts += torch.ones_like(counts) * dcounts_sum   # branch 2: counts -> counts_sum (a sum: broadcast back)
cmp('counts', dcounts, counts)

dnorm_logits = counts * dcounts              # d/dx e^x = e^x, and counts = e^norm_logits
cmp('norm_logits', dnorm_logits, norm_logits)

# Why the notebook writes counts_sum**-1 and not 1.0/counts_sum: the power rule matches autograd bit for bit
for name, f in [('counts_sum**-1', lambda c: c**-1), ('1.0/counts_sum', lambda c: 1.0/c)]:
    cs = counts_sum.detach().clone().requires_grad_()
    inv = f(cs); inv.retain_grad()
    (inv * dcounts_sum_inv.detach()).sum().backward()
    cmp(name, (-cs**-2) * inv.grad, cs)
