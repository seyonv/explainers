# Card: exp-and-log (products become sums) · L2 https://www.youtube.com/watch?v=PaCmpygFfXo&t=3014s
# Run from z2h-1-toolkit/: python labs/exp-and-log.py
import math, torch

print('log(exp(2.0)) =', math.log(math.exp(2.0)))
a, b = 0.2, 0.05
print(f'log(a*b) = {math.log(a*b):.4f};  log a + log b = {math.log(a) + math.log(b):.4f}')
for p in [1.0, 0.5, 0.1, 0.01]:
    print(f'  log({p}) = {math.log(p):.4f}')
print(f'log(1/27) = {math.log(1/27):.4f}   -> -log(1/27) = {-math.log(1/27):.4f}')

# the word "emma" under the L2 bigram model: 5 next-character probabilities (.e em mm ma a.)
from common import data_path
words = open(data_path('names.txt')).read().splitlines()
stoi = {s: i + 1 for i, s in enumerate(sorted(set(''.join(words))))}; stoi['.'] = 0
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    cs = ['.'] + list(w) + ['.']
    for c1, c2 in zip(cs, cs[1:]):
        N[stoi[c1], stoi[c2]] += 1
P = N.float() / N.float().sum(1, keepdim=True)
cs = ['.'] + list('emma') + ['.']
probs = torch.stack([P[stoi[c1], stoi[c2]] for c1, c2 in zip(cs, cs[1:])])
print('\nP(.e em mm ma a.) =', [round(v, 4) for v in probs.tolist()])
print(f'product         = {probs.prod().item():.4e}')
print(f'sum of logs     = {probs.log().sum().item():.4f}')
print(f'exp(sum of logs)= {probs.log().sum().exp().item():.4e}')

# 1000 factors of 1/27 in float32: the product underflows to 0, the log-sum is fine
p = torch.full((1000,), 1/27, dtype=torch.float32)
print(f'\n1000 x (1/27), float32 product   = {p.prod().item()}')
print(f'1000 x (1/27), float32 sum of log = {p.log().sum().item():.2f}')
print(f'  exact: 1000 * log(1/27)         = {1000*math.log(1/27):.2f}')
print(f'float32 smallest normal number = {torch.finfo(torch.float32).smallest_normal:.3e}; (1/27)**1000 = 10^{1000*math.log10(1/27):.1f}')

# exp blows up: float32 overflows past about 88.7
for v in [10., 88., 89., 100.]:
    print(f'exp({v:g}) float32 = {torch.tensor(v).exp().item():.4e}')
logits = torch.tensor([100., 101., 102.])
print('softmax naive     :', (logits.exp() / logits.exp().sum()).tolist())
m = logits - logits.max()
print('softmax minus max :', [round(v, 4) for v in (m.exp() / m.exp().sum()).tolist()])
