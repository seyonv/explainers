# Card: probability-distributions-and-sampling · L2 https://www.youtube.com/watch?v=PaCmpygFfXo&t=1442s
# Run from z2h-1-toolkit/: python labs/probability-distributions-and-sampling.py
import torch
from common import data_path

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
print('p =', [round(v, 4) for v in p.tolist()], ' sum =', round(p.sum().item(), 4))
print('cumsum =', [round(v, 4) for v in p.cumsum(0).tolist()])
# the L2 notebook keeps drawing from the same g right after rand(3)
s = torch.multinomial(p, num_samples=100, replacement=True, generator=g)
print('notebook replay, first 20 of 100:', s[:20].tolist())
print('  counts of 0/1/2 in 100:', torch.bincount(s, minlength=3).tolist())

# cumulative-sum sampling by hand: draw u in [0,1), pick the first bucket whose cumsum exceeds u
g = torch.Generator().manual_seed(2147483647)
u = torch.rand(5, generator=g)
picks = [int((p.cumsum(0) > ui).nonzero()[0]) for ui in u]
print('u =', [round(v, 4) for v in u.tolist()], '-> picks', picks)

g = torch.Generator().manual_seed(2147483647)
s = torch.multinomial(p, num_samples=20, replacement=True, generator=g)
print('multinomial x20:', s.tolist())
try:
    torch.multinomial(p, num_samples=20, generator=g)
except RuntimeError as e:
    print('replacement=False (default), 20 of 3 ->', str(e).splitlines()[0])
print('replacement=False, 3 of 3 ->', torch.multinomial(p, num_samples=3, generator=g).tolist(), '(each index once)')

for n in [10, 100, 10_000]:
    g = torch.Generator().manual_seed(2147483647)
    s = torch.multinomial(p, num_samples=n, replacement=True, generator=g)
    freq = torch.bincount(s, minlength=3).float() / n
    print(f'n={n:>6}: frequencies {[round(v, 4) for v in freq.tolist()]}')

# the 27-character distribution for the first letter of a name
words = open(data_path('names.txt')).read().splitlines()
chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}; stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
N0 = torch.zeros(27, dtype=torch.int32)
for w in words:
    N0[stoi[w[0]]] += 1
p0 = N0.float() / N0.sum()
print(f'\n{len(words)} names; first-letter counts: a={N0[1].item()} m={N0[13].item()}; total={N0.sum().item()}')
print(f"p['.'] = {p0[0].item():.4f}  p['a'] = {p0[1].item():.4f}  p['m'] = {p0[13].item():.4f}")
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p0, num_samples=10, replacement=True, generator=g)
print('10 first letters:', ''.join(itos[i] for i in ix.tolist()))
