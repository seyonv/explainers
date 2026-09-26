# Card: sampling-multinomial.html · Lecture 2 · sampling from the model · https://www.youtube.com/watch?v=PaCmpygFfXo&t=1442s
# Run from the course folder: python labs/sampling-multinomial.py
import torch
from common import setup

words, stoi, itos, N = setup()

p = N[0].float()
p = p / p.sum()
print("row 0 as probs: p['.'] = %.4f, p['a'] = %.4f, sum = %.4f" % (p[0], p[1], p.sum()))

g = torch.Generator().manual_seed(2147483647)
p3 = torch.rand(3, generator=g)
p3 = p3 / p3.sum()
print('torch.rand(3) normalised:', [round(x, 4) for x in p3.tolist()])
draws = torch.multinomial(p3, num_samples=100, replacement=True, generator=g)
print('100 draws from it, counts of 0/1/2:', torch.bincount(draws, minlength=3).tolist())

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print('first draw from row 0:', ix, repr(itos[ix]))

P = N.float()
P /= P.sum(1, keepdim=True)

def sample(get_p, n=5):
    g = torch.Generator().manual_seed(2147483647)
    names = []
    for i in range(n):
        out, ix = [], 0
        while True:
            ix = torch.multinomial(get_p(ix), num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        names.append(''.join(out))
    return names

print('bigram samples :', sample(lambda ix: P[ix]))
print('uniform samples:', sample(lambda ix: torch.ones(27) / 27))
lens = [len(s) - 1 for s in sample(lambda ix: P[ix], 1000)]
print('bigram: mean length over 1000 samples: %.2f (real names: %.2f)'
      % (sum(lens) / len(lens), sum(len(w) for w in words) / len(words)))
print('bigram: 1-letter names in 1000 samples:', sum(1 for l in lens if l == 1))
