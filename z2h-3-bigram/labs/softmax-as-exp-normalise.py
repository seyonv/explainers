# Card: softmax-as-exp-normalise.html · Lecture 2 · the softmax + vectorized loss · https://www.youtube.com/watch?v=PaCmpygFfXo&t=4726s
# Run from the course folder: python labs/softmax-as-exp-normalise.py
import torch
import torch.nn.functional as F
from common import load_words, vocab

words = load_words()
stoi, itos = vocab(words)
xs, ys = [], []
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g)

xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W                               # log-counts
counts = logits.exp()                           # like N
probs = counts / counts.sum(1, keepdims=True)   # softmax
print('logits[0, :5]:', [round(x, 4) for x in logits[0, :5].tolist()])
print('counts[0, :5]:', [round(x, 4) for x in counts[0, :5].tolist()])
print('probs[0, :5] :', [round(x, 4) for x in probs[0, :5].tolist()])
print('probs.shape:', tuple(probs.shape), '| row sums:', [round(x, 4) for x in probs.sum(1).tolist()])
print('matches torch.softmax:', torch.allclose(probs, torch.softmax(logits, dim=1)))

nlls = torch.zeros(5)
for i in range(5):
    x, y = xs[i].item(), ys[i].item()
    p = probs[i, y]
    nlls[i] = -torch.log(p)
    print(f'{itos[x]}{itos[y]} (indexes {x},{y}): p={p:.4f} logp={torch.log(p):.4f} nll={nlls[i]:.4f}')
print('average nll (loop):   %.4f' % nlls.mean())
loss = -probs[torch.arange(5), ys].log().mean()
print('vectorised loss:      %.4f' % loss)
print('F.cross_entropy:      %.4f' % F.cross_entropy(logits, ys))
print('uniform guess log(27) = %.4f' % torch.log(torch.tensor(27.0)))
