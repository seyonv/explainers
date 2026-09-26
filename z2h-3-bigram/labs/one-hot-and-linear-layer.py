# Card: one-hot-and-linear-layer.html · Lecture 2 · one-hot encodings + one linear layer · https://www.youtube.com/watch?v=PaCmpygFfXo&t=4201s
# Run from the course folder: python labs/one-hot-and-linear-layer.py
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
print('xs:', xs.tolist(), 'ys:', ys.tolist())
print('torch.tensor dtype:', xs.dtype, '| torch.Tensor dtype:', torch.Tensor([0, 5]).dtype)

print('one_hot without num_classes, shape:', tuple(F.one_hot(xs).shape))
xenc = F.one_hot(xs, num_classes=27)
print('one_hot dtype:', xenc.dtype)
xenc = xenc.float()
print('xenc.shape:', tuple(xenc.shape), xenc.dtype)
print('xenc[1] (the e):', xenc[1].int().tolist())

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 1), generator=g)
print('one neuron, (xenc @ W).shape:', tuple((xenc @ W).shape))

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g)
out = xenc @ W
print('27 neurons, (xenc @ W).shape:', tuple(out.shape))
print('(xenc @ W)[3, 13] = %.4f' % out[3, 13])
print('(xenc[3] * W[:, 13]).sum() = %.4f' % (xenc[3] * W[:, 13]).sum())
print('W[13, 13] (row xs[3]=13, col 13) = %.4f' % W[13, 13])
