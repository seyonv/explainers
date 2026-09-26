# Card: bigram-counts.html · Lecture 2 · counting bigrams in a 2D torch tensor · https://www.youtube.com/watch?v=PaCmpygFfXo&t=765s
# Run from the course folder: python labs/bigram-counts.py   (add --plot to save bigram-counts.png)
import sys
import torch
from common import load_words, vocab

words = load_words()

# 1) a python dict with separate <S> and <E> tokens
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        b[(ch1, ch2)] = b.get((ch1, ch2), 0) + 1
print('distinct bigrams in dict:', len(b))
print('top 5:', sorted(b.items(), key=lambda kv: -kv[1])[:5])

# 2) the same counts in a 27x27 int32 tensor, with '.' as start and end
stoi, itos = vocab(words)
print('stoi a, z, .:', stoi['a'], stoi['z'], stoi['.'])
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1
print('N.shape, dtype:', tuple(N.shape), N.dtype)
print('N[0, :5] (.. .a .b .c .d):', N[0, :5].tolist())
print("N[stoi['e'], stoi['m']] (em):", N[stoi['e'], stoi['m']].item())
print("N[stoi['m'], 0] (m.):", N[stoi['m'], 0].item())
print("N[0, stoi['m']] (.m):", N[0, stoi['m']].item())
print('N[0].sum() (names):', N[0].sum().item())
print('N.sum() (bigrams):', N.sum().item())

if '--plot' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(N, cmap='Blues')
    plt.axis('off')
    plt.savefig(__file__.replace('.py', '.png'), dpi=120, bbox_inches='tight')
    print('saved', __file__.replace('.py', '.png'))
