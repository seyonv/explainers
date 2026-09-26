# Card: count-matrix-visualised.html · Lecture 2 · visualizing the bigram tensor · https://www.youtube.com/watch?v=PaCmpygFfXo&t=1099s
# Run from the course folder: python labs/count-matrix-visualised.py   (add --plot to save the labelled imshow)
import sys
import torch
from common import setup

words, stoi, itos, N = setup()

def top(vals, idx_to_label, k=5):
    v, ix = vals.topk(k)
    return [(idx_to_label(i.item()), c.item()) for c, i in zip(v, ix)]

print("'e' row, top 5 next chars:", top(N[stoi['e']].float(), lambda i: 'e' + itos[i]))
print("'e' row total:", N[stoi['e']].sum().item())
print("'.' row (first letters), top 5:", top(N[0].float(), lambda i: '.' + itos[i]))
print("'.' column (last letters), top 5:", top(N[:, 0].float(), lambda i: itos[i] + '.'))
flat = N.flatten().float()
print('top 5 bigrams overall:', top(flat, lambda i: itos[i // 27] + itos[i % 27]))
print('N[0,0] (empty name):', N[0, 0].item())
print('cells that are zero:', (N == 0).sum().item(), 'of', N.numel())
print('max cell:', N.max().item())

if '--plot' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')
    for i in range(27):
        for j in range(27):
            plt.text(j, i, itos[i] + itos[j], ha='center', va='bottom', color='gray')
            plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
    plt.axis('off')
    plt.savefig(__file__.replace('.py', '.png'), dpi=80, bbox_inches='tight')
    print('saved', __file__.replace('.py', '.png'))
