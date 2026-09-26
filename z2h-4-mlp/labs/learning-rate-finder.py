# Card: learning-rate-finder · Lecture 3 ch.12 https://www.youtube.com/watch?v=TCH_1BHY58I&t=2740s
# Run from the course folder: python labs/learning-rate-finder.py [--plot]
import sys
import torch
from common import HERE, words, build_dataset, init_params, split_loss, train

X, Y = build_dataset(words)  # all 228,146 examples, as in this chapter

# 1) the sweep: step i uses lr = lrs[i], from 10**-3 up to 10**0
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
g, parameters = init_params(n_embd=2, n_hidden=100)
lossi = torch.tensor(train(parameters, X, Y, 1000, lambda i: lrs[i].item(), g))
print('lr sweep, mean minibatch loss per exponent bin (100 steps each):')
for k in range(10):
    s = slice(100 * k, 100 * (k + 1))
    print(f'  exponent {lre[s][0].item():+.2f} .. {lre[s][-1].item():+.2f}  (lr {lrs[s][0].item():.4f} .. {lrs[s][-1].item():.3f})'
          f'  loss {lossi[s].mean().item():6.3f}')
smooth = torch.nn.functional.avg_pool1d(lossi.view(1, 1, -1), 50, 1).flatten()
best = smooth.argmin().item() + 25
print(f'lowest 50-step moving average at step {best}: exponent {lre[best].item():.2f}, lr {lrs[best].item():.3f}')

if '--plot' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(lre, lossi)
    plt.xlabel('learning-rate exponent'); plt.ylabel('minibatch loss')
    plt.savefig(f'{HERE}/learning-rate-finder.png', dpi=120)
    print('saved labs/learning-rate-finder.png')

# 2) train at lr 0.1, then decay 10x to 0.01
g, parameters = init_params(n_embd=2, n_hidden=100)
for steps, lr in [(10000, 0.1), (10000, 0.1), (10000, 0.1), (10000, 0.01)]:
    train(parameters, X, Y, steps, lambda i: lr, g)
    print(f'{steps} more steps at lr {lr}: full-set loss {split_loss(parameters, X, Y):.4f}')
