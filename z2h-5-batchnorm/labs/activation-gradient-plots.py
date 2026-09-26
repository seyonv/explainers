# Card: activation-gradient-plots (z2h-5-batchnorm) · Lecture 4 "viz #1/#2" and "the fully linear case" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=5211s
# Run from the course folder: python labs/activation-gradient-plots.py [--plot: save forward/backward histograms for each gain]
import os, torch
from common import Linear, Tanh, deep_mlp, train_deep, flag, HERE

def stats(layers, kind):
    """per-layer (index, mean, std, % saturated, grad std) for every layer of type `kind`"""
    rows = []
    for i, layer in enumerate(layers[:-1]):  # exclude the output layer
        if isinstance(layer, kind):
            t = layer.out
            rows.append((i, t.mean().item(), t.std().item(), (t.abs() > 0.97).float().mean().item()*100, t.grad.std().item()))
    return rows

def show(title, rows, sat=True):
    print(title)
    print('  act std  ' + ' '.join(f'{r[2]:8.2f}' for r in rows))
    if sat:
        print('  sat %    ' + ' '.join(f'{r[3]:8.2f}' for r in rows))
    print('  grad std ' + ' '.join(f'{r[4]:8.1e}' for r in rows))

hists = {}
# 1. no BatchNorm, tanh: one forward + backward at init, sweeping the gain on the hidden Linear weights
for name, gain in [('0.5', 0.5), ('1', 1.0), ('5/3', 5/3), ('3', 3.0)]:
    C, layers, parameters = deep_mlp(bn=False, gain=gain)
    train_deep(C, layers, parameters, 1, track=False)
    show(f'no BN, tanh, gain {name}: Tanh layers {[r[0] for r in stats(layers, Tanh)]}', stats(layers, Tanh))
    hists[name] = [(l.out.detach(), l.out.grad) for l in layers[:-1] if isinstance(l, Tanh)]

# 2. the fully linear case: remove every Tanh
for name, gain in [('5/3', 5/3), ('1', 1.0)]:
    C, layers, parameters = deep_mlp(bn=False, gain=gain, tanh=False)
    train_deep(C, layers, parameters, 1, track=False)
    show(f'no BN, NO tanh, gain {name}: Linear layers {[r[0] for r in stats(layers, Linear)]}', stats(layers, Linear), sat=False)

# 3. with BatchNorm (the notebook's run: gain 1.0, 1001 steps)
for name, gain in [('1', 1.0), ('5/3', 5/3)]:
    C, layers, parameters = deep_mlp(bn=True, gain=gain)
    train_deep(C, layers, parameters, 1001, track=False)
    show(f'BN, tanh, gain {name}, after 1001 steps: Tanh layers {[r[0] for r in stats(layers, Tanh)]}', stats(layers, Tanh))

if flag('--plot'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(20, 7))
    for col, (name, hs) in enumerate(hists.items()):
        for k, (t, gr) in enumerate(hs):
            for row, v in enumerate([t, gr]):
                hy, hx = torch.histogram(v, density=True)
                ax[row][col].plot(hx[:-1], hy, label=f'Tanh {k+1}')
        ax[0][col].set_title(f'activations, gain {name}'); ax[1][col].set_title(f'gradients, gain {name}')
        ax[0][col].legend()
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'activation-gradient-plots.png'), dpi=100)
    print('saved labs/activation-gradient-plots.png')
