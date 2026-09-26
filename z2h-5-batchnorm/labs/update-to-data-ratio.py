# Card: update-to-data-ratio (z2h-5-batchnorm) · Lecture 4 "viz #3", "viz #4", "bringing back batchnorm" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=5775s
# Run from the course folder: python labs/update-to-data-ratio.py [--plot: save the ud-over-time plot for each setting]
import os
from common import Tanh, deep_mlp, train_deep, flag, HERE, Timer

# 1. grad:data ratio of each 2-D weight after the notebook's 1001 steps (BN, gain 1.0, lr 0.1)
C, layers, parameters = deep_mlp(bn=True, gain=1.0)
train_deep(C, layers, parameters, 1001, track=False)
for p in parameters:
    if p.ndim == 2:
        print(f'weight {str(tuple(p.shape)):10s} | grad std {p.grad.std().item():.2e} | grad:data ratio {(p.grad.std() / p.std()).item():.2e}')

# 2. ud = log10(std(lr * grad) / std(data)) per step; averaged over steps 900-1000, for each 2-D weight
runs = [('BN, gain 1, lr 0.1 (notebook)',        dict(bn=True, gain=1.0), 0.1),
        ('BN, gain 1, lr 0.001',                 dict(bn=True, gain=1.0), 0.001),
        ('no BN, gain 5/3, lr 0.1',              dict(bn=False, gain=5/3), 0.1),
        ('no BN, no fan_in scaling, lr 0.1',     dict(bn=False, gain=1.0, fan_in_scaling=False), 0.1),
        ('BN, gain 0.2, lr 0.1',                 dict(bn=True, gain=0.2), 0.1),
        ('BN, no fan_in scaling, lr 0.1',        dict(bn=True, gain=1.0, fan_in_scaling=False), 0.1),
        ('BN, no fan_in scaling, lr 1.0',        dict(bn=True, gain=1.0, fan_in_scaling=False), 1.0)]
plots = {}
for name, kw, lr in runs:
    C, layers, parameters = deep_mlp(**kw)
    with Timer() as t:
        lossi, ud = train_deep(C, layers, parameters, 1001, lr=lr)
    idx = [i for i, p in enumerate(parameters) if p.ndim == 2]
    late = [sum(ud[j][i] for j in range(900, 1001)) / 101 for i in idx]
    tanhs = [l.out for l in layers if isinstance(l, Tanh)]
    sat = sum((t.abs() > 0.97).float().mean().item() for t in tanhs) / len(tanhs) * 100
    print(f'{name:34s} ud step 0: {" ".join(f"{ud[0][i]:5.2f}" for i in idx)}')
    print(f'{"":34s} ud 900-1000: {" ".join(f"{v:5.2f}" for v in late)} | Tanh sat {sat:4.1f}% | '
          f'loss last 100 {sum(lossi[-100:])/100:.3f} ({t.s:.1f}s)')
    plots[name] = [[ud[j][i] for j in range(len(ud))] for i in idx]
print('columns: C (27,10), W (30,100), 4 x W (100,100), W (100,27); target is about -3')

if flag('--plot'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(plots), 1, figsize=(12, 3 * len(plots)))
    for a, (name, curves) in zip(ax, plots.items()):
        for k, c in enumerate(curves):
            a.plot(c, label=f'param {k}')
        a.plot([0, len(curves[0])], [-3, -3], 'k')  # these ratios should be ~1e-3
        a.set_title(name); a.set_ylim(-6, 0)
    ax[0].legend(ncol=7)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'update-to-data-ratio.png'), dpi=90)
    print('saved labs/update-to-data-ratio.png')
