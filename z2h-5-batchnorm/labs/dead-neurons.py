# Card: dead-neurons (z2h-5-batchnorm) · Lecture 4 "fixing the saturated tanh" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=779s
# Run from the course folder: python labs/dead-neurons.py [--plot: save the four functions and their local gradients]
import os, torch
import torch.nn.functional as F
from common import Xtr, Ytr, g, reset_g, flag, HERE

acts = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu,
        'leaky_relu': lambda x: F.leaky_relu(x, 0.01)}

# 1. value and local gradient of each nonlinearity
xs = torch.tensor([-4., -2., -0.5, 0.5, 2., 4.], requires_grad=True)
print('x            ' + ' '.join(f'{v:>7.1f}' for v in xs.tolist()))
for name, f in acts.items():
    xs.grad = None
    f(xs).sum().backward()
    print(f'{name:10s} f  ' + ' '.join(f'{v:7.3f}' for v in f(xs).tolist()))
    print(f'{"":10s} df ' + ' '.join(f'{v:7.3f}' for v in xs.grad.tolist()))

# 2. how much of [-5, 5] is (nearly) flat, i.e. passes back < 1% of the gradient?
grid = torch.linspace(-5, 5, 10001, requires_grad=True)
for name, f in acts.items():
    grid.grad = None
    f(grid).sum().backward()
    print(f'{name:10s}: {(grid.grad.abs() < 0.01).float().mean().item()*100:5.1f}% of [-5, 5] has |local grad| < 0.01')

# 3. a ReLU MLP (dead = outputs <= 0 on all data, so no gradient ever reaches its weights again)
# (Kaiming init, sqrt(2)/sqrt(30)): 1000 normal steps, ONE step at lr=100, 1000 normal steps
def run(name, act, shock_lr):
    reset_g()
    C = torch.randn((27, 10), generator=g)
    W1 = torch.randn((30, 200), generator=g) * (2**0.5) / 30**0.5
    b1 = torch.zeros(200)
    W2 = torch.randn((200, 27), generator=g) * 0.01
    b2 = torch.zeros(27)
    ps = [C, W1, b1, W2, b2]
    for p in ps:
        p.requires_grad = True
    def step(lr):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        h = act(C[Xtr[ix]].view(-1, 30) @ W1 + b1)
        loss = F.cross_entropy(h @ W2 + b2, Ytr[ix])
        for p in ps:
            p.grad = None
        loss.backward()
        for p in ps:
            p.data += -lr * p.grad
        return loss.item()
    @torch.no_grad()
    def dead():  # neurons negative on all 182,625 training examples: dead for ReLU, still 1% gradient for Leaky ReLU
        return (C[Xtr].view(-1, 30) @ W1 + b1 <= 0).all(0).sum().item()
    for i in range(1000): step(0.1)
    before = dead()
    step(shock_lr)
    losses = [step(0.1) for i in range(1000)]
    print(f'{name:10s} shock lr {shock_lr:<5}: all-negative neurons before {before:3d}/200, '
          f'after {dead():3d}/200 | mean loss of last 100 steps {sum(losses[-100:])/100:.4f}')

for shock in [0.1, 30, 100]:
    run('relu', torch.relu, shock)
for shock in [0.1, 30]:
    run('leaky_relu', acts['leaky_relu'], shock)

if flag('--plot'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
    for a, (name, f) in zip(ax, acts.items()):
        x = torch.linspace(-5, 5, 400, requires_grad=True)
        y = f(x); y.sum().backward()
        a.plot(x.detach(), y.detach(), label='f(x)'); a.plot(x.detach(), x.grad, '--', label="f'(x)")
        a.set_title(name); a.axhline(0, color='k', lw=0.5); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'dead-neurons.png'), dpi=110)
    print('saved labs/dead-neurons.png')
