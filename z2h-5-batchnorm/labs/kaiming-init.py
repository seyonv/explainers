# Card: kaiming-init (z2h-5-batchnorm) · Lecture 4 "calculating the init scale: Kaiming init" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=1673s
# Run from the course folder: python labs/kaiming-init.py [--quick: 20k steps]
import torch
from common import mlp1, mlp1_forward, train_mlp1, split_loss, steps_from_argv, Xtr, g, Timer

# 1. the toy: x is 1000 examples of 10 unit-Gaussian numbers, w maps 10 -> 200
gt = torch.Generator().manual_seed(2147483647)
x = torch.randn(1000, 10, generator=gt)
w = torch.randn(10, 200, generator=gt)
print(f'x std {x.std().item():.3f}')
for name, scale in [('w', 1), ('w * 5', 5), ('w * 0.2', 0.2), ('w / sqrt(10)', 10**-0.5)]:
    y = x @ (w * scale)
    print(f'y = x @ ({name:12s}): y std {y.std().item():.3f}')
print(f'sqrt(fan_in) = sqrt(10) = {10**0.5:.3f}')

# 2. ReLU throws away half the distribution: gain sqrt(2) puts the second moment back
for name, gain in [('1', 1), ('sqrt(2)', 2**0.5)]:
    y = torch.relu(x @ (w * gain / 10**0.5))
    print(f'relu(x @ w * {name:7s} / sqrt(10)): mean square {(y**2).mean().item():.3f}')

# 3. the gains torch uses, and the tanh one applied to W1 (fan_in = n_embd * block_size = 30)
for nl in ['linear', 'relu', 'tanh']:
    print(f'torch.nn.init.calculate_gain({nl!r}) = {torch.nn.init.calculate_gain(nl):.4f}')
print(f'W1 scale (5/3)/30**0.5 = {(5/3)/30**0.5:.4f}   (the hand-tuned value was 0.2)')

# 4. the MLP with Kaiming W1: saturation at init, then a full run
P = mlp1('kaiming')
state = g.get_state()
ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
with torch.no_grad():
    hpreact, h, _ = mlp1_forward(P, Xtr[ix])
g.set_state(state)
print(f'init: hpreact std {hpreact.std().item():.3f}, |h| > 0.99 on {(h.abs() > 0.99).float().mean().item()*100:.1f}%')
max_steps = steps_from_argv()
with Timer() as t:
    train_mlp1(P, max_steps, verbose=False)
print(f'kaiming ({max_steps} steps, {t.s:.0f}s): train {split_loss(P, "train"):.4f} val {split_loss(P, "val"):.4f}')
