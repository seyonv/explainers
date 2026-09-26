# Card: pytorchify-layers (z2h-5-batchnorm) · Lecture 4 "PyTorch-ifying the code" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=4715s
# Run from the course folder: python labs/pytorchify-layers.py   (the Linear / BatchNorm1d / Tanh classes live in labs/common.py)
import torch
import torch.nn.functional as F
from common import Linear, BatchNorm1d, Tanh, deep_mlp, deep_forward, train_deep, splits, itos, block_size, Timer

# 1. the 6-layer stack: 5 x (Linear, BatchNorm1d, Tanh) + (Linear, BatchNorm1d); notebook init (gain 1.0, last gamma x 0.1)
C, layers, parameters = deep_mlp(bn=True, gain=1.0)
for i, layer in enumerate(layers):
    shape = tuple(layer.weight.shape) if isinstance(layer, Linear) else (tuple(layer.gamma.shape) if isinstance(layer, BatchNorm1d) else '')
    print(f'layer {i:2d}: {layer.__class__.__name__:11s} {shape}')
print(f'parameters with BatchNorm: {sum(p.nelement() for p in parameters):,}')
_, _, p_nobn = deep_mlp(bn=False)
print(f'parameters without BatchNorm (Linear with bias): {sum(p.nelement() for p in p_nobn):,}')

# 2. train 1001 steps, as the notebook does before its `break`
with Timer() as t:
    lossi, _ = train_deep(C, layers, parameters, 1001, track=False)
print(f'step-0 loss {lossi[0]:.4f}, step-1000 loss {lossi[-1]:.4f} ({t.s:.1f}s)')

# 3. evaluation: flip the .training flag so BatchNorm uses its running buffers
for layer in layers:
    layer.training = False
with torch.no_grad():
    for split in ['train', 'val']:
        x, y = splits[split]
        print(f'{split} loss (eval mode): {F.cross_entropy(deep_forward(C, layers, x), y).item():.4f}')

# 4. sampling one name at a time works only because of eval mode
g = torch.Generator().manual_seed(2147483647 + 10)
names = []
for _ in range(10):
    out, context = [], [0] * block_size
    while True:
        with torch.no_grad():
            logits = deep_forward(C, layers, torch.tensor([context]))
        ix = torch.multinomial(F.softmax(logits, dim=1), num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    names.append(''.join(itos[i] for i in out))
print('samples:', ' '.join(names))
