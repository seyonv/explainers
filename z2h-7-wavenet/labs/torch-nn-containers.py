# Card: torch-nn-containers · Lecture 6 ch.4 pytorchifying our code https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=556s
# Run from the course folder: python labs/torch-nn-containers.py   (the layer classes live in labs/common.py)
import warnings
import torch
import torch.nn.functional as F
from common import splits, flat_model, n_params, train, set_training, split_loss, sample

block_size = 3
data = splits(block_size)
Xtr, Ytr = data['train']
torch.manual_seed(42); # seed rng for reproducibility
model = flat_model(block_size, n_embd=10, n_hidden=200)   # Sequential([Embedding, Flatten, Linear, BatchNorm1d, Tanh, Linear])
print('layers:', ' -> '.join(type(l).__name__ for l in model.layers))
print('parameters:', n_params(model), '=', ' + '.join(str(p.nelement()) for p in model.parameters()))
Xb = Xtr[:4]
logits = model(Xb)                                          # the whole forward pass is one call
for layer in model.layers:
    print(f'  {type(layer).__name__:12s} out {tuple(layer.out.shape)}')

lossi = train(model, Xtr, Ytr, max_steps=20000, log=False)  # a short run is enough to sample from
set_training(model, False)
split_loss(model, data, 'val')
torch.manual_seed(1)
print('eval mode samples :', ' '.join(sample(model, block_size, n=8)))

# The bug: forget training=False. BatchNorm now normalizes a batch of ONE example with its own statistics.
set_training(model, True)
x = model.layers[2](model.layers[1](model.layers[0](torch.tensor([[0, 0, 0]]))))
warnings.filterwarnings('ignore', message='var\\(\\): degrees of freedom')   # torch warns about exactly this
print('batch of 1, the variance BatchNorm1d computes: x.var(0)[:3] =', x.var(0, keepdim=True)[0, :3].tolist())
logits = model(torch.tensor([[0, 0, 0]]))
print('logits with training=True:', logits[0, :4].tolist())
try:
    torch.manual_seed(1)
    print('train mode samples:', ' '.join(sample(model, block_size, n=8)))
except RuntimeError as e:
    print('train mode sampling crashes:', str(e).split('\n')[0])
set_training(model, False)
print('and the running stats are poisoned too: running_var has nan:', bool(model.layers[3].running_var.isnan().any()))
torch.manual_seed(1)
try:
    print('eval mode again   :', ' '.join(sample(model, block_size, n=8)))
except RuntimeError as e:
    print('eval mode sampling now crashes too:', str(e).split('\n')[0])
