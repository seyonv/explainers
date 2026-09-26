# Card: flatten-consecutive · Lecture 6 ch.8 implementing WaveNet https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=1296s
# Run from the course folder: python labs/flatten-consecutive.py
import torch
from common import splits, Embedding, Flatten, FlattenConsecutive, wavenet_model, n_params, set_training

# 1) @ only multiplies the last dimension; every leading dimension is a batch dimension
print('(4, 80) @ (80, 200)    ->', tuple((torch.randn(4, 80) @ torch.randn(80, 200)).shape))
print('(4, 5, 80) @ (80, 200) ->', tuple((torch.randn(4, 5, 80) @ torch.randn(80, 200)).shape))
print('(4, 4, 20) @ (20, 200) ->', tuple((torch.randn(4, 4, 20) @ torch.randn(20, 200)).shape))

# 2) Group consecutive pairs of the 8 character embeddings: view (4,8,10) as (4,4,20)
data = splits(8)
Xtr = data['train'][0]
torch.manual_seed(42)
ix = torch.randint(0, Xtr.shape[0], (4,))
Xb = Xtr[ix]
e = Embedding(27, 10)(Xb)
print('Xb', tuple(Xb.shape), '-> Embedding', tuple(e.shape), '-> Flatten', tuple(Flatten()(e).shape))
explicit = torch.cat([e[:, ::2, :], e[:, 1::2, :]], dim=2)
print('e.view(4, 4, 20) == cat([e[:, ::2], e[:, 1::2]], 2):', torch.equal(e.view(4, 4, 20), explicit))
print('FlattenConsecutive(2)', tuple(FlattenConsecutive(2)(e).shape), '| FlattenConsecutive(8)', tuple(FlattenConsecutive(8)(e).shape), '(squeezed)')

# 3) Stack 3 levels (the lecture's shape walk-through uses n_embd 10, n_hidden 200)
torch.manual_seed(42)
model = wavenet_model(n_embd=10, n_hidden=200)
model(Xb)
for layer in model.layers:
    print(f'  {type(layer).__name__:18s} : {tuple(layer.out.shape)}')

# 4) n_hidden 68 gives about the flat block_size-8 model's 22,097 parameters
torch.manual_seed(42)
print('parameters, n_hidden 200:', n_params(model), '| n_hidden 68:', n_params(wavenet_model(n_embd=10, n_hidden=68)))
