# Card: more-context-baseline · Lecture 6 ch.6-7 bump the context size to 8 https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=1173s
# Run from the course folder: python labs/more-context-baseline.py [--quick]   (full: 200k steps; --quick: 20k)
import torch
from common import splits, itos, flat_model, n_params, train, set_training, split_loss, steps_from_argv

block_size = 8 # context length: how many characters do we take to predict the next one?
data = splits(block_size)
Xtr, Ytr = data['train']
print('Xtr', tuple(Xtr.shape), 'Xdev', tuple(data['val'][0].shape), 'Xte', tuple(data['test'][0].shape))
for x, y in zip(Xtr[:20], Ytr[:20]):
    print(''.join(itos[ix.item()] for ix in x), '-->', itos[y.item()])

torch.manual_seed(42); # seed rng for reproducibility
model = flat_model(block_size, n_embd=10, n_hidden=200)   # the same flat MLP, now 8*10 = 80 inputs
print('parameters:', n_params(model), '(block_size 3 had 12097)')
lossi = train(model, Xtr, Ytr, steps_from_argv())
set_training(model, False)
split_loss(model, data, 'train')
split_loss(model, data, 'val')
