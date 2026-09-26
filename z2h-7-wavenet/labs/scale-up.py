# Card: scale-up · Lecture 6 ch.12 scaling up our WaveNet https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=2767s
# Run from the course folder: python labs/scale-up.py [--quick]   (full: 200k steps, the lecture's final model; --quick: 20k steps)
import torch
from common import splits, wavenet_model, n_params, train, set_training, split_loss, sample, steps_from_argv

block_size = 8
data = splits(block_size)
Xtr, Ytr = data['train']
torch.manual_seed(42); # seed rng for reproducibility
model = wavenet_model(n_embd=24, n_hidden=128)
print('parameters:', n_params(model))

max_steps = steps_from_argv()
lossi = train(model, Xtr, Ytr, max_steps)
set_training(model, False)   # put layers into eval mode (needed for batchnorm especially)
split_loss(model, data, 'train')
split_loss(model, data, 'val')
print('samples:', ' '.join(sample(model, block_size)))
