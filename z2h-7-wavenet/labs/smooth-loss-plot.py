# Card: smooth-loss-plot · Lecture 6 ch.3 let's fix the learning rate plot https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=416s
# Run from the course folder: python labs/smooth-loss-plot.py [--quick] [--plot]   (--plot saves loss-raw.png and loss-smooth.png)
import os
import torch
from common import HERE, splits, flat_model, n_params, train, set_training, split_loss, steps_from_argv, flag

block_size = 3   # the starter code: 3 characters of context
data = splits(block_size)
Xtr, Ytr = data['train']
torch.manual_seed(42); # seed rng for reproducibility
model = flat_model(block_size, n_embd=10, n_hidden=200)
print('parameters:', n_params(model))
max_steps = steps_from_argv()
lossi = train(model, Xtr, Ytr, max_steps)

L = torch.tensor(lossi)                  # log10 of each minibatch loss
smooth = L.view(-1, 1000).mean(1)        # one point per 1000 steps
print(f'lossi: {len(lossi)} points -> smoothed: {tuple(smooth.shape)[0]} points')
print(f'raw spread over the last 1000 steps: min {L[-1000:].min():.3f}, max {L[-1000:].max():.3f} (log10 loss)')
every = max(1, len(smooth) // 20)
print('smoothed log10 loss, every', every, 'windows:')
print('  ' + ' '.join(f'{v:.3f}' for v in smooth[::every].tolist()))
d = len(smooth) * 3 // 4                  # the window where lr drops 0.1 -> 0.01
k = max(1, len(smooth) // 20)
before, after = smooth[d-k:d].mean().item(), smooth[d:d+k].mean().item()
print(f'lr decay at step {d*1000}: mean of {k} windows before {before:.3f} -> after {after:.3f}'
      f'  (loss {10**before:.3f} -> {10**after:.3f})')

set_training(model, False)
split_loss(model, data, 'train')
split_loss(model, data, 'val')

if flag('--plot'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, y in [('loss-raw.png', L), ('loss-smooth.png', smooth)]:
        plt.figure(figsize=(6, 3)); plt.plot(y); plt.ylabel('log10 loss')
        plt.savefig(os.path.join(HERE, name), dpi=120, bbox_inches='tight'); plt.close()
        print('saved', os.path.join(HERE, name))
