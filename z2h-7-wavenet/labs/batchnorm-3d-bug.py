# Card: batchnorm-3d-bug · Lecture 6 ch.9-11 fixing the batchnorm1d bug https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=2330s
# Run from the course folder: python labs/batchnorm-3d-bug.py [--quick]   (trains the 22K hierarchical net twice, buggy then fixed: 2 x 200k steps; --quick: 2 x 20k)
import torch
from common import splits, wavenet_model, n_params, train, set_training, split_loss, steps_from_argv

block_size = 8
data = splits(block_size)
Xtr, Ytr = data['train']

# What BatchNorm1d sees after FlattenConsecutive(2) + Linear on a batch of 32: a 3-D tensor
x = torch.randn(32, 4, 68)
print('input (32, 4, 68): x.mean(0) ->', tuple(x.mean(0, keepdim=True).shape), '| x.mean((0,1)) ->', tuple(x.mean((0, 1), keepdim=True).shape))

results = {}
for fixed in [False, True]:
    name = 'fixed: reduce over (0,1)' if fixed else 'buggy: reduce over 0 only'
    print(f'--- {name} ---')
    torch.manual_seed(42); # seed rng for reproducibility
    model = wavenet_model(n_embd=10, n_hidden=68, fixed_bn=fixed)
    print('parameters:', n_params(model))
    model(Xtr[:32])
    for i in (3, 7, 11):
        print(f'  model.layers[{i}].running_mean.shape = {tuple(model.layers[i].running_mean.shape)}')
    train(model, Xtr, Ytr, steps_from_argv(), log=False)
    set_training(model, False)
    results[name] = (split_loss(model, data, 'train'), split_loss(model, data, 'val'))
print('video/notebook: buggy train 1.941 val 2.029 | fixed train 1.912 val 2.022')
for name, (tr, va) in results.items():
    print(f'{name}: train {tr:.3f} val {va:.3f}')
