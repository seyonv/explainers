# Card: saturated-tanh (z2h-5-batchnorm) · Lecture 4 "fixing the saturated tanh" https://www.youtube.com/watch?v=P6sfmUTpUmc&t=779s
# Run from the course folder: python labs/saturated-tanh.py [--quick: 20k steps] [--plot: save h histogram + |h|>0.99 image]
import os, torch
from common import mlp1, mlp1_forward, train_mlp1, split_loss, steps_from_argv, flag, Xtr, g, HERE, Timer

def inspect(stage):
    P = mlp1(stage)
    state = g.get_state()
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)  # the step-0 minibatch
    with torch.no_grad():
        hpreact, h, _ = mlp1_forward(P, Xtr[ix])
        _, h_all, _ = mlp1_forward(P, Xtr)                      # every training example
    g.set_state(state)  # so training below draws the same minibatches as the notebook
    print(f'--- {stage} (W1 x {"0.2, b1 x 0.01" if stage == "fix-tanh" else "1, b1 x 1"}) ---')
    print(f'hpreact range: {hpreact.min().item():.1f} to {hpreact.max().item():.1f}')
    counts = torch.histc(h, bins=10, min=-1, max=1).int().tolist()
    print(f'h histogram, 10 bins from -1 to 1 ({h.numel()} values): {counts}')
    print(f'|h| > 0.99: {(h.abs() > 0.99).float().mean().item()*100:.1f}% of the 32x200 activations')
    print(f'mean local gradient (1 - h**2): {(1 - h**2).mean().item():.3f}  (1.0 = passes gradient fully)')
    print(f'dead columns (|h|>0.99 for all 32 examples): {(h.abs() > 0.99).all(0).sum().item()} of 200')
    print(f'dead over all {Xtr.shape[0]} training examples: {(h_all.abs() > 0.99).all(0).sum().item()} of 200')
    return P, h

P0, h0 = inspect('fix-logits')
P1, h1 = inspect('fix-tanh')

max_steps = steps_from_argv()
with Timer() as t:
    train_mlp1(P1, max_steps, verbose=False)
print(f'after training fix-tanh ({max_steps} steps, {t.s:.0f}s): train {split_loss(P1, "train"):.4f} val {split_loss(P1, "val"):.4f}')

if flag('--plot'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 6))
    for row, (h, name) in enumerate([(h0, 'W1 x 1'), (h1, 'W1 x 0.2')]):
        ax[row][0].hist(h.view(-1).tolist(), 50); ax[row][0].set_title(f'h histogram, {name}')
        ax[row][1].imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest'); ax[row][1].set_title('|h| > 0.99 (white)')
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'saturated-tanh.png'), dpi=110)
    print('saved labs/saturated-tanh.png')
