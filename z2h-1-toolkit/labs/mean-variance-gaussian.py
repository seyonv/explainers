# Card: mean-variance-gaussian (why weights get divided by sqrt(fan_in)) · L4 https://www.youtube.com/watch?v=P6sfmUTpUmc&t=1673s
# Run from z2h-1-toolkit/: python labs/mean-variance-gaussian.py [--plot]
import sys, torch

v = torch.tensor([2., 4., 4., 4., 5., 5., 7., 9.])
print('v =', v.tolist())
print(f'mean = {v.mean().item()}  var (n) = {v.var(unbiased=False).item()}  std = {v.std(unbiased=False).item()}')
print(f'var (n-1, torch default) = {v.var().item():.4f}  std = {v.std().item():.4f}')

g = torch.Generator().manual_seed(2147483647)
z = torch.randn(100_000, generator=g)
print(f'\nrandn(100000): mean {z.mean().item():.4f}  std {z.std().item():.4f}')
print(f'  within 1 std: {(z.abs() < 1).float().mean().item():.4f}  within 2: {(z.abs() < 2).float().mean().item():.4f}  within 3: {(z.abs() < 3).float().mean().item():.4f}')

for n in [1, 4, 10, 100]:
    s = torch.randn(100_000, n, generator=g).sum(1)
    print(f'sum of {n:>3} unit gaussians: std {s.std().item():.3f}  (sqrt(n) = {n**0.5:.3f})')

x = torch.randn(1000, 10, generator=g)
w = torch.randn(10, 200, generator=g)
y = x @ w
print(f'\nx (1000,10) std {x.std().item():.4f};  y = x @ w (200 outputs) std {y.std().item():.4f}')
for name, scale in [('w * 5', 5.0), ('w * 0.2', 0.2), ('w / sqrt(10)', 10**-0.5)]:
    print(f'  {name:13s} -> y std {(x @ (w * scale)).std().item():.4f}')

h = (y - y.mean(0, keepdim=True)) / y.std(0, keepdim=True)
print(f'standardised y (per column): mean {h.mean().item():.4f}  std {h.std(0).mean().item():.4f}')

if '--plot' in sys.argv:
    import os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.hist(x.flatten().tolist(), 50, density=True, alpha=0.6, label='x')
    plt.hist(y.flatten().tolist(), 50, density=True, alpha=0.6, label='x @ w')
    plt.legend(); out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mean-variance-gaussian.png')
    plt.savefig(out); print('saved', out)
