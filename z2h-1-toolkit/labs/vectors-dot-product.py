# Card: vectors-dot-product (the L1 neuron as a dot product) · L1 https://www.youtube.com/watch?v=VMj-3S1tku0&t=3172s
# Run from z2h-1-toolkit/: python labs/vectors-dot-product.py
import math, torch

x = [2.0, 0.0]
w = [-3.0, 1.0]
b = 6.8813735870195432
dot = sum(xi*wi for xi, wi in zip(x, w))
print('x =', x, ' w =', w)
print('x.w = 2*(-3) + 0*1 =', dot)
n = dot + b
print(f'n = x.w + b = {n:.4f}  (full: {n!r})')
print(f'o = tanh(n) = {math.tanh(n):.4f}  (1/sqrt(2) = {1/math.sqrt(2):.4f})')

xt, wt = torch.tensor(x), torch.tensor(w)
print('torch.dot(x, w) =', torch.dot(xt, wt).item(), ' x @ w =', (xt @ wt).item())

# alignment: dot = |a||b|cos(angle)
def cos(a, b): return (a @ b / (a.norm() * b.norm())).item()
u = torch.tensor([1.0, 0.0])
for name, v in [('same', torch.tensor([2.0, 0.0])), ('45 deg', torch.tensor([1.0, 1.0])),
                ('90 deg', torch.tensor([0.0, 3.0])), ('opposite', torch.tensor([-1.0, 0.0]))]:
    print(f'{name:8s} u.v = {(u @ v).item():5.2f}  cos = {cos(u, v):5.2f}')
print(f'x vs w: |x|={xt.norm():.4f} |w|={wt.norm():.4f} cos={cos(xt, wt):.4f}')
