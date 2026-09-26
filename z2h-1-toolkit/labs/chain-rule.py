# Card: chain-rule (dz/dx = dz/dy * dy/dx) · L1 https://www.youtube.com/watch?v=VMj-3S1tku0&t=1930s
# Run from z2h-1-toolkit/: python labs/chain-rule.py
import math, torch

x = 0.5
y = 2*x + 1            # inner
z = math.tanh(y)       # outer
dy_dx = 2.0
dz_dy = 1 - z**2
print(f'y = 2x+1 = {y}')
print(f'z = tanh(y) = {z:.6f}')
print(f'dy/dx = {dy_dx}')
print(f'dz/dy = 1 - tanh(y)^2 = {dz_dy:.6f}')
print(f'dz/dx = dz/dy * dy/dx = {dz_dy * dy_dx:.6f}')

h = 1e-6
num = (math.tanh(2*(x + h) + 1) - math.tanh(2*x + 1)) / h
print(f'numeric (h=1e-6)      = {num:.6f}')

xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
torch.tanh(2*xt + 1).backward()
print(f'torch autograd x.grad = {xt.grad.item():.6f}')

# car / bike / walker: car is 2x the bike, bike is 4x the walker -> car is 2*4 = 8x the walker
print('car vs walker = 2 * 4 =', 2 * 4)

# multi-path: x feeds two branches, the gradients ADD
# q = x*x uses x twice; dq/dx = x + x = 2x
xt = torch.tensor(3.0, requires_grad=True)
q = xt * xt
q.backward()
print(f'q = x*x at x=3: x.grad = {xt.grad.item()} (= 3 + 3, one per path)')
# micrograd's L1 bug example: b = a + a  -> a.grad must be 2, not 1
a = torch.tensor(3.0, requires_grad=True)
b = a + a
b.backward()
print(f'b = a + a: a.grad = {a.grad.item()}  (overwriting instead of += would give 1.0)')
