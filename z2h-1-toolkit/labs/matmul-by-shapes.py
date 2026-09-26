# Card: matmul-by-shapes ((n,k)@(k,m) -> (n,m)) · L2 https://www.youtube.com/watch?v=PaCmpygFfXo&t=4433s
# Run from z2h-1-toolkit/: python labs/matmul-by-shapes.py
import torch

A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])          # (2,3)
B = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., -1.]])             # (3,2)
C = A @ B
print('A', tuple(A.shape), '@ B', tuple(B.shape), '->', tuple(C.shape))
print(C)
print('C[0,0] = 1*1 + 2*0 + 3*1 =', 1*1 + 2*0 + 3*1)
print('C[1,1] = 4*0 + 5*1 + 6*(-1) =', 4*0 + 5*1 + 6*(-1))
print('entry = row of A . column of B:', torch.dot(A[1], B[:, 1]).item())
try:
    B @ B
except RuntimeError as e:
    print('B @ B (3,2)@(3,2) ->', str(e).splitlines()[0])

# a batch of inputs through one layer is one matmul
g = torch.Generator().manual_seed(2147483647)
X = torch.randn(32, 6, generator=g)          # 32 examples, 6 features
W = torch.randn(6, 100, generator=g)         # layer: 6 in, 100 neurons
b = torch.randn(100, generator=g)
H = X @ W + b
print('X', tuple(X.shape), '@ W', tuple(W.shape), '+ b', tuple(b.shape), '->', tuple(H.shape))
loop = torch.stack([X[i] @ W + b for i in range(32)])
print('same as looping over the 32 rows:', torch.allclose(H, loop, atol=1e-5))

# batched matmul (WaveNet preview)
x = torch.randn(4, 5, 80)
W = torch.randn(80, 200)
print('(4,5,80) @ (80,200) ->', tuple((x @ W).shape))

# parameters in a Linear layer: fan_in*fan_out + fan_out
for fan_in, fan_out in [(6, 100), (30, 200), (200, 27)]:
    lin = torch.nn.Linear(fan_in, fan_out)
    n = sum(p.numel() for p in lin.parameters())
    print(f'Linear({fan_in},{fan_out}): {fan_in}*{fan_out} + {fan_out} = {fan_in*fan_out + fan_out}  (torch counts {n})')
