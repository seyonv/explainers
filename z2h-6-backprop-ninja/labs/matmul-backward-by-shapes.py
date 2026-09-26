# Card: matmul-backward-by-shapes · Lecture 5 ch.3 exercise 1 (0:41-0:53) https://www.youtube.com/watch?v=q8SA3rM6ckI&t=2499s
# Run from the course folder: python labs/matmul-backward-by-shapes.py
import torch
from common import chunked_forward, cmp

# The paper example: D = A @ B + c with 2x2 matrices, L = (D * dD).sum() so that D.grad = dD
A = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
B = torch.tensor([[5., 6.], [7., 8.]], requires_grad=True)
c = torch.tensor([0.5, -0.5], requires_grad=True)
dD = torch.tensor([[1., 0.], [2., -1.]])
D = A @ B + c
(D * dD).sum().backward()
print('dA = dD @ B.T :', (dD @ B.T).tolist(), ' autograd:', A.grad.tolist())
print('dB = A.T @ dD :', (A.T @ dD).tolist(), ' autograd:', B.grad.tolist())
print('dc = dD.sum(0):', dD.sum(0).tolist(), ' autograd:', c.grad.tolist())

# In the net: logits = h @ W2 + b2. Take dlogits as given (autograd's, so only this layer is tested).
d = chunked_forward()
h, W2, b2, logits = d['h'], d['W2'], d['b2'], d['logits']
dlogits = logits.grad
print('shapes: dlogits', tuple(dlogits.shape), 'h', tuple(h.shape), 'W2', tuple(W2.shape), 'b2', tuple(b2.shape))
dh = dlogits @ W2.T        # (32,27) @ (27,64) -> (32,64) = h.shape
dW2 = h.T @ dlogits        # (64,32) @ (32,27) -> (64,27) = W2.shape
db2 = dlogits.sum(0)       # (32,27) -> (27,): b2 was broadcast over the 32 rows
cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)
try:
    h.T @ W2               # the wrong guess: shapes refuse to line up
except RuntimeError as e:
    print('wrong order h.T @ W2 ->', str(e).split('\n')[0])
