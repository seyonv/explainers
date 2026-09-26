# Card: one-step-along-gradient · Lecture 1 · https://www.youtube.com/watch?v=VMj-3S1tku0&t=3070s
# Run from the course folder: python labs/one-step-along-gradient.py
from common import Value

a = Value(2.0, label='a'); b = Value(-3.0, label='b'); c = Value(10.0, label='c')
f = Value(-2.0, label='f')
e = a*b; d = e + c; L = d * f
print('before:  L =', L.data)
# the grads from the chain-rule card
a.grad, b.grad, c.grad, f.grad = 6.0, -4.0, -2.0, 4.0

a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad
print(f'leaves now: a {a.data}  b {b.data}  c {c.data}  f {f.data}')

e = a * b
d = e + c
L = d * f
print('after one step (+0.01*grad):  L =', L.data)

# the other direction: step against the gradient to make L smaller
a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
for x, g in [(a, 6.0), (b, -4.0), (c, -2.0), (f, 4.0)]:
    x.data -= 0.01 * g
print('after one step (-0.01*grad):  L =', ((a*b + c) * f).data)
