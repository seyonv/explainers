# Card: more-ops-and-pytorch · Lecture 1 · https://www.youtube.com/watch?v=VMj-3S1tku0&t=5225s (and &t=5971s)
# Run from the course folder: python labs/more-ops-and-pytorch.py [--plot]
# The new ops (exp, __pow__, __truediv__, __neg__, __sub__, __radd__, __rmul__) live in labs/common.py.
import torch
from common import Value, trace, save_graph, want_plot

a = Value(2.0)
b = Value(4.0)
print('a + 1 =', a + 1, '   2 * a =', 2 * a, '   a / b =', a / b, '   a - b =', a - b, '   a.exp() =', a.exp())

def neuron(split_tanh):
    x1 = Value(2.0, label='x1'); x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1'); w2 = Value(1.0, label='w2')
    b = Value(6.8813735870195432, label='b')
    n = x1*w1 + x2*w2 + b
    if split_tanh:
        e = (2*n).exp()
        o = (e - 1) / (e + 1)
    else:
        o = n.tanh()
    o.backward()
    return o, x1, x2, w1, w2

for split in [False, True]:
    o, x1, x2, w1, w2 = neuron(split)
    name = 'tanh from exp, -, +, /' if split else 'tanh as one op        '
    print(f'{name}: o {o.data:.4f}  x2 {x2.grad:.4f}  w2 {w2.grad:.4f}  x1 {x1.grad:.4f}  w1 {w1.grad:.4f}   nodes {len(trace(o)[0])}')

x1 = torch.Tensor([2.0]).double()                ; x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()                ; x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()               ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()                ; w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double()  ; b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print('--- PyTorch (torch', torch.__version__ + ') ---')
print(o.data.item())
o.backward()
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())

# torch.Tensor([...]) is float32 first; building in float64 directly removes the tiny error
b64 = torch.tensor([6.8813735870195432], dtype=torch.double)
print('float32 b, then .double():', b.item(), '   float64 from the start:', b64.item())
x1 = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
o64 = torch.tanh(x1*-3.0 + 0.0*1.0 + b64); o64.backward()
print('all-float64 o', o64.item(), '  x1.grad', x1.grad.item())
print('x2.requires_grad default for a new tensor:', torch.Tensor([0.0]).requires_grad)

if want_plot():
    o, *_ = neuron(True)
    save_graph(o, 'more-ops-tanh-split')
