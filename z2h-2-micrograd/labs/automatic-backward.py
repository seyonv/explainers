# Card: automatic-backward · Lecture 1 · https://www.youtube.com/watch?v=VMj-3S1tku0&t=4142s (and &t=4652s)
# Run from the course folder: python labs/automatic-backward.py [--plot]
import math
from common import print_graph, save_graph, trace, want_plot

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None     # leaves have nothing to pass on
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        return topo

def neuron():
    x1 = Value(2.0, label='x1'); x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1'); w2 = Value(1.0, label='w2')
    b = Value(6.8813735870195432, label='b')
    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    return o, [o, n, b, x1w1x2w2, x2w2, x1w1]

# 1) forgot the base case: o.grad stays 0, so every _backward multiplies by 0
o, nodes = neuron()
for node in nodes: node._backward()
print('without o.grad = 1.0, every grad:', sorted({v.grad for v in trace(o)[0]}))

# 2) call each _backward by hand, in order
o, nodes = neuron()
o.grad = 1.0
for node in nodes: node._backward()
print('by hand, calling _backward() node by node:')
print_graph(o)

# 3) one call does it all
o, _ = neuron()
topo = o.backward()
print('topological order (inputs first, o last):')
print('  ' + ', '.join(f"{v.label}={v.data:.4f}" for v in topo))
print('after o.backward():')
print_graph(o)

# the on-camera bug: storing the *result* of _backward() instead of the function
try:
    a = Value(1.0); a._backward = (lambda: None)()   # _backward() instead of _backward
    a._backward()
except TypeError as err:
    print('bug demo:', err)

if want_plot():
    save_graph(o, 'automatic-backward')
