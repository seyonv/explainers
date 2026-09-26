# Card: value-object · Lecture 1 · https://www.youtube.com/watch?v=VMj-3S1tku0&t=1149s
# Run from the course folder: python labs/value-object.py [--plot]
from common import print_graph, save_graph, want_plot

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

print('L =', L)
print('d._prev (a set) holds:', sorted((v.label, v.data) for v in d._prev), '  d._op =', repr(d._op))
print('a*b+c without labels:', a*b + c)
print('graph of L (text draw_dot):')
print_graph(L)
if want_plot():
    save_graph(L, 'value-object')
