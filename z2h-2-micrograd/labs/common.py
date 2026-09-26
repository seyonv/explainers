# Shared helpers for the z2h-2-micrograd labs: the lecture's finished Value engine and graph printing.
# Imported by the labs; not meant to be run directly.
import math
import sys


class Value:
    """The Value class as it stands at the end of Lecture 1 (tanh version)."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __radd__(self, other):  # other + self
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
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


def trace(root):
    nodes, edges = [], set()
    def build(v):
        if v not in nodes:
            for child in sorted(v._prev, key=lambda c: c.label):
                edges.add((child, v))
                build(child)
            nodes.append(v)
    build(root)
    return nodes, edges


def print_graph(root):
    """Text version of draw_dot: one line per node, inputs first, root last."""
    nodes, _ = trace(root)
    for n in nodes:
        kids = ', '.join(c.label or '?' for c in sorted(n._prev, key=lambda c: c.label))
        src = f"= {n._op}({kids})" if n._op else "(leaf)"
        print(f"  {n.label or '?':>14} | data {n.data:9.4f} | grad {n.grad:9.4f}  {src}")


def draw_dot(root):
    from graphviz import Digraph
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot


def want_plot():
    return '--plot' in sys.argv


def save_graph(root, name):
    """With --plot: render the graph to <name>.png next to the script (needs the graphviz `dot` binary)."""
    import os
    from graphviz.backend import ExecutableNotFound
    path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), name)
    dot = draw_dot(root)
    try:
        dot.render(path, cleanup=True)
        print(f"saved {path}.png")
    except ExecutableNotFound:
        if os.path.exists(path):
            os.remove(path)  # render() writes the source before calling dot
        dot.save(path + '.gv')
        print(f"graphviz `dot` binary not found (brew install graphviz); saved the source to {path}.gv")
