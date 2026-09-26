# Card: accumulate-gradients · Lecture 1 · https://www.youtube.com/watch?v=VMj-3S1tku0&t=4948s
# Run from the course folder: python labs/accumulate-gradients.py
from common import Value as Fixed      # the engine with +=

class Buggy(Fixed):
    """Same Value, but + and * *set* the grad (=) instead of adding to it (+=)."""
    def __add__(self, other):
        out = Buggy(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Buggy(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out

for Value in [Buggy, Fixed]:
    print(f'--- {Value.__name__} ---')
    a = Value(3.0, label='a')
    b = a + a   ; b.label = 'b'
    b.backward()
    print(f'b = a + a = {b.data}   a.grad = {a.grad}   (should be 2)')

    a = Value(-2.0, label='a')
    b = Value(3.0, label='b')
    d = a * b    ; d.label = 'd'
    e = a + b    ; e.label = 'e'
    f = d * e    ; f.label = 'f'
    f.backward()
    print(f'f = (a*b)*(a+b) = {f.data}   a.grad = {a.grad}   b.grad = {b.grad}   (should be -3, -8)')

# numerical check of the second example
h = 1e-6
fn = lambda a, b: (a*b)*(a+b)
print('numerical: df/da =', round((fn(-2+h, 3) - fn(-2, 3))/h, 4), '  df/db =', round((fn(-2, 3+h) - fn(-2, 3))/h, 4))
