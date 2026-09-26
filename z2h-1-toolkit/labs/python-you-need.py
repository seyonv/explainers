# Card: python-you-need (the Python micrograd leans on) · used from L1 https://www.youtube.com/watch?v=VMj-3S1tku0&t=1149s
# Run from z2h-1-toolkit/: python labs/python-you-need.py
class Money:
    def __init__(self, cents):
        self.cents = cents
    def __repr__(self):
        return f"Money(cents={self.cents})"
    def __add__(self, other):
        other = other if isinstance(other, Money) else Money(other)
        return Money(self.cents + other.cents)
    def __mul__(self, k):
        return Money(self.cents * k)
    def __rmul__(self, k):          # k * self falls back here
        return self * k
    def __radd__(self, other):      # lets sum() start from 0
        return self + other

m1, m2 = Money(250), Money(100)
print('m1 + m2      =', m1 + m2)
print('m1 * 3       =', m1 * 3)
print('3 * m1       =', 3 * m1)
del Money.__rmul__
try:
    3 * m1
except TypeError as e:
    print('3 * m1 without __rmul__ -> TypeError:', e)

print('sum([m1, m2, m1]) =', sum([m1, m2, m1]))

# closure: a function that remembers variables from where it was made
def make_backward(name, log):
    def _backward():
        log.append(f'backward of {name}')
    return _backward
log = []
fns = [make_backward(n, log) for n in ['a', 'b', 'c']]
for f in reversed(fns):
    f()
print('closure calls:', log)

# set() + recursion: the build_topo pattern from micrograd
children = {'L': ['d', 'f'], 'd': ['e', 'c'], 'e': ['a', 'b'], 'f': [], 'c': [], 'a': [], 'b': []}
topo, visited = [], set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in children[v]:
            build_topo(child)
        topo.append(v)
build_topo('L')
print('topo order    :', topo)
print('reversed topo :', list(reversed(topo)))

print('list comprehension:', [x * x for x in range(5)], ' sum:', sum(x * x for x in range(5)))
