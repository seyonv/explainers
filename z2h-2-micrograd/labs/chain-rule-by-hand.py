# Card: chain-rule-by-hand · Lecture 1 · https://www.youtube.com/watch?v=VMj-3S1tku0&t=1930s
# Run from the course folder: python labs/chain-rule-by-hand.py
from common import Value, print_graph

def build(bump=None, h=0.0):
    a = Value(2.0, label='a'); b = Value(-3.0, label='b'); c = Value(10.0, label='c')
    f = Value(-2.0, label='f')
    leaves = {'a': a, 'b': b, 'c': c, 'f': f}
    if bump in leaves: leaves[bump].data += h
    e = a*b; e.label = 'e'
    if bump == 'e': e.data += h
    d = e + c; d.label = 'd'
    if bump == 'd': d.data += h
    L = d * f; L.label = 'L'
    return dict(leaves, e=e, d=d, L=L)

v = build()
# fill grads by hand, walking back from L
v['L'].grad = 1.0
v['d'].grad = v['f'].data * v['L'].grad          # L = d*f  -> dL/dd = f
v['f'].grad = v['d'].data * v['L'].grad          #          -> dL/df = d
v['c'].grad = 1.0 * v['d'].grad                  # d = e+c  -> plus routes
v['e'].grad = 1.0 * v['d'].grad
v['a'].grad = v['b'].data * v['e'].grad          # e = a*b  -> times swaps
v['b'].grad = v['a'].data * v['e'].grad
print_graph(v['L'])

def lol(name, h=0.001):
    # numerical gradient check: bump one node by h, see how far L moves
    L1 = build()['L'].data
    L2 = build(name, h)['L'].data
    return (L2 - L1)/h

print('numerical check (h=0.001):')
for name in ['a', 'b', 'c', 'e', 'd', 'f']:
    print(f'  dL/d{name}: by hand {v[name].grad:5.1f}   lol() {lol(name)}')
