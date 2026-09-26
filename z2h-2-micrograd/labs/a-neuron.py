# Card: a-neuron · Lecture 1 · https://www.youtube.com/watch?v=VMj-3S1tku0&t=3172s
# Run from the course folder: python labs/a-neuron.py [--plot]
import math
from common import Value, print_graph, save_graph, want_plot

# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
print('n =', n.data, '  o = tanh(n) =', o.data, '  1/sqrt(2) =', 1/math.sqrt(2))
print('with b = 8 instead: tanh(', 2.0, ') =', math.tanh(2.0))

# backprop by hand
o.grad = 1.0
n.grad = 1 - o.data**2          # do/dn = 1 - tanh(n)**2
print('1 - o**2 =', n.grad)
x1w1x2w2.grad = n.grad; b.grad = n.grad          # plus routes
x1w1.grad = x1w1x2w2.grad; x2w2.grad = x1w1x2w2.grad
x2.grad = w2.data * x2w2.grad                    # times swaps
w2.grad = x2.data * x2w2.grad
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad
print_graph(o)

if want_plot():
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2))); plt.grid()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a-neuron-tanh.png')
    plt.savefig(out); print('saved', out)
    save_graph(o, 'a-neuron')
