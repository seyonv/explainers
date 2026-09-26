# Card: derivative-as-nudge · Lecture 1 · https://www.youtube.com/watch?v=VMj-3S1tku0&t=488s
# Run from the course folder: python labs/derivative-as-nudge.py [--plot]
import sys

def f(x):
    return 3*x**2 - 4*x + 5

print('f(3.0) =', f(3.0))

h = 0.001
for x in [3.0, -3.0, 2/3]:
    slope = (f(x + h) - f(x))/h
    print(f'x = {x:.4f}   slope (h={h}) = {slope:.6f}   exact 6x-4 = {6*x - 4:.4f}')

# Karpathy's notebook cell: x = 2/3 with a much smaller h
h = 0.000001
x = 2/3
print('x = 2/3, h = 1e-6:', (f(x + h) - f(x))/h)

# too small an h: floating point eats the answer
print('slope at x=3 for shrinking h:')
for h in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]:
    print(f'  h = {h:.0e}   slope = {(f(3.0 + h) - f(3.0))/h}')

if '--plot' in sys.argv:
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = np.arange(-5, 5, 0.25)
    plt.plot(xs, f(xs)); plt.grid()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'derivative-as-nudge.png')
    plt.savefig(out); print('saved', out)
