# Card: mlp-and-training-loop · Lecture 1 · https://www.youtube.com/watch?v=VMj-3S1tku0&t=6235s (to &t=8043s)
# Run from the course folder: python labs/mlp-and-training-loop.py [--plot]
import random
from common import Value, want_plot

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)   # w * x + b
        return act.tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

def train(zero_grad, steps=20, lr=0.1, verbose=True):
    random.seed(1337)
    n = MLP(3, [4, 4, 1])
    losses = []
    for k in range(steps):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        # backward pass
        if zero_grad:
            for p in n.parameters():
                p.grad = 0.0
        loss.backward()
        # update
        for p in n.parameters():
            p.data += -lr * p.grad
        losses.append(loss.data)
        if verbose: print(k, loss.data)
    return n, losses

random.seed(1337)
n = MLP(3, [4, 4, 1])
print('number of parameters:', len(n.parameters()), '= (3*4+4) + (4*4+4) + (4*1+1) =', (3*4+4) + (4*4+4) + (4*1+1))
print('n([2.0, 3.0, -1.0]) =', n([2.0, 3.0, -1.0]), '   (random init, random.seed(1337))')
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print('initial ypred:', [round(y.data, 4) for y in ypred], '  loss:', loss.data)
loss.backward()
w = n.layers[0].neurons[0].w[0]
print(f'layer0 neuron0 w[0]: data {w.data:.4f}  grad {w.grad:.4f}')

print('--- training, 20 steps, lr 0.1, with zero_grad ---')
n, losses = train(zero_grad=True)
print('final ypred:', [round(n(x).data, 4) for x in xs])

print('--- same seed, forgot zero_grad (grads pile up across steps) ---')
_, bad = train(zero_grad=False, verbose=False)
for k in [0, 1, 2, 5, 10, 19]:
    print(f'step {k:2d}: zero_grad {losses[k]:.6f}   forgot {bad[k]:.6f}')

if want_plot():
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(losses, label='zero_grad'); plt.plot(bad, label='forgot zero_grad'); plt.legend(); plt.yscale('log')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp-and-training-loop.png')
    plt.savefig(out); print('saved', out)
