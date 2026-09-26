# Card: minibatches · Lecture 3 ch.11 https://www.youtube.com/watch?v=TCH_1BHY58I&t=2485s
# Run from the course folder: python labs/minibatches.py
import time
import torch
import torch.nn.functional as F
from common import words, build_dataset, init_params, forward, split_loss, train

X, Y = build_dataset(words)  # every name, no split yet (as in this chapter)
print('X.shape', tuple(X.shape))
g, parameters = init_params(n_embd=2, n_hidden=100)


def step(Xb, Yb):
    loss = F.cross_entropy(forward(parameters, Xb), Yb)
    for p in parameters:
        p.grad = None
    loss.backward()
    return loss


t0 = time.perf_counter()
for _ in range(10):
    step(X, Y)
full_ms = (time.perf_counter() - t0) / 10 * 1000
t0 = time.perf_counter()
for _ in range(1000):
    ix = torch.randint(0, X.shape[0], (32,), generator=g)
    step(X[ix], Y[ix])
mini_ms = (time.perf_counter() - t0) / 1000 * 1000
print(f'one full-batch step (228,146 rows): {full_ms:.1f} ms | one minibatch step (32 rows): {mini_ms:.3f} ms'
      f' | ratio {full_ms / mini_ms:.0f}x')

g, parameters = init_params(n_embd=2, n_hidden=100)  # restart from the same init
print(f'full-set loss at init: {split_loss(parameters, X, Y):.4f}')
done = 0
for chunk in [100, 900, 1000, 3000, 5000]:
    lossi = train(parameters, X, Y, chunk, lambda i: 0.1, g)
    done += chunk
    print(f'after {done:6d} steps (lr 0.1): last minibatch loss {lossi[-1]:.4f} | full-set loss {split_loss(parameters, X, Y):.4f}')
