"""The "why" numbers for the card "Train a 6-weight network on paper".
It reuses the network from tiny_net.py (which prints its own forward, backward and update first).

1. Nudge test: change one weight by a tiny amount and watch the loss. change / nudge should equal the gradient.
2. The ReLU gate: neuron 1 starts below zero, gets 0 gradient, and so never changes, however many steps you take.
3. Step size: the same gradient with different learning rates. Too small crawls, too big overshoots.

Run:  python labs/tiny-net-by-hand.py
Needs: numpy, torch (for tiny_net's own check)."""
import numpy as np
from tiny_net import x, y, W1, w2, forward

print("\n--- 1. nudge test (eps = 0.001) ---")
_, _, _, L0 = forward(W1, w2)
eps = 0.001
for name, (j, k) in [("W1[0,0] (neuron 1, gated off)", (0, 0)), ("W1[1,0]", (1, 0)), ("W1[1,1]", (1, 1))]:
    W = W1.copy(); W[j, k] += eps
    _, _, _, L = forward(W, w2)
    print(f"{name}: loss {L0:.6f} -> {L:.6f}, change/nudge = {(L - L0) / eps:.4f}")
v = w2.copy(); v[1] += eps
_, _, _, L = forward(W1, v)
print(f"w2[1]: change/nudge = {(L - L0) / eps:.4f}")


def grads(W1, w2):
    """The same chain rule as tiny_net.py: multiply local slopes from the loss back to each weight."""
    p, h, o, L = forward(W1, w2)
    dL_do = 2 * (o - y)
    dL_dp = dL_do * w2 * (p > 0)
    return np.outer(dL_dp, x) + 0.0, dL_do * h + 0.0, L   # + 0.0 turns -0.0 into 0.0


print("\n--- 2. the ReLU gate over 20 steps at lr 0.1 ---")
A, b = W1.copy(), w2.copy()
for step in range(21):
    gW1, gw2, L = grads(A, b)
    if step in (0, 1, 2, 5, 10, 20):
        p = A @ x
        print(f"step {step:2d}: L = {L:.4f} ({L:.1e})  neuron 1 weights {A[0]}  p1 = {p[0]:.4f}  its gradient {gW1[0]}")
    A, b = A - 0.1 * gW1, b - 0.1 * gw2

print("\n--- 3. one step, different learning rates ---")
gW1, gw2, L = grads(W1, w2)
for lr in (0.01, 0.1, 0.2, 0.3, 0.5):
    _, _, o, L1 = forward(W1 - lr * gW1, w2 - lr * gw2)
    print(f"lr {lr}: o = {o:.4f}  loss {L:.4f} -> {L1:.4f}")
