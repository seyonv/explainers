"""The smallest network worth training by hand: 2 inputs, 2 hidden neurons (ReLU), 1 output, 6 weights.
One forward pass, the loss, the backward pass by the chain rule, one update, and the loss again.
Every number is checked against PyTorch autograd at the end.

Run:  python labs/tiny_net.py
Needs: numpy, torch."""
import numpy as np

x = np.array([1.0, 2.0])                 # input
y = 1.0                                  # target
W1 = np.array([[0.5, -0.5],              # hidden layer: row j holds neuron j's two weights
               [0.3, 0.2]])
w2 = np.array([0.4, 0.6])                # output weights
LR = 0.1


def forward(W1, w2):
    p = W1 @ x                           # each hidden neuron: a dot product with x
    h = np.maximum(0, p)                 # ReLU: keep positives, zero the rest
    o = w2 @ h                           # output: another dot product
    L = (o - y) ** 2                     # squared error
    return p, h, o, L


p, h, o, L = forward(W1, w2)
print(f"forward:  p = W1 x = {p}   h = ReLU(p) = {h}   o = w2 . h = {o:.4f}   L = (o - y)^2 = {L:.4f}")

# backward: start at the loss and multiply local slopes, one step at a time
dL_do = 2 * (o - y)
dL_dw2 = dL_do * h + 0.0                # o = w2 . h, so do/dw2 = h
dL_dh = dL_do * w2                      # and do/dh = w2
dL_dp = dL_dh * (p > 0) + 0.0           # ReLU's slope: 1 where p > 0, else 0
dL_dW1 = np.outer(dL_dp, x) + 0.0       # p_j = W1[j] . x, so dp_j/dW1[j] = x
print(f"backward: dL/do = {dL_do:.4f}   dL/dw2 = {dL_dw2}   dL/dh = {dL_dh}   dL/dp = {dL_dp}")
print(f"          dL/dW1 =\n{dL_dW1}")

W1_new, w2_new = W1 - LR * dL_dW1, w2 - LR * dL_dw2
_, _, o_new, L_new = forward(W1_new, w2_new)
print(f"update (lr {LR}): W1 =\n{W1_new}\n  w2 = {w2_new}")
print(f"after one step: o = {o_new:.4f}   L = {L_new:.4f}   (was {L:.4f})")

import torch
tW1 = torch.tensor(W1, requires_grad=True); tw2 = torch.tensor(w2, requires_grad=True)
tL = (tw2 @ torch.relu(tW1 @ torch.tensor(x)) - y) ** 2
tL.backward()
print(f"PyTorch: loss {tL.item():.4f}, largest gradient difference "
      f"{max(np.abs(tW1.grad.numpy() - dL_dW1).max(), np.abs(tw2.grad.numpy() - dL_dw2).max()):.1e}")
