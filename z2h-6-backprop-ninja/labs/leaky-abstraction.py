# Card: leaky-abstraction · Lecture 5 ch.1 intro https://www.youtube.com/watch?v=q8SA3rM6ckI&t=0s
# Run from the course folder: python labs/leaky-abstraction.py   (the "clip the loss" bug, and two other leaks)
import torch
import torch.nn.functional as F

# 1) The bug from "Yes you should understand backprop": clip the error inside the loss
#    instead of clipping the gradient. Five predictions, targets all 0, so delta = prediction.
pred = torch.tensor([-3.0, -0.5, 0.2, 0.8, 2.5], requires_grad=True)
delta = pred - 0.0
loss_bug = (torch.clamp(delta, -1, 1)**2).mean()      # clip the delta, then square it
loss_bug.backward()
grad_bug = pred.grad.clone(); pred.grad = None
loss_ok = 2*F.huber_loss(pred, torch.zeros(5), delta=1.0)  # what was meant: = delta**2 inside +-1, gradient capped outside
loss_ok.backward()
print('delta                  :', [round(v, 3) for v in delta.tolist()])
print('grad, clipped delta    :', [round(v, 3) for v in grad_bug.tolist()])
print('grad, Huber (intended) :', [round(v, 3) for v in pred.grad.tolist()])
print('outliers ignored by the bug:', int((grad_bug == 0).sum()), 'of 5')

# 2) Saturated tanh: the local gradient 1 - tanh(x)**2 vanishes in the flat tails
x = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0])
print('x              :', x.tolist())
print('1 - tanh(x)**2 :', [f'{v:.2e}' for v in (1 - torch.tanh(x)**2).tolist()])

# 3) A dead ReLU: every input negative, so no gradient ever flows to its weights
w = torch.tensor([-1.0, -2.0], requires_grad=True)
X = torch.rand(100, 2)                                 # positive inputs, so X @ w < 0 always
F.relu(X @ w).sum().backward()
print('dead ReLU weight grad:', w.grad.tolist())
