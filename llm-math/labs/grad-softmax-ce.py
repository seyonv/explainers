"""Card 12, "The gradient at the top is p - y": the measurements behind it.

Uses the tiny model from tiny.py (it never changes it) and prints:
  1. dz = (p - y) / 3 for every position, and the fact that each row adds to 0
  2. the derivation for the last row, step by step: d(-ln p)/dp, the softmax Jacobian, the cancellation
  3. a nudge test straight on the logits: does the loss really move by dz x nudge?
  4. PyTorch's cross_entropy gives the same dz
  5. why frameworks fuse softmax and -log: what happens in float32 when the logits get big

Run:  python labs/grad-softmax-ce.py      (needs numpy and torch)"""
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tiny  # noqa: E402

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
v = tiny.forward(tiny.W)
z, p, T, V = v["z"], v["p"], tiny.T, tiny.V
tgt = tiny.targets
y = np.zeros((T, V)); y[np.arange(T), tgt] = 1    # one-hot targets: a 1 in the right word's column


def loss_from_logits(z):
    """The loss as a function of the logits alone: softmax each row, -ln the target's share, mean."""
    q = tiny.softmax(z)
    return -np.log(q[np.arange(T), tgt]).mean()


print("== 1. the gradient at the top: dz = (p - y) / 3")
dz = (p - y) / T
for t in range(T):
    print(f"position {t} (target {tiny.VOCAB[tgt[t]]}): p - y = {p[t] - y[t]}   / 3 = {dz[t]}"
          f"   row sum = {dz[t].sum():+.1e}")

print("\n== 2. the derivation for the last row (target 'on', column 3)")
pr, k = p[2], tgt[2]
print(f"p = {pr}")
print(f"step 1, the -ln:  dL2/dp_on = -1 / p_on = -1 / {pr[k]:.4f} = {-1 / pr[k]:.4f}")
J = np.diag(pr) - np.outer(pr, pr)                 # J[j, i] = dp_j / dz_i = p_j (delta_ij - p_i)
print("step 2, the softmax Jacobian J[j, i] = dp_j / dz_i = p_j (delta_ij - p_i), 5 x 5:")
print(J)
print(f"  row 'on' (how p_on moves as each logit moves): {J[k]}")
print(f"  each column of J adds to: {J.sum(0)}")
chain = (-1 / pr[k]) * J[k]
print(f"step 3, multiply: (-1 / p_on) x row 'on' = {chain}   (the p_on cancels: this is p - y)")
print(f"step 4, divide by 3 for the mean: {chain / T}")
print(f"plain-words reading: push 'on' up by 1 - p_on = {1 - pr[k]:.4f}; push each wrong word down by its p")

print("\n== 3. nudge test on the logits")
L0 = loss_from_logits(z)
for (r_, c_) in ((2, 3), (2, 4)):
    for eps in (0.1, 0.001):
        z2 = z.copy(); z2[r_, c_] += eps
        dL = loss_from_logits(z2) - L0
        print(f"z[{r_},{c_}] += {eps:<6}: loss {L0:.6f} -> {L0 + dL:.6f}  change {dL:+.6f}"
              f"  change/nudge {dL / eps:+.4f}   dz[{r_},{c_}] {dz[r_, c_]:+.4f}")
z3 = z.copy(); z3[2] += 1.0
print(f"add 1.0 to every logit in row 2: loss change {loss_from_logits(z3) - L0:+.1e}   (rows of dz add to 0)")

print("\n== 4. PyTorch autograd on the logits")
zt = torch.tensor(z, requires_grad=True)
lt = F.cross_entropy(zt, torch.tensor(tgt))
lt.backward()
print(f"torch loss {lt.item():.6f}   largest |torch dz - (p - y)/3| = {np.abs(zt.grad.numpy() - dz).max():.1e}")

print("\n== 5. why fuse softmax and -log: the last row of logits times 50, in float32")
big = torch.tensor(z[2:3] * 50, dtype=torch.float32, requires_grad=True)
print(f"logits = {big.detach().numpy().round(2)}   gap mat - on = {(big[0, 4] - big[0, 3]).item():.2f}")
tg = torch.tensor([tgt[2]])
p_sep = torch.softmax(big, -1)                     # separate: form p first (max-subtracted), then -log
l_sep = -torch.log(p_sep[0, tgt[2]])
l_sep.backward()
print(f"separate  softmax then -log: p_on = {p_sep[0, tgt[2]].item():.3e}   loss = {l_sep.item()}"
      f"   grad = {big.grad.numpy().round(4)}")
big.grad = None
l_fused = F.cross_entropy(big, tg)                  # fused: logsumexp(z) - z[target], never forms p
l_fused.backward()
print(f"fused     cross_entropy:     loss = {l_fused.item():.4f}   grad = {big.grad.numpy().round(4)}")
