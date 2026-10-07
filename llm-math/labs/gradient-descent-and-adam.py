"""Card 16, "Step downhill, at the right size": the measurements behind it.

Uses the tiny model from tiny.py (it never changes it) and prints:
  1. one plain SGD step on one weight, and the loss along the downhill line
     (where each learning rate's first step lands)
  2. the loss after every step for 30 steps: SGD at three learning rates, SGD with momentum, Adam
  3. Adam's first two steps on two weights with very different gradients
  4. what weight decay (AdamW) and the gradient's overall size look like on this model

Run:  python labs/gradient-descent-and-adam.py      (needs numpy)"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tiny  # noqa: E402

np.seterr(all="ignore")                       # lr 10 overflows on purpose
W0 = tiny.W
loss = lambda W: tiny.forward(W)["loss"]
grads = lambda W: tiny.backward(W, tiny.forward(W))[0]
g0 = grads(W0)

# ---------------------------------------------------------------- 1. one SGD step
print("== one SGD step on Wout[1,3]")
w, g = W0["Wout"][1, 3], g0["Wout"][1, 3]
for lr in (0.1, 1.0):
    print(f"lr {lr}: {w:+.4f} - {lr} x ({g:+.4f}) = {w - lr * g:+.4f}")

print("\n== the loss along the downhill line: W - lr * g, for one step from the start")
for lr in (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0):
    print(f"lr {lr:5}: loss {loss({k: W0[k] - lr * g0[k] for k in W0}):.4f}")


# ---------------------------------------------------------------- 2. 30 steps
def run(steps, lr, opt="sgd", beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0):
    W = {k: a.copy() for k, a in W0.items()}
    m = {k: np.zeros_like(a) for k, a in W.items()}
    v = {k: np.zeros_like(a) for k, a in W.items()}
    out = [loss(W)]
    for t in range(1, steps + 1):
        g = grads(W)
        for k in W:
            if opt == "sgd":
                W[k] = W[k] - lr * g[k]
            elif opt == "momentum":                       # m is a running average of gradients
                m[k] = beta1 * m[k] + (1 - beta1) * g[k]
                W[k] = W[k] - lr * m[k]
            else:                                         # Adam / AdamW
                m[k] = beta1 * m[k] + (1 - beta1) * g[k]
                v[k] = beta2 * v[k] + (1 - beta2) * g[k] ** 2
                mh, vh = m[k] / (1 - beta1 ** t), v[k] / (1 - beta2 ** t)
                W[k] = W[k] - lr * mh / (np.sqrt(vh) + eps) - lr * wd * W[k]
        out.append(loss(W))
    return out, W


print("\n== loss after each of 30 steps")
runs = {"SGD lr 0.01": run(30, 0.01), "SGD lr 0.1": run(30, 0.1), "SGD lr 1.0": run(30, 1.0),
        "momentum lr 1.0": run(30, 1.0, "momentum"), "Adam lr 0.1": run(30, 0.1, "adam")}
for name, (L, _) in runs.items():
    print(f"{name:16}: " + " ".join(f"{x:.4f}" for x in L))

# ---------------------------------------------------------------- 3. Adam's first two steps
print("\n== Adam (lr 0.1, beta1 0.9, beta2 0.999) on two weights")
g1 = grads({k: W0[k] - 0.0 for k in W0})
_, W1 = run(1, 0.1, "adam")
g2 = grads(W1)
for key, idx in [("Wout", (1, 3)), ("Wq", (1, 0))]:
    a, b = g1[key][idx], g2[key][idx]
    m1, v1 = 0.1 * a, 0.001 * a**2
    m2, v2 = 0.9 * m1 + 0.1 * b, 0.999 * v1 + 0.001 * b**2
    s1 = 0.1 * (m1 / 0.1) / (np.sqrt(v1 / 0.001) + 1e-8)
    s2 = 0.1 * (m2 / (1 - 0.9**2)) / (np.sqrt(v2 / (1 - 0.999**2)) + 1e-8)
    print(f"{key}{list(idx)}: start {W0[key][idx]:+.4f}")
    print(f"  step 1: g {a:+.5f}  m {m1:+.6f}  v {v1:.3e}  m_hat {m1/0.1:+.5f}  v_hat {v1/0.001:.3e}  step {-s1:+.4f}")
    print(f"  step 2: g {b:+.5f}  m {m2:+.6f}  v {v2:.3e}  m_hat {m2/(1-0.9**2):+.5f}  v_hat {v2/(1-0.999**2):.3e}  step {-s2:+.4f}")
    print(f"  SGD lr 0.1 would move it {-0.1 * a:+.5f}; ratio Adam / SGD = {s1 / (0.1 * a):.1f}")

# ---------------------------------------------------------------- 4. AdamW and gradient size
print("\n== weight decay: AdamW lr 0.1, 30 steps")
for wd in (0.0, 0.1):
    L, W = run(30, 0.1, "adam", wd=wd)
    norm = np.sqrt(sum((W[k] ** 2).sum() for k in W))
    print(f"weight decay {wd}: loss after 30 steps {L[-1]:.4f}   size of all 220 weights {norm:.4f}")
print(f"size of all 220 weights at the start: {np.sqrt(sum((W0[k] ** 2).sum() for k in W0)):.4f}")
print(f"\nsize of the whole gradient at step 0, sqrt(sum of g^2) over 220 weights: {np.sqrt(sum((g0[k] ** 2).sum() for k in g0)):.4f}")
