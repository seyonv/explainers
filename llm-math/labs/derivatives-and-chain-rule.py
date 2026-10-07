"""The numbers for the card "A derivative is a nudge ratio".

1. A two-step chain, m = x * w then L = m**2, and the same graph with a second path, y = m + x.
   The chain rule's answer is checked against a nudge.
2. The loss curve around W1[0,0] in the tiny transformer, for the sketch.
3. Nudge every one of the 220 weights, one at a time, and compare with one backward pass.
4. How small can the nudge get before rounding error takes over?

Run:  python labs/derivatives-and-chain-rule.py
Needs: numpy."""
import numpy as np

import tiny

NUDGE = 0.001

print("--- 1a. a two-step chain: m = x * w, L = m**2 ---")
x, w = 1.5, 2.0
m = x * w
L = m**2
dL_dm = 2 * m          # local slope of squaring
dm_dw = x              # local slope of x * w with respect to w
print(f"forward: m = {m}  L = {L}")
print(f"chain rule: dL/dw = dL/dm * dm/dw = {dL_dm} * {dm_dw} = {dL_dm * dm_dw}")
L2 = (x * (w + NUDGE))**2
print(f"nudge w by {NUDGE}: L = {L2:.8f}  change/nudge = {(L2 - L) / NUDGE:.4f}")

print("\n--- 1b. two paths: m = x * w, y = m + x, L = y**2 ---")
def f(x, w):
    return (x * w + x)**2
y = x * w + x
L = y**2
dL_dy = 2 * y
path1 = dL_dy * 1 * w  # through + (slope 1), then through x * w (slope w)
path2 = dL_dy * 1      # straight through + (slope 1)
print(f"forward: m = {x * w}  y = {y}  L = {L}")
print(f"dL/dy = {dL_dy}   path via m: {dL_dy} * 1 * {w} = {path1}   direct path: {dL_dy} * 1 = {path2}")
print(f"dL/dx = {path1} + {path2} = {path1 + path2}    dL/dw = {dL_dy} * 1 * {x} = {dL_dy * x}")
print(f"nudge x by {NUDGE}: change/nudge = {(f(x + NUDGE, w) - L) / NUDGE:.4f}   (one path alone would say {path1} or {path2})")
print(f"nudge w by {NUDGE}: change/nudge = {(f(x, w + NUDGE) - L) / NUDGE:.4f}")

print("\n--- 1c. by hand: the same graph with x = 1, w = 3 ---")
xb, wb = 1.0, 3.0
yb = xb * wb + xb
print(f"m = {xb * wb}  y = {yb}  L = {yb**2}  dL/dy = {2 * yb}  dL/dw = {2 * yb * xb}  dL/dx = {2 * yb * wb} + {2 * yb} = {2 * yb * (wb + 1)}")

W0 = tiny.W
v0 = tiny.forward(W0)
g, _ = tiny.backward(W0, v0)
L0 = v0["loss"]


def loss_with(key, idx, delta):
    W2 = {k: a.copy() for k, a in W0.items()}
    W2[key][idx] += delta
    return tiny.forward(W2)["loss"]


print("\n--- 2. the loss as W1[0,0] moves (every other weight held still) ---")
print(f"W1[0,0] starts at {W0['W1'][0, 0]:+.1f}, loss {L0:.4f}, gradient {g['W1'][0, 0]:+.4f}")
for delta in np.linspace(-1.5, 1.5, 13):
    print(f"  W1[0,0] {W0['W1'][0, 0] + delta:+.2f}: loss {loss_with('W1', (0, 0), delta):.4f}")

print("\n--- 3. nudge all 220 weights, one at a time ---")
n, worst, passes = 0, 0.0, 0
for key, a in W0.items():
    for idx in np.ndindex(a.shape):
        est = (loss_with(key, idx, NUDGE) - L0) / NUDGE
        passes += 1
        worst = max(worst, abs(est - g[key][idx]))
        n += 1
print(f"weights nudged: {n}   extra forward passes: {passes}   largest |nudge ratio - gradient|: {worst:.4f}")
print("backprop: 1 forward + 1 backward pass gives all", n, "gradients")
pos = sum(int((g[k] > 0).sum()) for k in g)
neg = sum(int((g[k] < 0).sum()) for k in g)
print(f"signs: {pos} gradients positive, {neg} negative, {n - pos - neg} exactly 0")

print("\n--- 4. shrinking the nudge on W1[0,0] (gradient -0.4500) ---")
EPS = [0.1, 0.01, 0.001, 1e-5, 1e-8, 1e-11, 1e-13, 1e-15]
for eps in EPS:
    r = (loss_with("W1", (0, 0), eps) - L0) / eps
    print(f"  nudge {eps:.0e}: change/nudge {r:+.6f}   error {abs(r - g['W1'][0, 0]):.1e}")
