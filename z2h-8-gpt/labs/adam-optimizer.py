# Card: adam-optimizer.html · Lecture 7 · training the bigram model (AdamW appears here) · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2093s
# Run from the course folder: python labs/adam-optimizer.py   (Adam by hand vs torch.optim, Kingma & Ba 2014; AdamW, Loshchilov & Hutter 2017)
import math
import torch

grads = [1.0, 0.5, -0.2]          # the gradient we pretend to see at steps 1, 2, 3
lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
p0 = 1.0                          # starting value of the one parameter

print(f"one parameter p = {p0}, gradients {grads}, lr {lr}, betas ({beta1}, {beta2}), eps {eps}")
print("-- Adam by hand --")
p, m, v = p0, 0.0, 0.0
for t, g in enumerate(grads, start=1):
    m = beta1 * m + (1 - beta1) * g          # running mean of gradients (momentum)
    v = beta2 * v + (1 - beta2) * g * g      # running mean of squared gradients
    m_hat = m / (1 - beta1**t)               # bias correction
    v_hat = v / (1 - beta2**t)
    step = lr * m_hat / (math.sqrt(v_hat) + eps)
    p = p - step
    print(f"t={t}  g={g:+.1f}  m={m:.6f}  v={v:.8f}  m_hat={m_hat:.6f}  v_hat={v_hat:.6f}  step={step:.6f}  p={p:.6f}")

def run(opt_cls, **kw):
    w = torch.tensor([p0], dtype=torch.float64, requires_grad=True)
    opt = opt_cls([w], lr=lr, **kw)
    out = []
    for g in grads:
        opt.zero_grad()
        (w * g).sum().backward()             # d(w*g)/dw = g, so w.grad == g
        opt.step()
        out.append(w.item())
    return out

fmt = lambda xs: '  '.join(f"{x:.6f}" for x in xs)
print("-- torch --")
print("torch.optim.Adam(betas=(0.9,0.999))       p after each step:", fmt(run(torch.optim.Adam, betas=(beta1, beta2), eps=eps)))
print("plain SGD, same lr                       p after each step:", fmt(run(torch.optim.SGD)))
print("torch.optim.AdamW default weight_decay   :", torch.optim.AdamW([torch.zeros(1, requires_grad=True)]).defaults['weight_decay'])
print("AdamW(weight_decay=0.01)                 p after each step:", fmt(run(torch.optim.AdamW, betas=(beta1, beta2), eps=eps, weight_decay=0.01)))
print("AdamW(weight_decay=0.1)                  p after each step:", fmt(run(torch.optim.AdamW, betas=(beta1, beta2), eps=eps, weight_decay=0.1)))
# AdamW by hand for step 1: first shrink p by lr*wd, then take the Adam step
print(f"AdamW step 1 by hand (wd 0.01): p = {p0} * (1 - {lr}*0.01) - 0.1 = {p0 * (1 - lr * 0.01) - 0.1:.6f}")

# why the first Adam step is ~lr regardless of gradient size
print("-- first step size for different gradient scales (Adam vs SGD, lr 0.1) --")
for g in [1e-3, 1.0, 1e3]:
    w = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    (w * g).sum().backward(); opt.step()
    print(f"g = {g:g}:  Adam step = {-w.item():.6f}   SGD step = {lr * g:g}")
