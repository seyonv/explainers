"""The tiny model behind every card in "The math of an LLM".

A one-block GPT, built the way Raschka builds it in chapter 4, at a size you can do by hand:
vocab 5, 3 tokens of context, d_model 4, 1 attention head, feed-forward hidden size 8.
This script runs one full training step in plain numpy (forward, loss, backward, update)
and checks every number against PyTorch's own modules and autograd.

Run:  python labs/tiny.py            prints every intermediate value, card by card
      python labs/tiny.py --json f   also writes all values to f
Needs: numpy, torch."""
import json
import sys

import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

# ---------------------------------------------------------------- the data
VOCAB = ["the", "cat", "sat", "on", "mat"]
inputs = np.array([0, 1, 2])    # "the cat sat"
targets = np.array([1, 2, 3])   # "cat sat on": each position predicts the next token
T, V, D, H = 3, 5, 4, 8         # tokens, vocab size, d_model, feed-forward hidden size

# ---------------------------------------------------------------- the weights
# Matrices start random, rounded to 1 decimal so you can multiply them by hand.
# Biases start at 0 and LayerNorm scales at 1, shifts at 0: PyTorch's own starting values.
# Convention: rows are tokens, so a layer is  x @ W  with W shaped (in, out).
# (PyTorch's nn.Linear stores the transpose, W.T, shaped (out, in).)
rng = np.random.default_rng(7)
def r(*shape):
    return np.round(rng.uniform(-1, 1, shape), 1) + 0.0   # + 0.0 turns -0.0 into 0.0

W = {
    "E": r(V, D),        # token embedding table: one row per token
    "P": r(T, D),        # position embedding table: one row per position (learned, as in GPT-2)
    "g1": np.ones(D), "b1": np.zeros(D),           # LayerNorm 1
    "Wq": r(D, D), "Wk": r(D, D), "Wv": r(D, D),   # attention projections, no bias (qkv_bias=False)
    "Wo": r(D, D), "bo": np.zeros(D),              # attention output projection
    "g2": np.ones(D), "b2": np.zeros(D),           # LayerNorm 2
    "W1": r(D, H), "c1": np.zeros(H),              # feed-forward up: 4 -> 8
    "W2": r(H, D), "c2": np.zeros(D),              # feed-forward down: 8 -> 4
    "gf": np.ones(D), "bf": np.zeros(D),           # final LayerNorm
    "Wout": r(D, V),                               # output head: 4 numbers -> 5 scores
}
EPS = 1e-5
GELU_C = np.sqrt(2 / np.pi)


# ---------------------------------------------------------------- forward pieces
def layernorm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)          # divides by D, not D-1 (Raschka's unbiased=False)
    xhat = (x - mu) / np.sqrt(var + EPS)
    return xhat * g + b, (xhat, np.sqrt(var + EPS))


def softmax(z):
    z = z - z.max(-1, keepdims=True)        # subtract the row max first: same answer, no overflow
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def gelu(u):                                 # the tanh approximation GPT-2 and Raschka use
    return 0.5 * u * (1 + np.tanh(GELU_C * (u + 0.044715 * u**3)))


def forward(W):
    v = {}
    v["x0"] = W["E"][inputs] + W["P"]                          # embeddings: look up 3 rows, add positions
    v["a"], v["ln1"] = layernorm(v["x0"], W["g1"], W["b1"])
    v["Q"], v["K"], v["V"] = v["a"] @ W["Wq"], v["a"] @ W["Wk"], v["a"] @ W["Wv"]
    v["S_raw"] = v["Q"] @ v["K"].T                             # every query against every key
    v["S"] = v["S_raw"] / np.sqrt(D)                           # scale by sqrt(d_head) = 2
    mask = np.triu(np.ones((T, T), bool), k=1)                 # True above the diagonal = the future
    v["S_masked"] = np.where(mask, -np.inf, v["S"])
    v["A"] = softmax(v["S_masked"])                            # attention weights, each row sums to 1
    v["C"] = v["A"] @ v["V"]                                   # weighted mix of value vectors
    v["o"] = v["C"] @ W["Wo"] + W["bo"]
    v["x1"] = v["x0"] + v["o"]                                 # residual: add, don't replace
    v["bn"], v["ln2"] = layernorm(v["x1"], W["g2"], W["b2"])
    v["u"] = v["bn"] @ W["W1"] + W["c1"]
    v["gu"] = gelu(v["u"])
    v["f"] = v["gu"] @ W["W2"] + W["c2"]
    v["x2"] = v["x1"] + v["f"]                                 # residual again
    v["h"], v["lnf"] = layernorm(v["x2"], W["gf"], W["bf"])
    v["z"] = v["h"] @ W["Wout"]                                # logits: one score per vocab word
    v["p"] = softmax(v["z"])
    v["nll"] = -np.log(v["p"][np.arange(T), targets])          # loss at each position
    v["loss"] = v["nll"].mean()
    return v


# ---------------------------------------------------------------- backward pieces
def layernorm_back(dy, g, cache):
    xhat, sigma = cache
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / sigma
    return dx, (dy * xhat).sum(0), dy.sum(0)


def gelu_back(u):
    t = np.tanh(GELU_C * (u + 0.044715 * u**3))
    return 0.5 * (1 + t) + 0.5 * u * (1 - t**2) * GELU_C * (1 + 3 * 0.044715 * u**2)


def backward(W, v):
    g, d = {}, {}
    # loss -> logits: softmax + cross-entropy collapse to (p - y), divided by T for the mean
    y = np.zeros((T, V)); y[np.arange(T), targets] = 1
    d["z"] = (v["p"] - y) / T
    # output head: z = h @ Wout
    g["Wout"] = v["h"].T @ d["z"]
    d["h"] = d["z"] @ W["Wout"].T
    d["x2"], g["gf"], g["bf"] = layernorm_back(d["h"], W["gf"], v["lnf"])
    # residual 2: x2 = x1 + f, so the gradient copies to both branches
    d["f"] = d["x2"]
    g["W2"], g["c2"] = v["gu"].T @ d["f"], d["f"].sum(0)
    d["gu"] = d["f"] @ W["W2"].T
    d["u"] = d["gu"] * gelu_back(v["u"])
    g["W1"], g["c1"] = v["bn"].T @ d["u"], d["u"].sum(0)
    d["bn"] = d["u"] @ W["W1"].T
    d_ln2, g["g2"], g["b2"] = layernorm_back(d["bn"], W["g2"], v["ln2"])
    d["x1"] = d["x2"] + d_ln2                    # highway + the branch through the FFN
    # residual 1: x1 = x0 + o
    d["o"] = d["x1"]
    g["Wo"], g["bo"] = v["C"].T @ d["o"], d["o"].sum(0)
    d["C"] = d["o"] @ W["Wo"].T
    # C = A @ V
    d["A"] = d["C"] @ v["V"].T
    d["V"] = v["A"].T @ d["C"]
    # softmax backward, row by row: dS = A * (dA - sum(dA * A))
    d["S"] = v["A"] * (d["A"] - (d["A"] * v["A"]).sum(-1, keepdims=True))
    d["S_raw"] = d["S"] / np.sqrt(D)
    d["Q"] = d["S_raw"] @ v["K"]
    d["K"] = d["S_raw"].T @ v["Q"]
    g["Wq"], g["Wk"], g["Wv"] = v["a"].T @ d["Q"], v["a"].T @ d["K"], v["a"].T @ d["V"]
    d["a"] = d["Q"] @ W["Wq"].T + d["K"] @ W["Wk"].T + d["V"] @ W["Wv"].T
    d_ln1, g["g1"], g["b1"] = layernorm_back(d["a"], W["g1"], v["ln1"])
    d["x0"] = d["x1"] + d_ln1
    # embeddings: each looked-up row gets its token's gradient; positions get theirs directly
    g["E"] = np.zeros_like(W["E"]); np.add.at(g["E"], inputs, d["x0"])
    g["P"] = d["x0"].copy()
    return g, d


# ---------------------------------------------------------------- the check against PyTorch
def torch_check(W):
    """Build the same model from PyTorch's own modules, run autograd, return its loss and grads."""
    import torch
    import torch.nn as nn
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    emb_tok, emb_pos = nn.Embedding(V, D).double(), nn.Embedding(T, D).double()
    ln1, ln2, lnf = (nn.LayerNorm(D, eps=EPS).double() for _ in range(3))
    q, k, val = (nn.Linear(D, D, bias=False).double() for _ in range(3))
    out_proj, up, down = nn.Linear(D, D).double(), nn.Linear(D, H).double(), nn.Linear(H, D).double()
    head = nn.Linear(D, V, bias=False).double()
    with torch.no_grad():                        # nn.Linear stores W.T, so load the transposes
        emb_tok.weight.copy_(t(W["E"])); emb_pos.weight.copy_(t(W["P"]))
        for m, wk in [(q, "Wq"), (k, "Wk"), (val, "Wv"), (out_proj, "Wo"), (up, "W1"), (down, "W2"), (head, "Wout")]:
            m.weight.copy_(t(W[wk]).T)
        for m, bk in [(out_proj, "bo"), (up, "c1"), (down, "c2")]:
            m.bias.copy_(t(W[bk]))
    x0 = emb_tok(torch.tensor(inputs)) + emb_pos(torch.arange(T))
    a = ln1(x0)
    S = q(a) @ k(a).T / D**0.5
    S = S.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    x1 = x0 + out_proj(torch.softmax(S, -1) @ val(a))
    x2 = x1 + down(nn.functional.gelu(up(ln2(x1)), approximate="tanh"))
    loss = nn.functional.cross_entropy(head(lnf(x2)), torch.tensor(targets))
    loss.backward()
    pairs = {"E": emb_tok.weight.grad, "P": emb_pos.weight.grad,
             "Wq": q.weight.grad.T, "Wk": k.weight.grad.T, "Wv": val.weight.grad.T,
             "Wo": out_proj.weight.grad.T, "bo": out_proj.bias.grad,
             "W1": up.weight.grad.T, "c1": up.bias.grad, "W2": down.weight.grad.T, "c2": down.bias.grad,
             "Wout": head.weight.grad.T,
             "g1": ln1.weight.grad, "b1": ln1.bias.grad, "g2": ln2.weight.grad, "b2": ln2.bias.grad,
             "gf": lnf.weight.grad, "bf": lnf.bias.grad}
    return loss.item(), {key: val_.numpy() for key, val_ in pairs.items()}


def adam_step(W, g, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    """Adam's first step (t = 1). m and v start at 0; bias correction divides by (1 - b^1)."""
    new = {}
    for key in W:
        m = (1 - b1) * g[key]
        v2 = (1 - b2) * g[key] ** 2
        m_hat, v_hat = m / (1 - b1), v2 / (1 - b2)
        new[key] = W[key] - lr * m_hat / (np.sqrt(v_hat) + eps)
    return new


def main():
    v = forward(W)
    g, d = backward(W, v)
    t_loss, t_grads = torch_check(W)
    worst = max(np.abs(g[key] - t_grads[key]).max() for key in t_grads)

    LR = 1.0
    W_sgd = {key: W[key] - LR * g[key] for key in W}
    v_sgd = forward(W_sgd)
    W_adam = adam_step(W, g)
    v_adam = forward(W_adam)

    def show(title, **arrays):
        print(f"\n== {title}")
        for name, a in arrays.items():
            print(f"{name} =\n{a}" if np.ndim(a) else f"{name} = {a:.4f}")

    show("weights (rounded to 1 decimal)", **{k_: W[k_] for k_ in ["E", "P", "Wq", "Wk", "Wv", "Wo", "W1", "W2", "Wout"]})
    print(f"\ninput  = {[VOCAB[i] for i in inputs]}   ids {inputs.tolist()}")
    print(f"target = {[VOCAB[i] for i in targets]}   ids {targets.tolist()}")
    show("embedding-lookup", E_rows=W["E"][inputs], x0=v["x0"])
    show("norm (LayerNorm 1)", mean=v["x0"].mean(-1), std=v["ln1"][1].ravel(), a=v["a"])
    show("attention-scores", Q=v["Q"], K=v["K"], S_raw=v["S_raw"], S_scaled=v["S"], S_masked=v["S_masked"])
    show("softmax / attention weights", A=v["A"])
    show("attention-output", V=v["V"], C=v["C"], o=v["o"], x1=v["x1"])
    show("ffn", bn=v["bn"], u=v["u"], gelu_u=v["gu"], f=v["f"], x2=v["x2"])
    show("logits-and-loss", h=v["h"], z=v["z"], p=v["p"], nll=v["nll"], loss=v["loss"], ln5=np.log(V))
    show("grad: softmax + cross-entropy", dz=d["z"])
    show("grad: output head", dWout=g["Wout"], dh=d["h"])
    show("grad: ffn", dx2=d["x2"], du=d["u"], dW1=g["W1"], dW2=g["W2"])
    show("grad: residual + norm", dx1=d["x1"], dx0=d["x0"])
    show("grad: attention", dC=d["C"], dA=d["A"], dS=d["S"], dQ=d["Q"], dK=d["K"], dV=d["V"],
         dWq=g["Wq"], dWk=g["Wk"], dWv=g["Wv"])
    show("grad: embeddings", dE=g["E"], dP=g["P"])
    print(f"\n== check against PyTorch autograd")
    print(f"loss numpy {v['loss']:.6f}   torch {t_loss:.6f}")
    print(f"largest gradient difference over all {sum(a.size for a in W.values())} weights: {worst:.1e}")
    print(f"\n== one update")
    print(f"SGD  (lr {LR}):   loss {v['loss']:.4f} -> {v_sgd['loss']:.4f}")
    print(f"Adam (lr 0.1): loss {v['loss']:.4f} -> {v_adam['loss']:.4f}")

    if "--json" in sys.argv:
        out = {"W": W, "values": {k_: v[k_] for k_ in v if k_ not in ("ln1", "ln2", "lnf")},
               "ln_sigma": {k_: v[k_][1].ravel() for k_ in ("ln1", "ln2", "lnf")},
               "grads": g, "d": d, "torch_loss": t_loss, "max_grad_diff": worst,
               "sgd_lr": LR, "loss_after_sgd": v_sgd["loss"], "W_after_sgd": W_sgd,
               "loss_after_adam": v_adam["loss"], "W_after_adam": W_adam}
        conv = lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, np.floating) else o)
        json.dump(json.loads(json.dumps(out, default=conv).replace("-Infinity", '"-inf"')),
                  open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
