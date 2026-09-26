# Shared setup for the z2h-5-batchnorm labs: names.txt, the 80/10/10 split, the lecture's layer classes.
# Lecture 4 (makemore Part 3): https://www.youtube.com/watch?v=P6sfmUTpUmc . Imported by the other labs; not run on its own.
import os, random, sys, time, urllib.request
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES_URL = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
torch.set_num_threads(1)  # tiny matmuls: one thread is fastest and deterministic

def load_words():
    path = os.path.join(HERE, 'data', 'names.txt')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print('downloading names.txt ...')
        urllib.request.urlretrieve(NAMES_URL, path)
    return open(path, 'r').read().splitlines()

words = load_words()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
block_size = 3  # context length: how many characters do we take to predict the next one?

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%
splits = {'train': (Xtr, Ytr), 'val': (Xdev, Ydev), 'test': (Xte, Yte)}

def flag(name):
    return name in sys.argv[1:]

# ---- the lecture's torch.nn-style layers (chapter 10, "PyTorch-ifying the code") ----
# They draw from a module-level generator `g`; each lab resets it with reset_g() before building.
g = torch.Generator().manual_seed(2147483647)

def reset_g(seed=2147483647):
    g.manual_seed(seed)  # reseed in place so `from common import g` stays the same generator
    return g

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)  # batch mean
            xvar = x.var(0, keepdim=True)    # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

def deep_mlp(bn=True, gain=5/3, tanh=True, n_embd=10, n_hidden=100, last_scale=0.1, fan_in_scaling=True):
    """The chapter-10 six-layer MLP. Returns C, layers, parameters (requires_grad set)."""
    reset_g()
    C = torch.randn((vocab_size, n_embd), generator=g)
    dims = [n_embd*block_size] + [n_hidden]*5 + [vocab_size]
    layers = []
    for k in range(6):
        layers.append(Linear(dims[k], dims[k+1], bias=not bn))
        if bn:
            layers.append(BatchNorm1d(dims[k+1]))
        if tanh and k < 5:
            layers.append(Tanh())
    with torch.no_grad():
        if bn:
            layers[-1].gamma *= last_scale    # last layer: make less confident
        else:
            layers[-1].weight *= last_scale
        for layer in layers[:-1]:          # all other layers: apply gain
            if isinstance(layer, Linear):
                if not fan_in_scaling:
                    layer.weight *= layer.weight.shape[0]**0.5   # undo the / fan_in**0.5
                layer.weight *= gain
    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    for p in parameters:
        p.requires_grad = True
    return C, layers, parameters

def deep_forward(C, layers, Xb):
    x = C[Xb].view(Xb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    return x

def train_deep(C, layers, parameters, steps, lr=0.1, track=True):
    """The chapter-10 loop (lr 0.1, batch 32). Returns lossi, ud. Leaves .out.grad on each layer from the last step."""
    lossi, ud = [], []
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        x = deep_forward(C, layers, Xb)
        loss = F.cross_entropy(x, Yb)
        for layer in layers:
            layer.out.retain_grad()
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -lr * p.grad
        lossi.append(loss.item())
        if track:
            with torch.no_grad():
                ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])
    return lossi, ud

# ---- the one-hidden-layer MLP of chapters 2-7, one init per stage of the lecture's loss log ----
STAGES = ['original', 'fix-logits', 'fix-tanh', 'kaiming', 'batchnorm']

def mlp1(stage, n_embd=10, n_hidden=200):
    """Parameters for the chapter-2 MLP after each fix. Draws from g in the notebook's order."""
    reset_g()
    k = STAGES.index(stage)
    P = {}
    P['C'] = torch.randn((vocab_size, n_embd), generator=g)
    P['W1'] = torch.randn((n_embd * block_size, n_hidden), generator=g)
    if stage == 'kaiming' or stage == 'batchnorm':
        P['W1'] *= (5/3)/((n_embd * block_size)**0.5)
    elif k >= 2:
        P['W1'] *= 0.2
    if stage != 'batchnorm':  # a bias before BatchNorm is useless: BN subtracts it right back out
        P['b1'] = torch.randn(n_hidden, generator=g) * (0.01 if k >= 2 else 1)
    P['W2'] = torch.randn((n_hidden, vocab_size), generator=g) * (0.01 if k >= 1 else 1)
    P['b2'] = torch.randn(vocab_size, generator=g) * (0 if k >= 1 else 1)
    if stage == 'batchnorm':
        P['bngain'] = torch.ones((1, n_hidden))
        P['bnbias'] = torch.zeros((1, n_hidden))
        P['bnmean_running'] = torch.zeros((1, n_hidden))  # buffers: no requires_grad
        P['bnstd_running'] = torch.ones((1, n_hidden))
    for name in ['C', 'W1', 'b1', 'W2', 'b2', 'bngain', 'bnbias']:
        if name in P:
            P[name].requires_grad = True
    return P

def params_of(P):
    return [P[n] for n in ['C', 'W1', 'b1', 'W2', 'b2', 'bngain', 'bnbias'] if n in P]

def mlp1_forward(P, X, bn_stats=None):
    """bn_stats=None: batch statistics (training); else a (mean, std) pair to use instead."""
    emb = P['C'][X]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ P['W1']
    if 'b1' in P:
        hpreact = hpreact + P['b1']
    if 'bngain' in P:
        if bn_stats is None:
            bnmeani = hpreact.mean(0, keepdim=True)
            bnstdi = hpreact.std(0, keepdim=True)
            with torch.no_grad():
                P['bnmean_running'] = 0.999 * P['bnmean_running'] + 0.001 * bnmeani
                P['bnstd_running'] = 0.999 * P['bnstd_running'] + 0.001 * bnstdi
        else:
            bnmeani, bnstdi = bn_stats
        hpreact = P['bngain'] * (hpreact - bnmeani) / bnstdi + P['bnbias']
    h = torch.tanh(hpreact)
    logits = h @ P['W2'] + P['b2']
    return hpreact, h, logits

def train_mlp1(P, max_steps=200000, batch_size=32, every=10000, verbose=True):
    """The lecture's loop: lr 0.1, then 0.01 after half the steps. Returns lossi (log10 loss per step)."""
    parameters = params_of(P)
    lossi = []
    for i in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        _, _, logits = mlp1_forward(P, Xb)
        loss = F.cross_entropy(logits, Yb)
        for p in parameters:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < max_steps // 2 else 0.01  # step learning rate decay
        for p in parameters:
            p.data += -lr * p.grad
        if verbose and i % every == 0:
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
    return lossi

@torch.no_grad()
def calibrate_bn(P):
    """Chapter 6: pass the whole training set through once and measure the BN mean/std."""
    emb = P['C'][Xtr]
    hpreact = emb.view(emb.shape[0], -1) @ P['W1']
    return hpreact.mean(0, keepdim=True), hpreact.std(0, keepdim=True)

@torch.no_grad()
def split_loss(P, split, bn_stats=None):
    x, y = splits[split]
    if 'bngain' in P and bn_stats is None:
        bn_stats = (P['bnmean_running'], P['bnstd_running'])
    _, _, logits = mlp1_forward(P, x, bn_stats)
    return F.cross_entropy(logits, y).item()

def steps_from_argv(full=200000):
    return full // 10 if flag('--quick') else full

class Timer:
    def __enter__(self):
        self.t = time.time(); return self
    def __exit__(self, *a):
        self.s = time.time() - self.t
