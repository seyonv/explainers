# Shared setup for the z2h-7-wavenet labs: names.txt, the 80/10/10 split, the lecture's layer classes, train/eval/sample.
# Lecture 6 (makemore Part 5): https://www.youtube.com/watch?v=t3YJ5hKiMQ0 . Imported by the other labs; not run on its own.
import os, random, sys, time, urllib.request
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES_URL = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
torch.set_num_threads(1)  # small matmuls: one thread is fast, and steady when other jobs share the CPU

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
random.seed(42)
random.shuffle(words)   # shuffle up the words

def build_dataset(words, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append
    return torch.tensor(X), torch.tensor(Y)

def splits(block_size):
    """80/10/10 split of the shuffled words: {'train': (X, Y), 'val': ..., 'test': ...}"""
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))
    return {'train': build_dataset(words[:n1], block_size),
            'val':   build_dataset(words[n1:n2], block_size),
            'test':  build_dataset(words[n2:], block_size)}

def flag(name):
    return name in sys.argv[1:]

# ---- the lecture's layers (notebook cell 7). Like the notebook, they draw from the global RNG: torch.manual_seed(42) ----
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # note: kaiming init
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1, fixed=True):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.fixed = fixed  # False reproduces the chapter-9 bug: 3-D input only reduced over dim 0
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    def __call__(self, x):
        if self.training:
            if x.ndim == 2 or not self.fixed:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim, keepdim=True) # batch mean
            xvar = x.var(dim, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
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

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    def parameters(self):
        return [self.weight]

class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    def parameters(self):
        return []

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        # get parameters of all layers and stretch them out into one list
        return [p for layer in self.layers for p in layer.parameters()]

# ---- the two model shapes of the lecture ----
def flat_model(block_size, n_embd=10, n_hidden=200):
    """The starter-code MLP: every character of the context squashed into one hidden layer."""
    model = Sequential([
        Embedding(vocab_size, n_embd), Flatten(),
        Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])
    return finish(model)

def wavenet_model(n_embd=24, n_hidden=128, fixed_bn=True):
    """The hierarchical net (block_size 8): fuse 2 characters, then 2 pairs, then 2 quads."""
    model = Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden, fixed=fixed_bn), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden, fixed=fixed_bn), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden, fixed=fixed_bn), Tanh(),
        Linear(n_hidden, vocab_size),
    ])
    return finish(model)

def finish(model):
    with torch.no_grad():
        model.layers[-1].weight *= 0.1 # last layer make less confident
    for p in model.parameters():
        p.requires_grad = True
    return model

def n_params(model):
    return sum(p.nelement() for p in model.parameters())

def train(model, Xtr, Ytr, max_steps=200000, batch_size=32, log=True):
    """The lecture's loop: SGD, lr 0.1 then 0.01 for the last quarter (step 150,000 of 200,000). Returns lossi (log10)."""
    parameters = model.parameters()
    lossi = []
    decay_at = max_steps * 3 // 4
    t0 = time.time()
    for i in range(max_steps):
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
        # forward pass
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb) # loss function
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        # update: simple SGD
        lr = 0.1 if i < decay_at else 0.01 # step learning rate decay
        for p in parameters:
            p.data += -lr * p.grad
        # track stats
        if log and i % (max_steps // 20) == 0: # print every once in a while
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
    if log:
        print(f'trained {max_steps} steps in {time.time() - t0:.0f} s')
    return lossi

def set_training(model, training):
    for layer in model.layers:
        layer.training = training

@torch.no_grad() # this decorator disables gradient tracking inside pytorch
def split_loss(model, data, split):
    x, y = data[split]
    loss = F.cross_entropy(model(x), y)
    print(split, f'{loss.item():.4f}')
    return loss.item()

@torch.no_grad()
def sample(model, block_size, n=20):
    out_words = []
    for _ in range(n):
        out = []
        context = [0] * block_size # initialize with all ...
        while True:
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        out_words.append(''.join(itos[i] for i in out))
    return out_words

def steps_from_argv(full=200000, quick=20000):
    return quick if flag('--quick') else full
