# Shared setup for the z2h-6-backprop-ninja labs: names.txt, the 80/10/10 split, cmp(), and the chunked forward pass.
# Lecture 5 (makemore Part 4): https://www.youtube.com/watch?v=q8SA3rM6ckI . Imported by the other labs; not run on its own.
import os, random, sys, urllib.request
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

def flag(name):
    return name in sys.argv[1:]

# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

def init_params(n_embd=10, n_hidden=64):
    """The notebook's init (cell 8): small random biases and BN params so a wrong backward can't hide.
    Karpathy draws bngain/bnbias from the unseeded global RNG; we seed it (torch.manual_seed(42)) so runs repeat."""
    torch.manual_seed(42)
    g = torch.Generator().manual_seed(2147483647) # for reproducibility
    C  = torch.randn((vocab_size, n_embd),            generator=g)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
    b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun, it's useless because of BN
    W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
    b2 = torch.randn(vocab_size,                      generator=g) * 0.1
    bngain = torch.randn((1, n_hidden))*0.1 + 1.0
    bnbias = torch.randn((1, n_hidden))*0.1
    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True
    return g, parameters

def chunked_forward():
    """Cells 8-10: build the 4,137-param net, take one batch of 32, run the forward pass in atomic
    steps, retain_grad() every intermediate and call loss.backward(). Returns a dict of every tensor."""
    g, parameters = init_params()
    C, W1, b1, W2, b2, bngain, bnbias = parameters
    batch_size = 32
    n = batch_size # a shorter variable also, for convenience
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
    # Linear layer 1
    hprebn = embcat @ W1 + b1 # hidden layer pre-activation
    # BatchNorm layer
    bnmeani = 1/n*hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff**2
    bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # Non-linearity
    h = torch.tanh(hpreact) # hidden layer
    # Linear layer 2
    logits = h @ W2 + b2 # output layer
    # cross entropy loss (same as F.cross_entropy(logits, Yb))
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes # subtract max for numerical stability
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), Yb].mean()

    # PyTorch backward pass
    for p in parameters:
        p.grad = None
    for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
              norm_logits, logit_maxes, logits, h, hpreact, bnraw,
              bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
              embcat, emb]:
        t.retain_grad()
    loss.backward()
    return {k: v for k, v in locals().items() if k not in ('t', 'p')}
