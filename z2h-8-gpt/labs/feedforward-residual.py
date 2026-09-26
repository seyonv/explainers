# Card: feedforward-residual.html · Lecture 7 · feedforward layers / residual connections · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5065s
# Run from the course folder: python labs/feedforward-residual.py   (trains three small models on CPU)
import torch
import torch.nn as nn
from torch.nn import functional as F
from common import Shakespeare, MultiHeadAttention, train, nparams

batch_size, block_size, max_iters, eval_interval, learning_rate, eval_iters = 32, 8, 5000, 500, 1e-3, 200
n_embd, n_head = 32, 4
ds = Shakespeare()
vocab_size = ds.vocab_size

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, residual):
        super().__init__()
        if residual:   # 4x inner width, then project back into the residual pathway
            self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
        else:          # the first version: one Linear + ReLU, applied to each token on its own
            self.net = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, residual):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, proj=residual)
        self.ffwd = FeedFoward(n_embd, residual)
        self.residual = residual

    def forward(self, x):
        if self.residual:
            x = x + self.sa(x)
            x = x + self.ffwd(x)
        else:
            x = self.sa(x)
            x = self.ffwd(x)
        return x

class Model(nn.Module):

    def __init__(self, blocks):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = blocks
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T))
        logits = self.lm_head(self.blocks(x))
        B, T, C = logits.shape
        return logits, F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

# the "gradient superhighway": through x + f(x), d(out)/dx = 1 + df/dx, so the gradient always has a direct path
x = torch.tensor(2.0, requires_grad=True)
f = lambda x: 0.01 * torch.tanh(x)            # a sublayer that barely passes gradient at init
(f(f(f(x)))).backward(); g_plain = x.grad.item(); x.grad = None
y = x; y = y + f(y); y = y + f(y); y = y + f(y)
y.backward(); g_res = x.grad.item()
print(f"3 stacked sublayers, d(out)/dx: plain f(f(f(x))) = {g_plain:.3e}   residual x+f(x) three times = {g_res:.4f}")

stages = [
    ("heads + one feed-forward (Linear+ReLU)", lambda: nn.Sequential(MultiHeadAttention(n_embd, n_head, n_embd//n_head, block_size, proj=False),
                                                                      FeedFoward(n_embd, residual=False))),
    ("3 Blocks, no residual", lambda: nn.Sequential(*[Block(n_embd, n_head, residual=False) for _ in range(3)])),
    ("3 Blocks + residual, projections, 4x FFN", lambda: nn.Sequential(*[Block(n_embd, n_head, residual=True) for _ in range(3)])),
]
for name, make in stages:
    torch.manual_seed(1337)
    model = Model(make())
    print(f"== {name}: {nparams(model)} params ==")
    train(model, ds, batch_size, block_size, max_iters, eval_interval, learning_rate, eval_iters)
