# Card: embeddings-and-positions.html · Lecture 7 · minor code cleanup / positional encoding · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3506s
# Run from the course folder: python labs/embeddings-and-positions.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from common import Shakespeare, train, nparams

ds = Shakespeare()
vocab_size = ds.vocab_size
batch_size, block_size, n_embd = 32, 8, 32

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, verbose=False):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if verbose:
            print("idx", tuple(idx.shape), "-> tok_emb", tuple(tok_emb.shape), "+ pos_emb", tuple(pos_emb.shape),
                  "-> x", tuple(x.shape), "-> logits", tuple(logits.shape))
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        return logits, F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

torch.manual_seed(1337)
m = BigramLanguageModel()
xb, yb = ds.get_batch('train', batch_size, block_size)
m(xb, yb, verbose=True)
print(f"params: token table {vocab_size}x{n_embd}={vocab_size*n_embd}, position table {block_size}x{n_embd}={block_size*n_embd},"
      f" lm_head {n_embd}x{vocab_size}+{vocab_size}={n_embd*vocab_size+vocab_size}, total {nparams(m)}")

# same character, two positions: the vectors differ only by the position rows
e = m.token_embedding_table.weight[ds.stoi['e']]
x2 = e + m.position_embedding_table.weight[2]
x5 = e + m.position_embedding_table.weight[5]
print("'e' at position 2 vs 5: first 4 dims", [round(v, 4) for v in x2[:4].tolist()], "vs", [round(v, 4) for v in x5[:4].tolist()])

# a position index past block_size has no row: this is why generate() must crop to block_size
try:
    m(torch.zeros((1, block_size + 1), dtype=torch.long))
except IndexError as err:
    print(f"T = {block_size + 1} > block_size: IndexError: {err}")

# trained with bigram.py's settings, the extra tables don't help yet: each token still sees only itself
print("== training: batch 32, block 8, lr 1e-2, 3000 iters (bigram.py settings) ==")
torch.manual_seed(1337)
m = BigramLanguageModel()
train(m, ds, batch_size, block_size, 3000, 1000, 1e-2)
