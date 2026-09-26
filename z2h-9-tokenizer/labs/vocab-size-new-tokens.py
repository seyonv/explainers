# Card: vocab-size-new-tokens.html · Lecture 8 · how to set vocabulary size? revisiting gpt.py (1:43:27), training new tokens (1:48:11) · https://www.youtube.com/watch?v=zduSFxRajkE&t=6207s
# Run from the course folder: python labs/vocab-size-new-tokens.py
import torch
import torch.nn as nn

# gpt.py's hyperparameters (ng-video-lecture)
block_size, n_embd, n_head, n_layer = 256, 384, 6, 6

def block():  # the layers of one gpt.py Block (heads' k/q/v stacked: same parameter count)
    return nn.ModuleList([nn.Linear(n_embd, 3 * n_embd, bias=False), nn.Linear(n_embd, n_embd),
                          nn.Linear(n_embd, 4 * n_embd), nn.Linear(4 * n_embd, n_embd),
                          nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)])

def count(m):
    return sum(p.numel() for p in m.parameters())

body = count(nn.Embedding(block_size, n_embd)) + n_layer * count(block()) + count(nn.LayerNorm(n_embd))
print(f'parameters that do not depend on vocab_size: {body:,}')
for vocab_size, name in [(65, 'characters (gpt.py)'), (50257, 'GPT-2'), (100277, 'cl100k')]:
    token_embedding_table = nn.Embedding(vocab_size, n_embd)
    lm_head = nn.Linear(n_embd, vocab_size)
    v = count(token_embedding_table) + count(lm_head)
    print(f'vocab_size {vocab_size:>6} ({name}): embedding + lm_head = {v:>10,}; total = {(body + v) / 1e6:.2f}M;'
          f' vocab share {v / (body + v):.1%}; lm_head multiply-adds per token = {n_embd * vocab_size:,}')

# model surgery: add one new token (e.g. a chat special token)
torch.manual_seed(1337)
vocab_size = 50257
token_embedding_table = nn.Embedding(vocab_size, n_embd)
lm_head = nn.Linear(n_embd, vocab_size)
new_emb = nn.Embedding(vocab_size + 1, n_embd)
new_head = nn.Linear(n_embd, vocab_size + 1)
with torch.no_grad():
    new_emb.weight[:vocab_size] = token_embedding_table.weight
    new_emb.weight[vocab_size] = 0.02 * torch.randn(n_embd)   # a small random new row
    new_head.weight[:vocab_size] = lm_head.weight
    new_head.bias[:vocab_size] = lm_head.bias
print('embedding', tuple(token_embedding_table.weight.shape), '->', tuple(new_emb.weight.shape))
print('lm_head  ', tuple(lm_head.weight.shape), '->', tuple(new_head.weight.shape))
print('old rows unchanged:', torch.equal(new_emb.weight[:vocab_size], token_embedding_table.weight))
x = torch.randn(1, n_embd)
print('old logits unchanged:', torch.allclose(new_head(x)[:, :vocab_size], lm_head(x)))
# train only the new rows: zero every other row's gradient with a hook
mask = torch.zeros(vocab_size + 1, 1); mask[vocab_size] = 1
new_emb.weight.register_hook(lambda g: g * mask)
new_emb(torch.tensor([5, vocab_size])).sum().backward()
print('rows with nonzero grad:', new_emb.weight.grad.abs().sum(1).nonzero().flatten().tolist())
