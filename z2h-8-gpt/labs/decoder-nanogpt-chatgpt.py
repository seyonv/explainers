# Card: decoder-nanogpt-chatgpt.html · Lecture 7 · nanoGPT walkthrough, batched multi-headed self-attention · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6382s
# Run from the course folder: python labs/decoder-nanogpt-chatgpt.py   (EX1: all heads in one class, checked against Head + MultiHeadAttention)
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from common import MultiHeadAttention, load_text

n_embd, n_head, block_size = 384, 6, 256

class CausalSelfAttention(nn.Module):
    """ nanoGPT style: one Linear makes q, k, v for every head; the heads are a batch dimension """

    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        return self.c_proj(y)

torch.manual_seed(1337)
mha = MultiHeadAttention(n_embd, n_head, n_embd // n_head, block_size)
csa = CausalSelfAttention()
# copy the weights: stack every head's query rows, then every key, then every value
with torch.no_grad():
    Wq = torch.cat([h.query.weight for h in mha.heads]); Wk = torch.cat([h.key.weight for h in mha.heads])
    Wv = torch.cat([h.value.weight for h in mha.heads])
    csa.c_attn.weight.copy_(torch.cat([Wq, Wk, Wv]))
    csa.c_proj.load_state_dict(mha.proj.state_dict())

x = torch.randn(4, 32, n_embd)
a, b = mha(x), csa(x)
count = lambda m: sum(p.numel() for p in m.parameters())
print("Head x 6 + MultiHeadAttention params:", count(mha), "   CausalSelfAttention params:", count(csa))
print("outputs", tuple(a.shape), "allclose:", torch.allclose(a, b, atol=1e-5), f"  max diff {(a - b).abs().max().item():.2e}")
y = F.scaled_dot_product_attention(*[t.view(4, 32, n_head, -1).transpose(1, 2) for t in csa.c_attn(x).split(n_embd, dim=2)], is_causal=True)
y = csa.c_proj(y.transpose(1, 2).contiguous().view(4, 32, n_embd))
print("flash path F.scaled_dot_product_attention(is_causal=True) allclose:", torch.allclose(a, y, atol=1e-5))

# ours vs GPT-3 (the numbers the video compares)
ours_params, ours_chars = 10_788_929, 1_115_394
try:
    import tiktoken
    ours_tokens = len(tiktoken.get_encoding('gpt2').encode(load_text()))
    print(f"tiny shakespeare in GPT-2 tokens: {ours_tokens:,} (the video estimates ~300k)")
except ImportError:
    ours_tokens = 300_000
print(f"GPT-3 175B / our {ours_params:,} params = {175e9 / ours_params:,.0f}x")
print(f"GPT-3 300B tokens / our {ours_tokens:,} tokens = {300e9 / ours_tokens:,.0f}x")
print(f"GELU vs ReLU at x = -1, 0.5: gelu {F.gelu(torch.tensor([-1.0, 0.5])).tolist()}  relu {F.relu(torch.tensor([-1.0, 0.5])).tolist()}")
