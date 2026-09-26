# Card: attention-notes.html · Lecture 7 · notes 1-5 on attention · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4298s
# Run from the course folder: python labs/attention-notes.py   (the card is mostly reading; this checks notes 2-4 in code)
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B,T,C = 4,8,32
head_size = 16
x = torch.randn(B,T,C)
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
tril = torch.tril(torch.ones(T, T))

def head(x, mask=True):
    k, q, v = key(x), query(x), value(x)
    wei = q @ k.transpose(-2, -1) * head_size**-0.5
    if mask:   # decoder block: delete this line and you have an encoder block
        wei = wei.masked_fill(tril[:x.shape[1], :x.shape[1]] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    return wei @ v, wei

torch.set_printoptions(precision=3, linewidth=120)
_, wei_dec = head(x, mask=True)
_, wei_enc = head(x, mask=False)
print("note 4, decoder (masked) wei[0] rows 0 and 3:")
print(wei_dec[0, [0, 3]])
print("note 4, encoder (tril line deleted) wei[0] rows 0 and 3:")
print(wei_enc[0, [0, 3]])

# note 1: a directed graph. Node t receives edges from nodes 0..t in the decoder, from all 8 in the encoder.
print("note 1, incoming edges per node, decoder:", (wei_dec[0] > 0).sum(-1).tolist(), " encoder:", (wei_enc[0] > 0).sum(-1).tolist())

# note 2: no notion of space. Shuffle the tokens (encoder, no positions): the outputs are the same rows, shuffled.
perm = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4])
out, _ = head(x, mask=False)
out_p, _ = head(x[:, perm], mask=False)
print("note 2, encoder output of shuffled input == shuffled output:", torch.allclose(out_p, out[:, perm], atol=1e-6))

# note 3: no talk across the batch. Change sequence 1 completely; sequence 0's output does not move.
out_dec, _ = head(x, mask=True)
x2 = x.clone(); x2[1] = torch.randn(T, C)
out_dec2, _ = head(x2, mask=True)
print("note 3, seq 0 unchanged after editing seq 1:", torch.equal(out_dec[0], out_dec2[0]),
      "  seq 1 changed:", not torch.allclose(out_dec[1], out_dec2[1]))
# the decoder mask means the future can't reach the past: edit token 5 and tokens 0..4 stay put
x3 = x.clone(); x3[0, 5] = torch.randn(C)
out_dec3, _ = head(x3, mask=True)
same = [torch.allclose(out_dec[0, t], out_dec3[0, t]) for t in range(T)]
print("editing token 5 of seq 0: output unchanged at positions", [t for t in range(T) if same[t]])

# note 5: cross-attention. Queries from x, keys and values from another source (say, 6 encoder outputs).
enc = torch.randn(B, 6, C)
q = query(x); k = key(enc); v = value(enc)
wei = F.softmax(q @ k.transpose(-2, -1) * head_size**-0.5, dim=-1)
print("note 5, cross-attention: wei", tuple(wei.shape), "(8 queries x 6 keys), out", tuple((wei @ v).shape))
