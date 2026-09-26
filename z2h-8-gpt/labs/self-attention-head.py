# Card: self-attention-head.html · Lecture 7 · THE CRUX OF THE VIDEO: version 4: self-attention · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s
# Run from the course folder: python labs/self-attention-head.py
import torch
import torch.nn as nn
from torch.nn import functional as F

# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
print("raw affinities wei[0] row 7 (q·k, no scaling):", [round(v, 4) for v in wei[0, 7].tolist()])
print(f"raw wei range over the whole batch: {wei.min().item():.2f} .. {wei.max().item():.2f}")

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x
print("out.shape:", out.shape)

torch.set_printoptions(precision=4, linewidth=120)
print("wei[0] =")
print(wei[0])
print("wei[1] row 7 (a different sequence, different weights):", [round(v, 4) for v in wei[1, 7].tolist()])
# which earlier token does the last token of sequence 0 attend to most?
j = wei[0, 7].argmax().item()
print(f"token 7 of sequence 0 puts {wei[0, 7, j].item():.4f} of its attention on token {j}")
