# Card: scaled-attention.html · Lecture 7 · note 6: "scaled" self-attention, why divide by sqrt(head_size) · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4616s
# Run from the course folder: python labs/scaled-attention.py
import torch
import torch.nn as nn

# replay the crux cell first so the random numbers line up with the notebook
torch.manual_seed(1337)
B,T,C = 4,8,32
head_size = 16
x = torch.randn(B,T,C)
key, query, value = [nn.Linear(C, head_size, bias=False) for _ in range(3)]

k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei_raw = q @ k.transpose(-2, -1)
wei = q @ k.transpose(-2, -1) * head_size**-0.5
print(f"k.var() = {k.var().item():.4f}")
print(f"q.var() = {q.var().item():.4f}")
print(f"(q @ k^T).var() unscaled         = {wei_raw.var().item():.4f}   (head_size = {head_size})")
print(f"(q @ k^T * head_size**-0.5).var() = {wei.var().item():.4f}")

# the same effect with a bigger sample, for head_size 16 and 64
torch.manual_seed(0)
for hs in [16, 64]:
    k = torch.randn(1000, hs); q = torch.randn(1000, hs)
    d = (q * k).sum(-1)
    print(f"head_size {hs:2d}: var(q.k) = {d.var().item():6.2f}   var(q.k / sqrt({hs})) = {(d * hs**-0.5).var().item():.3f}")

# why it matters: softmax of big numbers heads toward one-hot
torch.set_printoptions(precision=4)
s = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
print("softmax(s)      =", torch.softmax(s, dim=-1))
print("softmax(s * 8)  =", torch.softmax(s*8, dim=-1)) # gets too peaky, converges to one-hot
print("softmax(s * 50) =", torch.softmax(s*50, dim=-1))
