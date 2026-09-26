# Card: masked-softmax.html · Lecture 7 · version 3: adding softmax · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3282s
# Run from the course folder: python labs/masked-softmax.py
import torch
from torch.nn import functional as F

torch.manual_seed(1337)
B,T,C = 4,8,2
x = torch.randn(B,T,C)
wei2 = torch.tril(torch.ones(T, T)); wei2 = wei2 / wei2.sum(1, keepdim=True)
xbow2 = wei2 @ x

# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
torch.set_printoptions(precision=4, linewidth=120)
print("after masked_fill (first 4 rows):")
print(wei[:4])
wei = F.softmax(wei, dim=-1)
print("after softmax (first 4 rows):")
print(wei[:4])
xbow3 = wei @ x
print("torch.allclose(xbow2, xbow3):", torch.allclose(xbow2, xbow3))

# the payoff: the zeros can be any scores ("affinities"); the mask still keeps the future out
scores = torch.tensor([[0.0, 0, 0, 0],
                       [1.0, 3.0, 0, 0],
                       [2.0, 0.0, 1.0, 0],
                       [0.5, 0.5, 2.0, 5.0]])
m4 = torch.tril(torch.ones(4, 4))
w = F.softmax(scores.masked_fill(m4 == 0, float('-inf')), dim=-1)
print("learned-looking scores, masked and softmaxed:")
print(w)
print("row sums:", [round(s, 4) for s in w.sum(-1).tolist()])
print("exp(0) =", torch.exp(torch.tensor(0.)).item(), "  exp(-inf) =", torch.exp(torch.tensor(float('-inf'))).item())
# what goes wrong without the mask: the last row of scores leaks into row 1
print("row 1 without the mask:", F.softmax(scores[1], dim=-1))
