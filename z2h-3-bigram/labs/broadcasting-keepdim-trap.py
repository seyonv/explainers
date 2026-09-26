# Card: broadcasting-keepdim-trap.html · Lecture 2 · vectorized normalization, broadcasting · https://www.youtube.com/watch?v=PaCmpygFfXo&t=2177s
# Run from the course folder: python labs/broadcasting-keepdim-trap.py
import torch
from common import setup

words, stoi, itos, N = setup()
P = N.float()

print('P.shape:', tuple(P.shape))
print('P.sum(0, keepdim=True).shape:', tuple(P.sum(0, keepdim=True).shape))
print('P.sum(1, keepdim=True).shape:', tuple(P.sum(1, keepdim=True).shape))
print('P.sum(1).shape:', tuple(P.sum(1).shape))
print('P.sum().shape:', tuple(P.sum().shape))

good = P / P.sum(1, keepdim=True)   # (27,27) / (27,1): each row divided by its own sum
bad = P / P.sum(1)                  # (27,27) / (27,) -> treated as (1,27): column j divided by row j's sum
print('correct: P[0].sum() = %.4f' % good[0].sum())
print('correct: all rows sum to 1:', torch.allclose(good.sum(1), torch.ones(27)))
print('bug:     P[0].sum() = %.4f' % bad[0].sum())
print('bug:     P[:, 0].sum() = %.4f' % bad[:, 0].sum())
print('bug:     rows that sum to 1:', torch.isclose(bad.sum(1), torch.ones(27)).sum().item(), 'of 27')

P /= P.sum(1, keepdim=True)         # in place, no new tensor
print('in-place /= matches:', torch.equal(P, good))
