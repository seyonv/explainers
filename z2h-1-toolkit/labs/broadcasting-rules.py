# Card: broadcasting-rules (the two rules, and the keepdim bug) · L2 https://www.youtube.com/watch?v=PaCmpygFfXo&t=2177s
# Run from z2h-1-toolkit/: python labs/broadcasting-rules.py
import torch

pairs = [((27, 27), (27, 1)), ((27, 27), (27,)), ((32, 64), (64,)),
         ((4, 8, 32), (8, 32)), ((3, 4), (3,)), ((5, 1, 4), (3, 1))]
for s1, s2 in pairs:
    try:
        r = tuple((torch.zeros(s1) + torch.zeros(s2)).shape)
    except RuntimeError:
        r = 'error'
    print(f'{str(s1):12s} + {str(s2):9s} -> {r}')

N = torch.tensor([[1., 1., 2.],
                  [3., 0., 1.],
                  [6., 2., 2.]])
print('\nN =', N.tolist())
good = N / N.sum(1, keepdim=True)     # (3,3)/(3,1): each ROW divided by its row sum
bad = N / N.sum(1)                    # (3,3)/(3,) -> (1,3): each COLUMN j divided by row-sum j
print('N.sum(1, keepdim=True) shape', tuple(N.sum(1, keepdim=True).shape), '=', N.sum(1, keepdim=True).flatten().tolist())
print('N.sum(1)               shape', tuple(N.sum(1).shape), '=', N.sum(1).tolist())
print('keepdim=True :', [[round(v, 4) for v in r] for r in good.tolist()])
print('   row sums  :', [round(v, 4) for v in good.sum(1).tolist()])
print('keepdim=False:', [[round(v, 4) for v in r] for r in bad.tolist()])
print('   row sums  :', [round(v, 4) for v in bad.sum(1).tolist()])
