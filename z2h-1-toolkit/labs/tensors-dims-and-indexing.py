# Card: tensors-dims-and-indexing (shape, dim, keepdim, view, index) · L3 https://www.youtube.com/watch?v=TCH_1BHY58I&t=739s
# Run from z2h-1-toolkit/: python labs/tensors-dims-and-indexing.py
import torch

for t in [torch.tensor(5.), torch.tensor([1., 2., 3.]), torch.zeros(2, 3), torch.zeros(4, 5, 80)]:
    print(f'ndim={t.dim()} shape={tuple(t.shape)}')

M = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
print('\nM =', M.tolist())
print('M.sum(dim=0) =', M.sum(dim=0).tolist(), ' shape', tuple(M.sum(0).shape), '(collapse rows -> one per column)')
print('M.sum(dim=1) =', M.sum(dim=1).tolist(), ' shape', tuple(M.sum(1).shape), '(collapse columns -> one per row)')
print('M.sum(dim=1, keepdim=True) shape', tuple(M.sum(1, keepdim=True).shape), '=', M.sum(1, keepdim=True).tolist())

a = torch.arange(18)
v = a.view(3, 3, 2)
print('\narange(18).view(3,3,2) shape', tuple(v.shape), ' shares storage:', v.data_ptr() == a.data_ptr())
print('a.view(9, 2)[1] =', a.view(9, 2)[1].tolist(), '  a.view(-1, 6).shape =', tuple(a.view(-1, 6).shape))
try:
    M.t().view(6)
except RuntimeError as e:
    print('M.t().view(6) ->', str(e)[:60], '...  (use .reshape)')

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)       # embedding table, as in L3
X = torch.randint(0, 27, (32, 3), generator=g)
emb = C[X]
print('\nC', tuple(C.shape), ' X', tuple(X.shape), ' C[X]', tuple(emb.shape))
print('C[5] =', [round(v, 4) for v in C[5].tolist()])
print('C[[5, 6, 7]].shape =', tuple(C[[5, 6, 7]].shape))
print('C[X][13, 2] == C[X[13, 2]]:', torch.equal(emb[13, 2], C[X[13, 2]]))

print('\ntorch.tensor([1, 2, 3]).dtype =', torch.tensor([1, 2, 3]).dtype)
print('torch.Tensor([1, 2, 3]).dtype =', torch.Tensor([1, 2, 3]).dtype)
print('torch.tensor([1,2,3]) / 2 =', (torch.tensor([1, 2, 3]) / 2).tolist(), '; // 2 =', (torch.tensor([1, 2, 3]) // 2).tolist())
