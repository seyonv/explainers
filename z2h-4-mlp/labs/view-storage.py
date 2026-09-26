# Card: view-storage · Lecture 3 ch.5 https://www.youtube.com/watch?v=TCH_1BHY58I&t=1115s
# Run from the course folder: python labs/view-storage.py
import torch
from common import words, build_dataset

a = torch.arange(18)
print('a.shape', tuple(a.shape), 'stride', a.stride())
for shape in [(2, 9), (9, 2), (3, 3, 2)]:
    v = a.view(*shape)
    print(f'a.view{shape}: stride {v.stride()}, same storage: {v.data_ptr() == a.data_ptr()}')
print('a.view(3, 3, 2) =\n', a.view(3, 3, 2))
print('storage:', a.untyped_storage().nbytes(), 'bytes =', a.untyped_storage().nbytes() // a.element_size(), 'int64s')

X, Y = build_dataset(words[:5])
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
emb = C[X]  # (32, 3, 2)

cat1 = torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], 1)
cat2 = torch.cat(torch.unbind(emb, 1), 1)
view = emb.view(-1, 6)
print('emb.shape', tuple(emb.shape), '-> view(-1, 6).shape', tuple(view.shape))
print('cat == cat(unbind) == view:', torch.equal(cat1, cat2) and torch.equal(cat2, view))
print('view shares emb memory:', view.data_ptr() == emb.data_ptr(), '| cat shares it:', cat2.data_ptr() == emb.data_ptr())
print('row 0 of view(-1, 6):', view[0])
