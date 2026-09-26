# Card: embedding-lookup · Lecture 3 ch.4 https://www.youtube.com/watch?v=TCH_1BHY58I&t=739s
# Run from the course folder: python labs/embedding-lookup.py
import torch
import torch.nn.functional as F
from common import words, build_dataset

X, Y = build_dataset(words[:5])
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)

print('C.shape', tuple(C.shape))
print('C[5]                       ', C[5])
print('F.one_hot(5, 27).float()@C ', F.one_hot(torch.tensor(5), num_classes=27).float() @ C)
print('equal:', torch.allclose(C[5], F.one_hot(torch.tensor(5), num_classes=27).float() @ C))

# the two errors shown on purpose in the video
try:
    F.one_hot(5, num_classes=27)
except TypeError as e:
    print('one_hot(int)  ->', type(e).__name__ + ':', str(e).splitlines()[0])
try:
    F.one_hot(torch.tensor(5), num_classes=27) @ C
except RuntimeError as e:
    print('long @ float  ->', type(e).__name__ + ':', str(e).splitlines()[0])

print('C[[5, 6, 7]].shape', tuple(C[[5, 6, 7]].shape))
emb = C[X]
print('X.shape', tuple(X.shape), '-> emb = C[X] .shape', tuple(emb.shape))
print('X[13, 2] =', X[13, 2].item(), '| emb[13, 2] =', emb[13, 2], '| C[X[13, 2]] =', C[X[13, 2]])
