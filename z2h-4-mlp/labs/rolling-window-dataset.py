# Card: rolling-window-dataset · Lecture 3 ch.3 https://www.youtube.com/watch?v=TCH_1BHY58I&t=543s
# Run from the course folder: python labs/rolling-window-dataset.py
from common import words, stoi, itos, build_dataset

block_size = 3
for w in words[:1]:  # emma
    context = [0] * block_size
    for ch in w + '.':
        ctx = "".join(itos[i] for i in context)
        print(ctx, "--->", ch)
        context = context[1:] + [stoi[ch]]

X, Y = build_dataset(words[:5], block_size)
print('first 5 words:', words[:5])
print('X.shape', tuple(X.shape), X.dtype, '| Y.shape', tuple(Y.shape), Y.dtype)
print('X.shape == (32, 3):', tuple(X.shape) == (32, 3))
print('rows whose context is "..." ->', [itos[y] for y in Y[(X == 0).all(1)].tolist()])
Xall, Yall = build_dataset(words, block_size)
print('all words: X.shape', tuple(Xall.shape))
