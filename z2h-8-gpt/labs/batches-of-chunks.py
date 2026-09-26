# Card: batches-of-chunks.html · Lecture 7 · data loader: batches of chunks of data · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=867s
# Run from the course folder: python labs/batches-of-chunks.py
import torch
from common import Shakespeare

ds = Shakespeare()
train_data = ds.train_data

block_size = 8
print("train_data[:block_size+1] =", train_data[:block_size+1].tolist())
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context.tolist()} the target: {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = ds.train_data if split == 'train' else ds.val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y, ix

xb, yb, ix = get_batch('train')
print("ix (random offsets) =", ix.tolist())
print('inputs:', xb.shape)
print(xb)
print('targets:', yb.shape)
print(yb)
print('----')
n = 0
for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        n += 1
        if b == 0:
            print(f"when input is {context.tolist()} the target: {target}")
print(f"... {n} (context -> target) examples in one {batch_size}x{block_size} batch")
print("row 0 as text:", repr(ds.decode(xb[0].tolist())), "->", repr(ds.decode(yb[0].tolist())))
