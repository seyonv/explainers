# Card: average-the-past.html · Lecture 7 · version 1: averaging past context with for loops · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2533s
# Run from the course folder: python labs/average-the-past.py
import time
import torch

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
print("x.shape:", x.shape)

# We want x[b,t] = mean_{i<=t} x[b,i]
t0 = time.perf_counter()
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)
dt = time.perf_counter() - t0

torch.set_printoptions(precision=4)
print("x[0] =")
print(x[0])
print("xbow[0] =")
print(xbow[0])
print(f"check row 1: ({x[0,0,0]:.4f} + {x[0,1,0]:.4f}) / 2 = {(x[0,0,0] + x[0,1,0]) / 2:.4f}")
print(f"double loop over {B*T} (b,t) pairs took {dt*1e3:.2f} ms")
