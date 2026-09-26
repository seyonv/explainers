# Card: tril-matmul-trick.html · Lecture 7 · matrix multiply as weighted aggregation / version 2 · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2831s
# Run from the course folder: python labs/tril-matmul-trick.py
import torch

# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
b = torch.randint(0,10,(3,2)).float()
for name, a in [("ones", torch.ones(3, 3)),
                ("tril", torch.tril(torch.ones(3, 3)))]:
    print(f"--- a = {name} ---")
    print('a=');  print(a)
    print('b=');  print(b)
    print('c=');  print(a @ b)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
torch.set_printoptions(precision=4)
print("--- a = tril, rows normalised to sum to 1 ---")
print('a=');  print(a)
print('c=');  print(a @ b)

# version 2: using matrix multiply for a weighted aggregation
torch.manual_seed(1337)
B,T,C = 4,8,2
x = torch.randn(B,T,C)
xbow = torch.zeros((B,T,C))
for bi in range(B):
    for t in range(T):
        xbow[bi,t] = torch.mean(x[bi,:t+1], 0)

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
print("wei (T=8), rounded:")
print(wei)
print("shapes:", tuple(wei.shape), "@", tuple(x.shape), "->", tuple(xbow2.shape))
print("torch.allclose(xbow, xbow2):", torch.allclose(xbow, xbow2))
print("max |xbow - xbow2|:", (xbow - xbow2).abs().max().item())
