# Card: overfit-one-batch · Lecture 3 ch.10 https://www.youtube.com/watch?v=TCH_1BHY58I&t=2276s
# Run from the course folder: python labs/overfit-one-batch.py
import torch
import torch.nn.functional as F
from common import words, itos, build_dataset, init_params

X, Y = build_dataset(words[:5])  # 32 examples
g, parameters = init_params(n_embd=2, n_hidden=100)  # 3,481 params
C, W1, b1, W2, b2 = parameters

for i in range(1000):
    emb = C[X]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -0.1 * p.grad
    if i in (0, 1, 10, 100, 500) or i == 999:
        print(f'step {i:4d}  loss {loss.item():.4f}')

with torch.no_grad():
    logits = torch.tanh(C[X].view(-1, 6) @ W1 + b1) @ W2 + b2
    print('final loss', round(F.cross_entropy(logits, Y).item(), 4))
    pred = logits.max(1).indices
print('logits.max(1).indices:', pred.tolist())
print('Y                    :', Y.tolist())
print(f'correct: {(pred == Y).sum().item()}/32')
wrong = (pred != Y).nonzero().flatten().tolist()
for i in wrong:
    print(f'  row {i:2d} context {"".join(itos[j] for j in X[i].tolist())!r}: predicted {itos[pred[i].item()]!r}, label {itos[Y[i].item()]!r}')
