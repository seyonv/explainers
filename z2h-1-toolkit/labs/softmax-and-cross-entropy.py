# Card: softmax-and-cross-entropy (first look) · L2 https://www.youtube.com/watch?v=PaCmpygFfXo&t=4726s
# Run from z2h-1-toolkit/: python labs/softmax-and-cross-entropy.py
import math, torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
counts = logits.exp()
probs = counts / counts.sum()
print('logits  =', [round(v, 4) for v in logits.tolist()])
print('exp     =', [round(v, 4) for v in counts.tolist()], ' sum =', round(counts.sum().item(), 4))
print('softmax =', [round(v, 4) for v in probs.tolist()])
print('F.softmax matches:', torch.allclose(probs, F.softmax(logits, dim=0)))

for scale in [0.5, 1.0, 8.0]:
    print(f'softmax(logits * {scale:<3}) =', [round(v, 4) for v in F.softmax(logits * scale, 0).tolist()])
print('softmax(logits + 100) =', [round(v, 4) for v in F.softmax(logits + 100, 0).tolist()], '(shift does nothing)')
big = torch.tensor([1000., 999., 998.])
print('naive exp on [1000, 999, 998] ->', (big.exp() / big.exp().sum()).tolist())
m = big - big.max()
print('subtract max first            ->', [round(v, 4) for v in (m.exp() / m.exp().sum()).tolist()])

print()
for target in range(3):
    print(f'target {target}: -log p = {-probs[target].log().item():.4f}   F.cross_entropy = {F.cross_entropy(logits[None], torch.tensor([target])).item():.4f}')
print(f'certain (p=1): -log 1 = {0.0 - math.log(1.0):.4f}')
print(f'uniform over 27: -log(1/27) = {-math.log(1/27):.4f}')
print(f'F.cross_entropy(zeros(27), any target) = {F.cross_entropy(torch.zeros(1, 27), torch.tensor([5])).item():.4f}')

ps = torch.tensor([0.4, 0.1, 0.25])            # model's p(correct) on 3 examples
print(f'\nlikelihood = prod p = {ps.prod().item():.4f}')
print(f'log likelihood = sum log p = {ps.log().sum().item():.4f}')
print(f'average negative log likelihood = {-ps.log().mean().item():.4f}')
