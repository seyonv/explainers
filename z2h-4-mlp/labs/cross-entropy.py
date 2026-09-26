# Card: cross-entropy · Lecture 3 ch.7, 9 https://www.youtube.com/watch?v=TCH_1BHY58I&t=1793s
# Run from the course folder: python labs/cross-entropy.py
import torch
import torch.nn.functional as F
from common import words, build_dataset, init_params, forward

X, Y = build_dataset(words[:5])
g, parameters = init_params(n_embd=2, n_hidden=100)  # the 3,481-param net, seed 2147483647
with torch.no_grad():
    logits = forward(parameters, X)  # (32, 27)

counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
print('prob.shape', tuple(prob.shape), '| each row sums to 1:', torch.allclose(prob.sum(1), torch.ones(32)))
print('prob of the correct char, first 5 rows:', ' '.join(f'{v:.2e}' for v in prob[torch.arange(32), Y][:5].tolist()))
loss = -prob[torch.arange(32), Y].log().mean()
print('manual loss      ', round(loss.item(), 4))
print('F.cross_entropy  ', round(F.cross_entropy(logits, Y).item(), 4))

print('\nwhy F.cross_entropy is safer:')
for row in ([-2, 3, -3, 0, 5], [-100, 3, -3, 0, 5], [-2, 3, -3, 0, 100]):
    lg = torch.tensor(row, dtype=torch.float32)
    counts = lg.exp()
    probs = counts / counts.sum()
    print(f'  logits {str(row):22s} exp -> {counts.max().item():.4g} max; manual probs {[round(p, 4) for p in probs.tolist()]}')
lg = torch.tensor([-2., 3, -3, 0, 100])
print('  manual softmax with 100 gives nan:', torch.isnan(lg.exp() / lg.exp().sum()).any().item())
print('  subtract the max first:', [round(p, 4) for p in ((lg - lg.max()).exp() / (lg - lg.max()).exp().sum()).tolist()])
print('  F.softmax (does that internally):', [round(p, 4) for p in F.softmax(lg, 0).tolist()])
lg = torch.tensor([-2., 3, -3, 0, 5])
print('  adding a constant changes nothing:', torch.allclose(F.softmax(lg, 0), F.softmax(lg - 5, 0)))
