# Card: smoothing.html · Lecture 2 · model smoothing with fake counts · https://www.youtube.com/watch?v=PaCmpygFfXo&t=3650s
# Run from the course folder: python labs/smoothing.py
import torch
from common import setup

words, stoi, itos, N = setup()

def avg_nll(P, ws):
    log_likelihood, n = 0.0, 0
    for w in ws:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]]).item()
            n += 1
    return -log_likelihood / n

for k in [0, 1]:
    P = (N + k).float()
    P /= P.sum(1, keepdim=True)
    print(f'fake counts +{k}: P[j,q]={P[stoi["j"], stoi["q"]]:.6f}  '
          f'andrejq={avg_nll(P, ["andrejq"]):.4f}  all words={avg_nll(P, words):.4f}')

print('more fake counts -> flatter model (loss on all words):')
for k in [10, 100, 1000, 100000]:
    P = (N + k).float()
    P /= P.sum(1, keepdim=True)
    print(f'  +{k}: {avg_nll(P, words):.4f}')
print('uniform, log(27) = %.4f' % torch.log(torch.tensor(27.0)))
