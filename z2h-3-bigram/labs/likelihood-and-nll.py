# Card: likelihood-and-nll.html · Lecture 2 · loss function (negative log likelihood) · https://www.youtube.com/watch?v=PaCmpygFfXo&t=3014s
# Run from the course folder: python labs/likelihood-and-nll.py
import torch
from common import setup

words, stoi, itos, N = setup()
P = N.float()
P /= P.sum(1, keepdim=True)

def nll_of(ws, verbose=False):
    log_likelihood, n = 0.0, 0
    for w in ws:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            prob = P[stoi[ch1], stoi[ch2]]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1
            if verbose:
                print(f'  {ch1}{ch2}: prob={prob:.4f} logprob={logprob:.4f}')
    return log_likelihood, n

print('emma, bigram by bigram:')
nll_of(['emma'], verbose=True)
print('uniform baseline 1/27 = %.4f' % (1 / 27))

ll, n = nll_of(words[:3])
print('first 3 words: log_likelihood=%.4f n=%d nll=%.4f avg nll=%.4f' % (ll, n, -ll, -ll / n))

ll, n = nll_of(words)
print('all words: log_likelihood=%.2f n=%d avg nll=%.4f' % (ll, n, -ll / n))
# the loop above adds float32 tensors, so rounding creeps in; the exact float64 value:
xs = torch.tensor([stoi[c] for w in words for c in '.' + w])
ys = torch.tensor([stoi[c] for w in words for c in w + '.'])
print('all words, float64: avg nll=%.6f' % -P.double()[xs, ys].log().mean())

for w in ['andrej', 'andrejq']:
    ll, n = nll_of([w])
    print('%s: avg nll=%.4f' % (w, -ll / n))
print("P[j, q] =", P[stoi['j'], stoi['q']].item(), '| count N[j, q] =', N[stoi['j'], stoi['q']].item())
