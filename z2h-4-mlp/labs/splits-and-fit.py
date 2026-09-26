# Card: splits-and-fit · Lecture 3 ch.13-14 https://www.youtube.com/watch?v=TCH_1BHY58I&t=3200s
# Run from the course folder: python labs/splits-and-fit.py
from common import words, splits, init_params, split_loss, train

(Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = splits()  # random.seed(42); 80/10/10 of the words
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
print(f'words: train {n1} / dev {n2 - n1} / test {len(words) - n2}')
print(f'examples: Xtr {tuple(Xtr.shape)} / Xdev {tuple(Xdev.shape)} / Xte {tuple(Xte.shape)}')

for n_hidden in (100, 300):
    g, parameters = init_params(n_embd=2, n_hidden=n_hidden)
    print(f'\nhidden {n_hidden}: {sum(p.nelement() for p in parameters)} params')
    for steps, lr in [(30000, 0.1), (20000, 0.01)]:
        train(parameters, Xtr, Ytr, steps, lambda i: lr, g)
        print(f'  +{steps} steps at lr {lr}: train {split_loss(parameters, Xtr, Ytr):.4f} | dev {split_loss(parameters, Xdev, Ydev):.4f}')
