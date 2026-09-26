# Card: convolutions-and-workflow · Lecture 6 ch.14 dilated causal convolutions https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=2864s
# Run from the course folder: python labs/convolutions-and-workflow.py
import torch
from common import splits, itos, wavenet_model, set_training

block_size = 8
data = splits(block_size)
Xtr, Ytr = data['train']
for x, y in zip(Xtr[7:15], Ytr[7:15]):              # one name = 8 training examples
    print(''.join(itos[ix.item()] for ix in x), '-->', itos[y.item()])

torch.manual_seed(42)
model = wavenet_model(n_embd=24, n_hidden=128)      # untrained is fine: we only look at shapes and reuse
set_training(model, False)                          # eval mode, so each row's output depends only on that row

with torch.no_grad():
    logits = model(Xtr[[7]])                         # forward a single example
    print('one example:', tuple(logits.shape))
    logits = torch.zeros(8, 27)                      # forward all of them, one at a time
    for i in range(8):
        logits[i] = model(Xtr[[7+i]])
    print('8 examples in a Python loop:', tuple(logits.shape))
    print('same as one batched call:', torch.allclose(logits, model(Xtr[7:15]), atol=1e-6))

    # Which tree nodes does the loop compute more than once? Label each node by the slice of the name it covers.
    name = '.' * 7 + 'diondre.'                        # 7 pads + the name + the '.' it ends with
    computed, unique = 0, set()
    for i in range(8):                                  # example i sees positions i .. i+7 of `name`
        for level, width in [(1, 2), (2, 4), (3, 8)]:
            for start in range(i, i + 8, width):
                computed += 1
                unique.add((level, start))
    print(f'tree nodes computed by the loop: {computed}; distinct (level, position) nodes: {len(unique)}; '
          f'computed more than once: {computed - len(unique)}')
    for level, width in [(1, 2), (2, 4), (3, 8)]:
        n = sum(1 for (l, s) in unique if l == level)
        print(f'  level {level} (covers {width} chars): {n} distinct nodes, {8 * 8 // width} computed')
    print('e.g. the level-1 node for', repr(name[9:11]), 'is recomputed by examples', [i for i in range(8) if 9 in range(i, i + 8, 2)])
