# Card: train-loop.html · Lecture 8 · training the tokenizer: the while loop, compression ratio · https://www.youtube.com/watch?v=zduSFxRajkE&t=2098s
# Run from the course folder: python labs/train-loop.py
from common import blog_text, train, build_vocab

text = blog_text()
tokens = list(map(int, text.encode('utf-8')))

vocab_size = 276  # the desired final vocabulary size
merges, ids = train(tokens, vocab_size, verbose=True)
vocab = build_vocab(merges)

print('tokens length:', len(tokens))
print('ids length:', len(ids))
print(f'compression ratio: {len(tokens) / len(ids):.2f}X')
print('the 20 new tokens:', [vocab[i] for i in range(256, vocab_size)])
print('merges of merges (a forest):', {p: i for p, i in merges.items() if max(p) >= 256})
for vs in [276, 356, 512]:
    _, ids = train(tokens, vs)
    print(f'vocab_size {vs}: {len(ids)} ids, ratio {len(tokens) / len(ids):.2f}X')
