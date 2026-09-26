# Card: bpe-by-hand.html · Lecture 8 · Byte Pair Encoding algorithm walkthrough · https://www.youtube.com/watch?v=zduSFxRajkE&t=1430s
# Run from the course folder: python labs/bpe-by-hand.py   (do the pencil version first)
from common import get_stats, merge, train, decode, build_vocab

# the Wikipedia example with letters: aa -> Z, ab -> Y, ZY -> X
seq = list('aaabdaaabac')
print('start:', ''.join(seq), f'({len(seq)} symbols, vocab 4)')
for pair, new, vocab in [(('a', 'a'), 'Z', 5), (('a', 'b'), 'Y', 6), (('Z', 'Y'), 'X', 7)]:
    stats = get_stats(seq)
    top = max(stats.values())
    seq = merge(seq, pair, new)
    print(f'{pair[0]+pair[1]} -> {new} (count {stats[pair]}, max count {top}):', ''.join(seq), f'({len(seq)} symbols, vocab {vocab})')

# the same text as bytes, 3 merges (minbpe README)
ids = list('aaabdaaabac'.encode('utf-8'))
print('bytes:', ids)
merges, ids = train(ids, 256 + 3)
print('merges:', merges)
print('encoded:', ids)
vocab = build_vocab(merges)
print('tokens:', [vocab[i] for i in ids], '->', decode(ids, vocab))
