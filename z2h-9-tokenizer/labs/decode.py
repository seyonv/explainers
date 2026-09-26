# Card: decode.html · Lecture 8 · decoding tokens to strings · https://www.youtube.com/watch?v=zduSFxRajkE&t=2567s
# Run from the course folder: python labs/decode.py
from common import blog_text, train

tokens = list(blog_text().encode('utf-8'))
merges, _ = train(tokens, 276)

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():  # dicts keep insertion order (Python 3.7+)
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    tokens = b''.join(vocab[idx] for idx in ids)
    return tokens.decode('utf-8', errors='replace')

print('vocab[256], vocab[275]:', vocab[256], vocab[275])
print('decode([104, 105, 256, 275]):', repr(decode([104, 105, 256, 275])))
print('128 in binary:', format(128, '08b'))
try:
    b''.join(vocab[i] for i in [128]).decode('utf-8')
except UnicodeDecodeError as e:
    print('strict decode([128]) ->', type(e).__name__ + ':', e)
print("decode([128]) with errors='replace':", decode([128]))
print("'👋' as bytes:", list('👋'.encode('utf-8')), '-> only the first 2:', repr(decode([240, 159])))
