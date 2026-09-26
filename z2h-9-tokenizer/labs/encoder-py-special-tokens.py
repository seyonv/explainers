# Card: encoder-py-special-tokens.html · Lecture 8 · GPT-2 encoder.py (1:14:59) and special tokens (1:18:26) · https://www.youtube.com/watch?v=zduSFxRajkE&t=4499s
# Run from the course folder: python labs/encoder-py-special-tokens.py   (downloads encoder.json + vocab.bpe, ~1.5 MB)
import json
import tiktoken
from common import data_path

with open(data_path('encoder.json'), 'r') as f:
    encoder = json.load(f)  # <--- ~equivalent to our "vocab" (inverted: str -> id)
with open(data_path('vocab.bpe'), 'r', encoding='utf-8') as f:
    bpe_data = f.read()
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
# ^---- ~equivalent to our "merges"

print('len(encoder):', len(encoder), '# 256 raw byte tokens. 50,000 merges. +1 special token')
print('len(bpe_merges):', len(bpe_merges))
print("encoder['<|endoftext|>']:", encoder['<|endoftext|>'])
print('first 5 merges:', bpe_merges[:5], '(Ġ = space in the byte_encoder alphabet)')
print("encoder['Ġt'], encoder['Ġthe']:", encoder['Ġt'], encoder['Ġthe'])

gpt2 = tiktoken.get_encoding('gpt2')
print('gpt2 special tokens:', gpt2._special_tokens)
cl = tiktoken.get_encoding('cl100k_base')
print('cl100k n_vocab:', cl.n_vocab)
print('cl100k special tokens:', dict(sorted(cl._special_tokens.items(), key=lambda kv: kv[1])))
s = '<|endoftext|>hello world'
print('typed as text  :', gpt2.encode(s, disallowed_special=()))
print('allowed_special:', gpt2.encode(s, allowed_special='all'))
try:
    gpt2.encode(s)
except ValueError as e:
    print('default encode raises ValueError:', str(e).splitlines()[0])
