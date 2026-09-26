# Card: why-tokenization.html · Lecture 8 · intro: tokenization, GPT-2 paper · https://www.youtube.com/watch?v=zduSFxRajkE&t=0s
# Run from the course folder: python labs/why-tokenization.py
import tiktoken
from common import read

# the GPT lecture: one token per character
text = read('input.txt')
chars = sorted(list(set(text)))
print('shakespeare characters:', f'{len(text):,}')
print('char vocab_size:', len(chars))

# GPT-2: byte-level BPE over chunks of characters
enc = tiktoken.get_encoding('gpt2')
print('gpt2 n_vocab:', f'{enc.n_vocab:,}', '= 256 bytes + 50,000 merges + 1 special')
ids = enc.encode(text)
print('shakespeare in gpt2 tokens:', f'{len(ids):,}')
print(f'characters per gpt2 token: {len(text) / len(ids):.2f}')
cl = tiktoken.get_encoding('cl100k_base')
print('shakespeare in cl100k tokens:', f'{len(cl.encode(text)):,}')
