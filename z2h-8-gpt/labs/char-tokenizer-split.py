# Card: char-tokenizer-split.html · Lecture 7 · tokenization, train/val split · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=568s
# Run from the course folder: python labs/char-tokenizer-split.py
import torch
from common import load_text

text = load_text()
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print("encode('hii there') =", encode("hii there"))
print("decode(...)         =", repr(decode(encode("hii there"))))
print("stoi['\\n'] =", stoi['\n'], "  stoi[' '] =", stoi[' '], "  stoi['a'] =", stoi['a'], "  stoi['A'] =", stoi['A'])

data = torch.tensor(encode(text), dtype=torch.long)
print("data.shape, data.dtype:", data.shape, data.dtype)
print("data[:20] =", data[:20].tolist())

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
print(f"n = int(0.9*{len(data)}) = {n}")
print("len(train_data) =", len(train_data), "  len(val_data) =", len(val_data))
print("val starts with:", repr(decode(val_data[:40].tolist())))

# the same text with GPT-2's tokenizer, for the vocab-size vs sequence-length trade-off
try:
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    print("gpt2 tiktoken: n_vocab =", enc.n_vocab, "  encode('hii there') =", enc.encode("hii there"))
    print("gpt2 tokens for the whole text:", len(enc.encode(text)))
except Exception as e:
    print("(tiktoken not available:", type(e).__name__, ")")
