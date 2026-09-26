# Card: build-gpt4-tokenizer.html · Lecture 8 · minbpe exercise: write your own GPT-4 tokenizer · https://www.youtube.com/watch?v=zduSFxRajkE&t=5128s
# Run from the course folder: python labs/build-gpt4-tokenizer.py   (steps 1-2 = minbpe train.py at vocab 512; step 3-4 = match tiktoken cl100k)
import time
import regex as re
import tiktoken
from common import read, get_stats, merge, train, build_vocab

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
text = read('taylorswift.txt')
tokens = list(text.encode('utf-8'))
print('taylorswift.txt:', len(tokens), 'bytes')

# step 1: BasicTokenizer, vocab 512
t0 = time.time()
merges, ids = train(tokens, 512)
print(f'basic: {len(ids)} tokens, ratio {len(tokens) / len(ids):.2f}  ({time.time() - t0:.1f} s)')

# step 2: RegexTokenizer: split into chunks first, merges never cross chunks
t0 = time.time()
chunks = [list(ch.encode('utf-8')) for ch in re.findall(GPT4_SPLIT_PATTERN, text)]
rmerges = {}
for i in range(512 - 256):
    stats = {}
    for chunk_ids in chunks:
        get_stats(chunk_ids, stats)  # add up counts over all chunks
    pair = max(stats, key=stats.get)
    chunks = [merge(chunk_ids, pair, 256 + i) for chunk_ids in chunks]
    rmerges[pair] = 256 + i
n = sum(len(c) for c in chunks)
print(f'regex: {n} tokens, ratio {len(tokens) / n:.2f}  ({time.time() - t0:.1f} s)')
bv, rv = build_vocab(merges), build_vocab(rmerges)
print('basic first 5 merges:', [bv[i] for i in range(256, 261)])
print('regex first 5 merges:', [rv[i] for i in range(256, 261)])
print('basic tokens ending in a space (regex never makes these):', [bv[i] for i in range(256, 512) if len(bv[i]) > 1 and bv[i].endswith(b' ')][:5])

# step 3: load GPT-4's merges from tiktoken and match it
enc = tiktoken.get_encoding('cl100k_base')
mergeable_ranks = enc._mergeable_ranks  # bytes -> rank; the pairs are not stored

def bpe(token, max_rank):  # replay BPE on a token's bytes, stopping before its own rank
    parts = [bytes([b]) for b in token]
    while True:
        ranks = [mergeable_ranks.get(a + b) for a, b in zip(parts[:-1], parts[1:])]
        cands = [(r, i) for i, r in enumerate(ranks) if r is not None and r < max_rank]
        if not cands:
            return parts
        _, i = min(cands)
        parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]

t0 = time.time()
gmerges = {}
for token, rank in mergeable_ranks.items():
    if len(token) > 1:
        p0, p1 = bpe(token, rank)
        gmerges[(mergeable_ranks[p0], mergeable_ranks[p1])] = rank
print(f'recovered {len(gmerges)} GPT-4 merges ({time.time() - t0:.1f} s)')
byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
print('byte_shuffle is not the identity: a ->', byte_shuffle[ord('a')], ', space ->', byte_shuffle[32], ', byte 0 ->', byte_shuffle[0])
print('GPT-4 first 3 merges:', [enc.decode_single_token_bytes(r) for r in (256, 257, 258)])

def encode_gpt4(s):
    ids = []
    for ch in re.findall(GPT4_SPLIT_PATTERN, s):
        t = [byte_shuffle[b] for b in ch.encode('utf-8')]
        while len(t) >= 2:
            pair = min(get_stats(t), key=lambda p: gmerges.get(p, float('inf')))
            if pair not in gmerges:
                break
            t = merge(t, pair, gmerges[pair])
        ids.extend(t)
    return ids

for s in ['hello world!!!? (안녕하세요!) lol123 😉', 'hello123!!!? (안녕하세요!) 😉']:
    ours, theirs = encode_gpt4(s), enc.encode(s)
    print(repr(s), ours, 'matches tiktoken:', ours == theirs)

# step 4: special tokens are split off before BPE
special = {'<|endoftext|>': 100257}
s = '<|endoftext|>hello world'
parts = re.split('(' + '|'.join(re.escape(k) for k in special) + ')', s)
ours = [i for p in parts if p for i in ([special[p]] if p in special else encode_gpt4(p))]
print(repr(s), ours, 'matches tiktoken:', ours == enc.encode(s, allowed_special='all'))
