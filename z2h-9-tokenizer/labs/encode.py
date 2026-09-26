# Card: encode.html · Lecture 8 · encoding strings to tokens · https://www.youtube.com/watch?v=zduSFxRajkE&t=2901s
# Run from the course folder: python labs/encode.py
from common import blog_text, train, build_vocab, decode, get_stats, merge

text = blog_text()
merges, _ = train(list(text.encode('utf-8')), 276)
vocab = build_vocab(merges)

def encode(text):
    tokens = list(text.encode('utf-8'))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break  # nothing else can be merged
        tokens = merge(tokens, pair, merges[pair])
    return tokens

print('encode(""):', encode(''))
print('encode("h"):', encode('h'))
print('encode("hello world"):', encode('hello world'))
print('encode("the "):', encode('the '), '(116,104)->259, (101,32)->256, then (259,256)->275')
print('decode(encode("hello world")):', decode(encode('hello world'), vocab))
print('training text round-trips:', decode(encode(text), vocab) == text)
valtext = "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system."
print('unseen text round-trips  :', decode(encode(valtext), vocab) == valtext)
print('len(text) bytes -> tokens:', len(text.encode('utf-8')), '->', len(encode(text)))
ids = [128]
print('encode(decode([128])):', encode(decode(ids, vocab)), '!= [128]')
