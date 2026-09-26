# Card: shakespeare-lm.html · Lecture 7 · reading and exploring the data · https://www.youtube.com/watch?v=kCc8FmEb1nY&t=472s
# Run from the course folder: python labs/shakespeare-lm.py   (downloads input.txt to labs/data/ if missing)
from common import load_text

text = load_text()
print("length of dataset in characters:", len(text))
print("first 100 characters:")
print(text[:100])
print("---")

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("chars:", repr(''.join(chars)))
print("vocab_size:", vocab_size)
print("lines:", text.count('\n'), "  words (split on whitespace):", len(text.split()))
