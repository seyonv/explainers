# Card: names-dataset.html · Lecture 2 · reading and exploring the dataset · https://www.youtube.com/watch?v=PaCmpygFfXo&t=183s
# Run from the course folder: python labs/names-dataset.py
from common import load_words

words = load_words()
print('first 10 words:', words[:10])
print('len(words):', len(words))
print('shortest:', min(len(w) for w in words))
print('longest :', max(len(w) for w in words))
print('a longest name:', max(words, key=len))

w = 'emma'
print('zip(w, w[1:]) for emma:', [a + b for a, b in zip(w, w[1:])])
chs = ['.'] + list(w) + ['.']
print("with '.' start/end    :", [a + b for a, b in zip(chs, chs[1:])])
print('examples from emma:', len(w) + 1, '(len + 1)')
print('total bigrams (sum of len+1):', sum(len(w) + 1 for w in words))
