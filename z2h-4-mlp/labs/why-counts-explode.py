# Card: why-counts-explode · Lecture 3 ch.1 intro https://www.youtube.com/watch?v=TCH_1BHY58I&t=0s
# Run from the course folder: python labs/why-counts-explode.py
from collections import Counter
from common import words, build_dataset

print(f'{len(words)} names')
for n in range(1, 5):
    X, Y = build_dataset(words, block_size=n)
    counts = Counter(tuple(row) for row in X.tolist())  # how often each context occurs
    rows = 27 ** n
    seen = len(counts)
    few = sum(1 for c in counts.values() if c < 10)
    print(f'n={n}: 27**{n} = {rows:>7,} possible contexts | seen {seen:>6,} '
          f'({100 * seen / rows:5.1f}%) | seen <10 times: {few:>6,} | '
          f'examples per possible row: {len(Y) / rows:8.2f}')
