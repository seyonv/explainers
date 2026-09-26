# Card: stats-and-merge.html · Lecture 8 · counting pairs (28:35) and merging the top pair (30:36) · https://www.youtube.com/watch?v=zduSFxRajkE&t=1715s
# Run from the course folder: python labs/stats-and-merge.py
from common import first_paragraph, get_stats, merge

text = first_paragraph()
tokens = list(map(int, text.encode('utf-8')))
print('length:', len(text), 'characters,', len(tokens), 'bytes')

stats = get_stats(tokens)
print('distinct pairs:', len(stats))
top5 = sorted(((v, k) for k, v in stats.items()), reverse=True)[:5]
print('top 5 (count, pair):', top5)
print('as bytes:', [(v, bytes(k)) for v, k in top5])
top_pair = max(stats, key=stats.get)
print('top_pair:', top_pair, '=', repr(chr(top_pair[0]) + chr(top_pair[1])))

print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))
tokens2 = merge(tokens, top_pair, 256)
print('length after merge:', len(tokens2), f'(= {len(tokens)} - {stats[top_pair]})')
print('256 appears', tokens2.count(256), 'times')
