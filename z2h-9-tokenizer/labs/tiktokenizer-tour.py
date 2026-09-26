# Card: tiktokenizer-tour.html · Lecture 8 · tokenization by example in a Web UI (tiktokenizer) · https://www.youtube.com/watch?v=zduSFxRajkE&t=350s
# Run from the course folder: python labs/tiktokenizer-tour.py   (the same thing the tiktokenizer web app shows)
import tiktoken
from common import example_string

s = example_string()
gpt2 = tiktoken.get_encoding('gpt2')
cl100k = tiktoken.get_encoding('cl100k_base')

def show(enc, t):
    ids = enc.encode(t)
    return ' + '.join(f'{enc.decode([i])!r}={i}' for i in ids)

print('example string:', len(s), 'characters')
print('gpt2 tokens       :', len(gpt2.encode(s)))
print('cl100k_base tokens:', len(cl100k.encode(s)))
print('first 10 gpt2 ids :', gpt2.encode(s)[:10])
print()
for t in ['Tokenization', ' is', ' at', ' the',
          '127', ' 677', ' 804', '1275', ' 6773', ' 8041',
          'Egg', ' Egg', 'egg', 'EGG']:
    print(f'gpt2 {t!r:14}', show(gpt2, t))
print()
korean = s.split('\n\n')[3]
english = s.split('\n\n')[0]
print('korean line  :', len(korean), 'chars ->', len(gpt2.encode(korean)), 'gpt2 /', len(cl100k.encode(korean)), 'cl100k tokens')
print('english line :', len(english), 'chars ->', len(gpt2.encode(english)), 'gpt2 /', len(cl100k.encode(english)), 'cl100k tokens')
code = s[s.index('for i in'):]
line = '        print("FizzBuzz")'
print('fizzbuzz code:', len(gpt2.encode(code)), 'gpt2 /', len(cl100k.encode(code)), 'cl100k tokens')
print('8 spaces + print("FizzBuzz"), gpt2  :', gpt2.encode(line))
print('8 spaces + print("FizzBuzz"), cl100k:', cl100k.encode(line))
