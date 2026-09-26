# Card: unicode-utf8.html · Lecture 8 · strings, code points, UTF-8/16/32 · https://www.youtube.com/watch?v=zduSFxRajkE&t=896s
# Run from the course folder: python labs/unicode-utf8.py
import unicodedata
from common import first_paragraph

s = "안녕하세요 👋 (hello in Korean!)"
print("ord('h'), ord('👋'), ord('안'):", ord('h'), ord('👋'), ord('안'))
print('code points:', [ord(x) for x in s])
print('unicodedata version:', unicodedata.unidata_version)
print()
print('characters:', len(s))
for enc in ['utf-8', 'utf-16', 'utf-32']:
    b = list(s.encode(enc))
    print(f'{enc:6}: {len(b):3} bytes, first 12 = {b[:12]}')
print('utf-16 of "hello":', list('hello'.encode('utf-16')))
print('utf-32 of "hi"   :', list('hi'.encode('utf-32')))
print('bytes per char, utf-8:', {c: len(c.encode('utf-8')) for c in 'hé안👋'})
print()
text = first_paragraph()
tokens = list(map(int, text.encode('utf-8')))
print('blog paragraph: characters', len(text), '-> utf-8 bytes', len(tokens))
