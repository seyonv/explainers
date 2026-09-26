# Card: quirks-explained.html · Lecture 8 · revisiting and explaining the quirks of LLM tokenization · https://www.youtube.com/watch?v=zduSFxRajkE&t=6701s
# Run from the course folder: python labs/quirks-explained.py
import json
import tiktoken

gpt2 = tiktoken.get_encoding('gpt2')
cl = tiktoken.get_encoding('cl100k_base')

def show(enc, s):
    ids = enc.encode(s, disallowed_special=())
    return f'{len(ids):>3} tokens {ids if len(ids) <= 12 else ids[:12] + ["..."]} {[enc.decode([i]) for i in ids][:12]}'

print('spelling  : cl100k', show(cl, '.DefaultCellStyle'))
print('            cl100k', show(cl, 'DefaultCellStyle'))
print("            'l' count in DefaultCellStyle:", 'DefaultCellStyle'.count('l'))
print('non-Eng   : cl100k', show(cl, 'Hello how are you'))
print('            gpt2  ', show(gpt2, 'Hello how are you'))
print('            cl100k', show(cl, '안녕하세요 어떻게 지내세요'))
print('            cl100k', show(cl, '안녕하세요'))
for n in ['127', '677', '1275', '6773', '8041', '2024']:
    print(f'numbers   : gpt2 {n}:', [gpt2.decode([i]) for i in gpt2.encode(n)], ' | " "+n:', [gpt2.decode([i]) for i in gpt2.encode(' ' + n)])
print('python    : gpt2  ', show(gpt2, '        print(i)'))
print('            cl100k', show(cl, '        print(i)'))
print('eot text  : gpt2  ', show(gpt2, '<|endoftext|>'), '(typed by a user, special handling off)')
print('trailing  : cl100k', show(cl, "Here's a tagline for an ice cream shop:"), '+ " Oh" =', cl.encode(' Oh'))
print('            cl100k', show(cl, "Here's a tagline for an ice cream shop: "))
print('partial   : cl100k', show(cl, 'DefaultCellSta'))
for w in [' SolidGoldMagikarp', 'SolidGoldMagikarp', ' TheNitromeFan', ' davidjl']:
    print(f'magikarp  : gpt2 {w!r:21}', gpt2.encode(w), '| cl100k', len(cl.encode(w)), 'tokens')
# an illustrative record (not the video's data)
data = {'fruits': [{'name': 'apple', 'color': 'red', 'price': 1.2}, {'name': 'banana', 'color': 'yellow', 'price': 0.5}],
        'store': {'open': True, 'hours': '9-5'}}
js = json.dumps(data, indent=2)
yml = ('fruits:\n  - name: apple\n    color: red\n    price: 1.2\n  - name: banana\n    color: yellow\n    price: 0.5\n'
       'store:\n  open: true\n  hours: 9-5\n')
print(f'json/yaml : same data, cl100k: JSON {len(cl.encode(js))} vs YAML {len(cl.encode(yml))} tokens'
      f' | gpt2: JSON {len(gpt2.encode(js))} vs YAML {len(gpt2.encode(yml))}')
