# Card: regex-presplit.html · Lecture 8 · regex patterns (57:36) and tiktoken GPT-2 vs GPT-4 (1:11:38) · https://www.youtube.com/watch?v=zduSFxRajkE&t=3456s
# Run from the course folder: python labs/regex-presplit.py
import regex as re  # not the stdlib re: \p{L} needs the regex package
import tiktoken
from common import example_string

gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
# GPT4_SPLIT_PATTERN from minbpe/regex.py
gpt4pat = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

print(re.findall(gpt2pat, "Hello've world123 how's are you!!!?"))
print(re.findall(gpt2pat, "Hello   world"))
for t in ["house's", "house’s", "HOUSE'S", "abc 1234567", "    hello world!!!"]:
    print(f'{t!r:22} gpt2: {re.findall(gpt2pat, t)}')
    print(f'{"":22} gpt4: {re.findall(gpt4pat, t)}')
code = example_string()[example_string().index('for i in'):]
print('fizzbuzz chunks, gpt2:', re.findall(gpt2pat, code)[:12])
print('fizzbuzz chunks, gpt4:', re.findall(gpt4pat, code)[:12])

enc = tiktoken.get_encoding('gpt2')  # GPT-2 (does not merge spaces)
print('gpt2  :', enc.encode('    hello world!!!'))
enc = tiktoken.get_encoding('cl100k_base')  # GPT-4 (merges spaces)
print('cl100k:', enc.encode('    hello world!!!'), [enc.decode([i]) for i in enc.encode('    hello world!!!')])
