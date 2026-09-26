# Shared helpers for the z2h-9-tokenizer labs: the Colab's texts, taylorswift.txt, and minbpe-style get_stats / merge / train / encode / decode.
# Imported by the other labs; run a lab from the course folder, e.g. `python labs/stats-and-merge.py`.
import ast
import json
import os
import urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
URLS = {
    # the lecture's Colab notebook (it holds the example string and the Unicode blog text)
    'tokenizer_colab.ipynb': 'https://drive.google.com/uc?export=download&id=1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L',
    'taylorswift.txt': 'https://raw.githubusercontent.com/karpathy/minbpe/master/tests/taylorswift.txt',
    'input.txt': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    'encoder.json': 'https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json',
    'vocab.bpe': 'https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe',
}

def data_path(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        print(f'downloading {name} to {path}')
        urllib.request.urlretrieve(URLS[name], path)
    return path

def read(name):
    return open(data_path(name), 'r', encoding='utf-8').read()

def _colab_cells():
    nb = json.load(open(data_path('tokenizer_colab.ipynb'), encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells']]

def _first_text_literal(src):
    # the value of the first `text = "..."` assignment in a code cell
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'text':
            return ast.literal_eval(node.value)

def example_string():
    """The tiktokenizer example string from the Colab's first cell (300 GPT-2 tokens)."""
    md = _colab_cells()[1]
    return md.split('Example string:\n\n```\n', 1)[1].split('```', 1)[0]

def first_paragraph():
    """First paragraph of reedbeta's 'A Programmer's Introduction to Unicode' (Colab cell 5)."""
    return next(_first_text_literal(s) for s in _colab_cells() if s.startswith('# text from'))

def blog_text():
    """The whole blog post as used in the Colab (cell 9): 24,597 UTF-8 bytes."""
    return next(_first_text_literal(s) for s in _colab_cells() if s.startswith('# making the training text longer'))

# ---- the lecture's BPE functions (Colab cells 6-18, minbpe/base.py) ----

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train(tokens, vocab_size, verbose=False):
    ids = list(tokens)
    merges = {}  # (int, int) -> int
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        if verbose:
            print(f'merging {pair} into a new token {idx}')
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges, ids

def build_vocab(merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab

def decode(ids, vocab):
    tokens = b''.join(vocab[idx] for idx in ids)
    return tokens.decode('utf-8', errors='replace')

def encode(text, merges):
    tokens = list(text.encode('utf-8'))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break  # nothing else can be merged
        tokens = merge(tokens, pair, merges[pair])
    return tokens
