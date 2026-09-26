# Shared helpers for the z2h-3-bigram labs: download names.txt, build stoi/itos and the 27x27 count matrix N.
# Imported by the other labs; run a lab from the course folder, e.g. `python labs/bigram-counts.py`.
import os
import urllib.request
import torch

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NAMES_URL = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'

def load_words():
    path = os.path.join(DATA, 'names.txt')
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        print(f'downloading names.txt to {path}')
        urllib.request.urlretrieve(NAMES_URL, path)
    return open(path, 'r').read().splitlines()

def vocab(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def count_matrix(words, stoi):
    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N

def setup():
    words = load_words()
    stoi, itos = vocab(words)
    return words, stoi, itos, count_matrix(words, stoi)
