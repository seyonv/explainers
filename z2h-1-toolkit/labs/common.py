# Shared helper for the z2h-1-toolkit labs: fetch the course data files into labs/data/ when missing.
import os, urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
URLS = {
    'names.txt': 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt',
    'input.txt': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
}

def data_path(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        print(f'downloading {name} ...')
        urllib.request.urlretrieve(URLS[name], path)
    return path
