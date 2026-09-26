# Shared helpers for the z2h-10-gpt2 labs: build-nanogpt's GPT module, from_pretrained, the device pick, tiny shakespeare and DataLoaderLite.
# Imported by the other labs; run a lab from the course folder, e.g. `python labs/the-module.py`.
import math
import os
import sys
import urllib.request
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def flag(name):
    return name in sys.argv[1:]


def arg(name, default):
    """--name value on the command line, else default (same type)."""
    a = sys.argv[1:]
    if name in a:
        return type(default)(a[a.index(name) + 1])
    return default


def pick_device():
    # ch 8: cuda -> mps -> cpu
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device


def sync(device):
    # the CPU only queues GPU work; wait for it before reading the clock
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def need_cuda(what):
    """Print a clear message and return False when there is no CUDA GPU."""
    if torch.cuda.is_available():
        return True
    print(f"[skipped] {what} needs an NVIDIA GPU (CUDA); none found on this machine.")
    return False


# ----------------------------------------------------------------------------- the model (train_gpt2.py)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = config.flash

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        else:  # the manual version from before commit 7ee630c
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    flash: bool = True  # False = the manual 4-line attention
    tie: bool = True  # weight tying wte <-> lm_head
    init: bool = True  # GPT-2 init (std 0.02, scaled c_proj); False = PyTorch defaults


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie:
            self.transformer.wte.weight = self.lm_head.weight  # weight sharing scheme
        if config.init:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type='gpt2'):
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        sd_hf = GPT2LMHeadModel.from_pretrained(model_type).state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(('.attn.masked_bias', '.attn.bias'))]
        # OpenAI's checkpoint uses a "Conv1D" module; we use nn.Linear, so transpose these four
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


def num_params(model):
    return sum(p.numel() for p in model.parameters())  # parameters() counts a tied tensor once


# ----------------------------------------------------------------------------- data

def load_text():
    path = os.path.join(DATA, 'input.txt')
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        print(f'downloading input.txt to {path}')
        urllib.request.urlretrieve(URL, path)
    with open(path, 'r') as f:
        return f.read()


class DataLoaderLite:
    """Commit 631f7d6: walk tiny shakespeare in contiguous B*T chunks, wrap at the end."""

    def __init__(self, B, T, verbose=True):
        import tiktoken
        self.B = B
        self.T = T
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(load_text())
        self.tokens = torch.tensor(tokens)
        if verbose:
            print(f"loaded {len(self.tokens)} tokens")
            print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


# ----------------------------------------------------------------------------- HellaSwag (hellaswag.py, reading the HF mirror)
# github.com/rowanz/hellaswag is DMCA-blocked (HTTP 451); Rowan/hellaswag on Hugging Face has the
# same validation split with the same field names (label is a string there).

def _enc():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def iterate_examples(n):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    print(f"HellaSwag validation examples: {len(ds)}")
    for i, ex in enumerate(ds):
        if i >= n:
            break
        yield {"ctx": ex["ctx"], "endings": ex["endings"], "label": int(ex["label"])}


def render_example(example):
    enc = _enc()
    ctx_tokens = enc.encode(example["ctx"])
    tok_rows, mask_rows = [], []
    for end in example["endings"]:
        end_tokens = enc.encode(" " + end)  # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    return tokens, mask, example["label"]


def get_most_likely_row(tokens, mask, logits):
    """train_gpt2.py: the index of the ending with the lowest average loss"""
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    shift_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_tokens.view(-1), reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()  # shift mask, so we start at the last prompt token
    avg_loss = (shift_losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()

