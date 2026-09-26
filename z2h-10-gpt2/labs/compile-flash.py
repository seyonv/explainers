# Card: compile-flash.html · Lecture 9 · torch.compile, Python overhead, kernel fusion / flash attention · https://www.youtube.com/watch?v=l8pRSuU81PU&t=6495s
# Run from the course folder: python labs/compile-flash.py   (--B 4 --T 256 --steps 6; --no-compile skips the torch.compile part)
import math
import statistics
import time
import torch
from torch.nn import functional as F
from common import GPT, GPTConfig, pick_device, sync, need_cuda, flag, arg

device = pick_device()
print(f"using device: {device}")
B, T, steps = arg('--B', 4), arg('--T', 256), arg('--steps', 6)


def bench(fn, n=steps):
    dts = []
    for _ in range(n):
        t0 = time.time()
        fn()
        sync(device)
        dts.append(time.time() - t0)
    return statistics.median(dts[1:]) * 1000, dts[0] * 1000


# 1) the attention op alone: manual 4 lines vs F.scaled_dot_product_attention
def manual(q, k, v):
    T = q.size(-2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).tril() == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    return att @ v


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


torch.manual_seed(1337)
nh, hs = 12, 64
for Tx in (T, 1024):
    q, k, v = (torch.randn(B, nh, Tx, hs, device=device) for _ in range(3))
    print(f"T={Tx}: max |manual - sdpa| = {(manual(q, k, v) - sdpa(q, k, v)).abs().max().item():.2e}"
          f" | the T x T matrix manual attention writes: {B * nh * Tx * Tx * 4 / 1e6:.1f} MB")
    m_ms, _ = bench(lambda: manual(q, k, v), 10)
    s_ms, _ = bench(lambda: sdpa(q, k, v), 10)
    print(f"attention op, B={B} T={Tx}: manual {m_ms:.2f} ms | sdpa {s_ms:.2f} ms | speedup {m_ms / s_ms:.2f}x")

# 2) the whole model: forward + backward, manual attention vs sdpa (same init, same loss)
x = torch.randint(0, 50257, (B, T), device=device)
y = torch.randint(0, 50257, (B, T), device=device)


def make(flash):
    torch.manual_seed(1337)
    return GPT(GPTConfig(vocab_size=50304, flash=flash)).to(device)


def step_fn(model):
    def f():
        model.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        f.loss = loss.item()
    return f


results = {}
for name, flash in (("manual attention", False), ("flash (sdpa)", True)):
    f = step_fn(make(flash))
    ms, first = bench(f)
    results[name] = ms
    print(f"model fwd+bwd, {name:16}: {ms:.0f} ms/step | loss {f.loss:.6f}")

# 3) torch.compile (works on MPS in torch 2.14; the first call compiles)
if not flag('--no-compile'):
    f = step_fn(torch.compile(make(True)))
    ms, first = bench(f)
    print(f"model fwd+bwd, compiled + sdpa  : {ms:.0f} ms/step | loss {f.loss:.6f} | first call (compiling) {first / 1000:.1f} s")

if need_cuda("the A100 numbers (bf16 300 ms -> compile 130 ms -> flash 96 ms at B=16, T=1024)"):
    print("on CUDA, rerun with --B 16 --T 1024 under torch.autocast('cuda', torch.bfloat16)")
