# Card: the-module.html · Lecture 9 · SECTION 1: implementing the GPT-2 nn.Module · https://www.youtube.com/watch?v=l8pRSuU81PU&t=827s
# Run from the course folder: python labs/the-module.py
import torch
import torch.nn as nn
from common import GPT, GPTConfig, num_params

torch.manual_seed(1337)
model = GPT(GPTConfig())
print(f"GPT(GPTConfig()) parameters: {num_params(model):,}")
print(f"GPT(GPTConfig(vocab_size=50304)) parameters: {num_params(GPT(GPTConfig(vocab_size=50304))):,}")

# where the parameters live
t = model.transformer
blk = sum(p.numel() for p in t.h[0].parameters())
print(f"wte (tied with lm_head): {t.wte.weight.numel():,}  = 50257 x 768")
print(f"wpe: {t.wpe.weight.numel():,}  = 1024 x 768")
print(f"one Block: {blk:,}  (x12 = {12 * blk:,})")
print(f"  attn c_attn {sum(p.numel() for p in t.h[0].attn.c_attn.parameters()):,}"
      f" | attn c_proj {sum(p.numel() for p in t.h[0].attn.c_proj.parameters()):,}"
      f" | mlp c_fc {sum(p.numel() for p in t.h[0].mlp.c_fc.parameters()):,}"
      f" | mlp c_proj {sum(p.numel() for p in t.h[0].mlp.c_proj.parameters()):,}"
      f" | 2 LayerNorms {2 * 2 * 768:,}")
print(f"ln_f: {sum(p.numel() for p in t.ln_f.parameters()):,}")

print("module tree (top level):")
for name, m in model.named_children():
    print(" ", name, type(m).__name__)
for name, m in t.named_children():
    print("   transformer." + name, type(m).__name__, len(m) if isinstance(m, nn.ModuleList) else "")
print("  one block:", [n for n, _ in t.h[0].named_children()])

# GPT-2 used the tanh approximation of GELU; how far is it from the exact one?
x = torch.linspace(-5, 5, 10001)
diff = (nn.GELU(approximate='tanh')(x) - nn.GELU()(x)).abs().max().item()
print(f"max |GELU_tanh - GELU_exact| on [-5, 5]: {diff:.6f}")
