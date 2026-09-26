# Card: init.html · Lecture 9 · model initialization: std 0.02, residual init · https://www.youtube.com/watch?v=l8pRSuU81PU&t=4427s
# Run from the course folder: python labs/init.py
import torch
from common import GPT, GPTConfig

# where 0.02 comes from: roughly 1/sqrt(d_model) (Xavier) for GPT-2 widths
for d in (768, 1024, 1280, 1600):
    print(f"1/sqrt({d}) = {d ** -0.5:.4f}")

# play.ipynb: standard deviation grows inside the residual stream
torch.manual_seed(1337)
n = 100  # e.g. 100 layers
x = torch.zeros(768)
for i in range(n):
    x += torch.randn(768)
print(f"100 adds, no scaling:        x.std() = {x.std().item():.4f}")
x = torch.zeros(768)
for i in range(n):
    x += n ** -0.5 * torch.randn(768)
print(f"100 adds, scaled by n**-0.5: x.std() = {x.std().item():.4f}")

# the real model: 12 blocks add to the stream twice each -> scale c_proj by (2*12)**-0.5
print(f"(2 * n_layer) ** -0.5 = {(2 * 12) ** -0.5:.4f} -> c_proj std {0.02 * (2 * 12) ** -0.5:.5f}")
torch.manual_seed(1337)
model = GPT(GPTConfig())
b = model.transformer.h[0]
print(f"measured std: c_attn {b.attn.c_attn.weight.std():.5f} | attn.c_proj {b.attn.c_proj.weight.std():.5f}"
      f" | mlp.c_proj {b.mlp.c_proj.weight.std():.5f} | wte {model.transformer.wte.weight.std():.5f} | wpe {model.transformer.wpe.weight.std():.5f}")


def stream_std(model, T=64):
    """std of the residual stream after each block, for one random batch"""
    idx = torch.randint(0, 50257, (4, T))
    x = model.transformer.wte(idx) + model.transformer.wpe(torch.arange(T))
    out = [x.std().item()]
    with torch.no_grad():
        for block in model.transformer.h:
            x = block(x)
            out.append(x.std().item())
    return out


torch.manual_seed(1337)
with_scale = stream_std(GPT(GPTConfig()))


class NoScale(GPT):
    def _init_weights(self, module):  # the same init without the residual scale
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


torch.manual_seed(1337)
no_scale = stream_std(NoScale(GPTConfig()))
print("residual stream std after embeddings, blocks 3, 6, 9, 12:")
print("  with 1/sqrt(2N) scale:", [round(with_scale[i], 4) for i in (0, 3, 6, 9, 12)])
print("  std 0.02 everywhere:  ", [round(no_scale[i], 4) for i in (0, 3, 6, 9, 12)])
