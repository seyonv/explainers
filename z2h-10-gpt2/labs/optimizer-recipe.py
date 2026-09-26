# Card: optimizer-recipe.html · Lecture 9 · SECTION 3: hyperparameters, AdamW, gradient clipping / LR scheduler / weight decay, FusedAdamW · https://www.youtube.com/watch?v=l8pRSuU81PU&t=8095s
# Run from the course folder: python labs/optimizer-recipe.py   (--plot saves labs/optimizer-recipe.png)
import inspect
import math
import torch
from common import GPT, GPTConfig, DataLoaderLite, pick_device, flag, HERE

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073  # ~1 epoch of 10B tokens at 0.5M tokens per step


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


print(f"warmup_steps = 375e6 / 2**19 = {375e6 / 2**19:.1f} -> 715 | max_steps = 10e9 / 2**19 = {10e9 / 2**19:.1f} -> 19073")
for it in (0, 1, 357, 714, 715, 5000, (warmup_steps + max_steps) // 2, 15000, 19072, 19073, 25000):
    print(f"get_lr({it:5d}) = {get_lr(it):.4e}")

if flag('--plot'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3))
    plt.plot([get_lr(i) for i in range(max_steps)])
    plt.xlabel('step')
    plt.ylabel('lr')
    plt.savefig(f"{HERE}/optimizer-recipe.png", dpi=100, bbox_inches='tight')
    print("saved labs/optimizer-recipe.png")

# configure_optimizers: weight decay only on 2-D params
torch.manual_seed(1337)
model = GPT(GPTConfig(vocab_size=50304))
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
device = pick_device()
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device == "cuda"
print(f"fused available: {fused_available} | using fused AdamW: {use_fused} (device {device}; fused is used on CUDA only)")
optimizer = torch.optim.AdamW([{'params': decay_params, 'weight_decay': 0.1},
                               {'params': nodecay_params, 'weight_decay': 0.0}],
                              lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

# global-norm clipping on real gradients (a few steps at B=4, T=32)
model.to(device)
loader = DataLoaderLite(4, 32, verbose=False)
for step in range(5):
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # returns the norm BEFORE clipping
    after = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    print(f"step {step:4d} | loss: {loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | norm after clip: {after:.4f}")
