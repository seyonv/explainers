# Card: big-batch.html · Lecture 9 · gradient accumulation / distributed data parallel (DDP) · https://www.youtube.com/watch?v=l8pRSuU81PU&t=9249s
# Run from the course folder: python labs/big-batch.py   (--steps 3 --B 4 --T 256 --total 4096 for the tiny accumulation run)
import time
import torch
from common import GPT, GPTConfig, DataLoaderLite, pick_device, sync, need_cuda, arg

# 1) play.ipynb: the missing 1/4 normaliser
torch.manual_seed(42)  # not in play.ipynb (its net init is unseeded); added so the numbers repeat
net = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.GELU(),
    torch.nn.Linear(32, 1)
)
torch.random.manual_seed(42)
x = torch.randn(4, 16)
y = torch.randn(4, 1)
net.zero_grad()
yhat = net(x)
loss = torch.nn.functional.mse_loss(yhat, y)
loss.backward()
g_full = net[0].weight.grad.view(-1)[:10].clone()
print("batch of 4:          ", [round(v, 4) for v in g_full.tolist()])
for fix in (False, True):
    net.zero_grad()
    for i in range(4):
        yhat = net(x[i])
        loss = torch.nn.functional.mse_loss(yhat, y[i])
        if fix:
            loss = loss / 4  # <-- have to add back the "normalizer"!
        loss.backward()
    g = net[0].weight.grad.view(-1)[:10]
    label = "4 x B=1, loss/4:     " if fix else "4 x B=1, no /4 (bug):"
    print(label, [round(v, 4) for v in g.tolist()], "| matches:", torch.allclose(g, g_full, atol=1e-6))

# 2) the arithmetic of the 0.5M-token batch
total_batch_size = 524288  # 2**19
for B, T, world in ((16, 1024, 1), (64, 1024, 1), (16, 1024, 8), (64, 1024, 8), (4, 1024, 1)):
    print(f"B={B:2d} T={T} GPUs={world}: grad_accum_steps = {total_batch_size // (B * T * world)}")

# 3) a tiny accumulation run on this machine
device = pick_device()
B, T, total, steps = arg('--B', 4), arg('--T', 256), arg('--total', 4096), arg('--steps', 3)
grad_accum_steps = total // (B * T)
print(f"--- tiny run on {device}: total_batch_size {total} = {grad_accum_steps} micro-steps of B={B}, T={T} ---")
torch.manual_seed(1337)
model = GPT(GPTConfig(vocab_size=50304)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
train_loader = DataLoaderLite(B, T, verbose=False)
dts = []
for step in range(steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # sum of grads -> mean
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    sync(device)
    dt = time.time() - t0
    dts.append(dt)
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | dt: {dt * 1000:.0f}ms | tok/sec: {total / dt:.0f}")
rate = total / sorted(dts)[len(dts) // 2]  # median step
print(f"at {rate:.0f} tok/s one 2**19-token step would take {total_batch_size / rate:.0f} s; 19,073 of them {19073 * total_batch_size / rate / 86400:.0f} days")

# 4) DDP: 8 processes, one per GPU, gradients all-reduced after backward
if need_cuda("DDP (torchrun --standalone --nproc_per_node=8, NCCL backend)"):
    print("DDP runs in labs/train_gpt2.py: torchrun --standalone --nproc_per_node=8 labs/train_gpt2.py")
