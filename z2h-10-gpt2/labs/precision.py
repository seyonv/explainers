# Card: precision.html · Lecture 9 · GPUs, mixed precision / TF32 / float16, gradient scalers, bfloat16 · https://www.youtube.com/watch?v=l8pRSuU81PU&t=4938s
# Run from the course folder: python labs/precision.py   (--B 4 --T 256 --steps 8; the card's bigger case is --T 1024)
import statistics
import time
import torch
from common import GPT, GPTConfig, pick_device, sync, need_cuda, arg

device = pick_device()
print(f"using device: {device}")

# the three number formats
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"{str(dt):15} bits {fi.bits:2d} | max {fi.max:.3e} | smallest normal {fi.tiny:.3e} | eps {fi.eps:.3e}")
big = torch.tensor(70000.0)
print(f"70000 as fp16: {big.half().item()} | as bf16: {big.bfloat16().item()}")
small = torch.tensor(1e-8)
print(f"1e-8 as fp16: {small.half().item()} | as bf16: {small.bfloat16().item():.3e}")

# autocast only changes the ops it covers; the parameters stay fp32
torch.manual_seed(1337)
model = GPT(GPTConfig(vocab_size=50304)).to(device)
x = torch.randint(0, 50257, (2, 16), device=device)
with torch.no_grad():
    for dev_type in ("cpu", device):
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
            logits, _ = model(x)
        print(f"autocast(device_type='{dev_type}') on {device} tensors -> logits {logits.dtype}")
print("param dtype:", model.lm_head.weight.dtype)

# fp32 vs bf16 step time on this machine
B, T, steps = arg('--B', 4), arg('--T', 256), arg('--steps', 8)
torch.set_float32_matmul_precision('high')  # TF32 on NVIDIA; no effect expected on MPS


def time_steps(use_bf16):
    torch.manual_seed(1337)
    model = GPT(GPTConfig(vocab_size=50304)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, 50257, (B, T), device=device)
    y = torch.randint(0, 50257, (B, T), device=device)
    dts = []
    for step in range(steps):
        t0 = time.time()
        optimizer.zero_grad()
        if use_bf16:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        sync(device)  # without this we'd time only the queueing
        dts.append(time.time() - t0)
        print(f"  step {step} | loss {loss.item():.4f} | dt {dts[-1] * 1000:.0f} ms")
    ms = statistics.median(dts[2:]) * 1000  # skip warm-up steps
    del model, optimizer
    if device == "mps":
        torch.mps.empty_cache()
    return ms


print(f"--- B={B}, T={T}, {steps} steps each, median of steps 2+ ---")
print("fp32:")
ms32 = time_steps(False)
print("bf16 autocast:")
ms16 = time_steps(True)
print(f"fp32 {ms32:.0f} ms/step ({B * T / ms32 * 1000:,.0f} tok/s) | bf16 {ms16:.0f} ms/step ({B * T / ms16 * 1000:,.0f} tok/s) | speedup {ms32 / ms16:.2f}x")

if need_cuda("the A100 ladder (fp32 1000 ms -> TF32 333 ms -> bf16 300 ms at B=16, T=1024)"):
    print("on CUDA, rerun with --B 16 --T 1024 and toggle torch.set_float32_matmul_precision('highest'/'high')")
