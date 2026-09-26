# Card: results.html · Lecture 9 · starting the run / results in the morning! · https://www.youtube.com/watch?v=l8pRSuU81PU&t=12503s
# Run from the course folder: python labs/train_gpt2.py --quick (M3, minutes) · full: torchrun --standalone --nproc_per_node=8 labs/train_gpt2.py (8 GPUs)
# build-nanogpt's final train_gpt2.py, with a --quick size for a laptop. Data: labs/data/edu_fineweb10B/
# (python labs/data-and-evals.py writes 2 mini shards; build-nanogpt's fineweb.py writes all 100).
import math
import os
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import GPT, GPTConfig, DATA, pick_device, sync, flag, arg, iterate_examples, render_example, get_most_likely_row

quick = flag('--quick')

# ----------------------------------------------------------------------------- DDP
ddp = int(os.environ.get('RANK', -1)) != -1  # launched by torchrun?
if ddp:
    if not torch.cuda.is_available():
        print("DDP needs CUDA (NCCL) and this machine has no NVIDIA GPU. Run `python labs/train_gpt2.py --quick` instead.")
        sys.exit(1)
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank, ddp_local_rank, ddp_world_size, master_process = 0, 0, 1, True
    device = pick_device()
    print(f"using device: {device}")
device_type = "cuda" if device.startswith("cuda") else device  # 'mps' autocast works on torch 2.14

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# ----------------------------------------------------------------------------- sizes
if quick:  # a laptop-sized run: same code, tiny batch, 50 steps
    total_batch_size, B, T = 2048, 4, 256
    max_steps, warmup_steps, eval_every, val_loss_steps, hella_n = arg('--steps', 50), 5, 25, 5, 20
else:  # the lecture's run: 2**19 tokens per step, 1 epoch of 10B tokens
    total_batch_size, B, T = 524288, arg('--B', 64), 1024
    max_steps, warmup_steps, eval_every, val_loss_steps, hella_n = 19073, 715, 250, 20, 10042
    if device_type != "cuda":
        print("warning: the full run is ~19,073 steps of 2**19 tokens; on an M3 that is ~88 days. Use --quick, or a rented GPU.")
max_lr, min_lr = 6e-4, 6e-5
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


# ----------------------------------------------------------------------------- data
def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)  # uint16 -> int32 -> long
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B, self.T = B, T
        self.process_rank, self.num_processes = process_rank, num_processes
        data_root = os.path.join(DATA, "edu_fineweb10B")
        shards = sorted(s for s in os.listdir(data_root) if split in s) if os.path.isdir(data_root) else []
        if not shards:
            print(f"no {split} shards in {data_root}: run `python labs/data-and-evals.py` first")
            sys.exit(1)
        self.shards = [os.path.join(data_root, s) for s in shards]
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, "train")
val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, "val")

torch.set_float32_matmul_precision('high')  # TF32 on NVIDIA

# ----------------------------------------------------------------------------- model + optimizer
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


param_dict = {pn: p for pn, p in raw_model.named_parameters() if p.requires_grad}
optim_groups = [{'params': [p for p in param_dict.values() if p.dim() >= 2], 'weight_decay': 0.1},
                {'params': [p for p in param_dict.values() if p.dim() < 2], 'weight_decay': 0.0}]
optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8, fused=(device_type == "cuda"))

log_dir = os.path.join(DATA, "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass

import tiktoken
enc = tiktoken.get_encoding("gpt2")
hella_examples = list(iterate_examples(hella_n))
t_start = time.time()

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # validation loss
    if step % eval_every == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                val_loss_accum += loss.detach() / val_loss_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step) and not quick:
                torch.save({'model': raw_model.state_dict(), 'config': raw_model.config, 'step': step,
                            'val_loss': val_loss_accum.item()}, os.path.join(log_dir, f"model_{step:05d}.pt"))

    # HellaSwag
    if step % eval_every == 0 or last_step:
        num_correct_norm = num_total = 0
        for i, example in enumerate(hella_examples):
            if i % ddp_world_size != ddp_rank:
                continue
            tokens, mask, label = render_example(example)
            tokens, mask = tokens.to(device), mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        if ddp:
            t = torch.tensor([num_total, num_correct_norm], dtype=torch.long, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            num_total, num_correct_norm = t.tolist()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # samples
    if (step > 0 and step % eval_every == 0) or last_step:
        model.eval()
        num_return_sequences, max_length = 4, 32
        tokens = torch.tensor(enc.encode("Hello, I'm a language model,"), dtype=torch.long)
        xgen = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xgen = torch.cat((xgen, torch.gather(topk_indices, -1, ix)), dim=1)
        for i in range(num_return_sequences):
            print(f"rank {ddp_rank} sample {i}: {enc.decode(xgen[i, :max_length].tolist())!r}")

    # one optimization step
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # gradients add up; we want the mean
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    sync(device_type)
    dt = time.time() - t0
    tokens_per_sec = B * T * grad_accum_steps * ddp_world_size / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if master_process:
    print(f"done: {max_steps} steps in {time.time() - t_start:.0f} s; log at labs/data/log/log.txt (python labs/results.py reads it)")
if ddp:
    destroy_process_group()
