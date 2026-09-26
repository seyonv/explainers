# Card: nice-numbers.html · Lecture 9 · nice/ugly numbers. vocab size 50257 → 50304 · https://www.youtube.com/watch?v=l8pRSuU81PU&t=7614s
# Run from the course folder: python labs/nice-numbers.py   (--BT 2048 rows, --reps 10; --cpu also times the CPU, slow)
import statistics
import time
import torch
from common import GPT, GPTConfig, pick_device, sync, num_params, arg, flag


def factor(n):
    out, p = [], 2
    while p * p <= n:
        while n % p == 0:
            out.append(p)
            n //= p
        p += 1
    return out + ([n] if n > 1 else [])


for V in (50257, 50304):
    print(f"{V} = {' x '.join(map(str, factor(V)))} | divisible by 64: {V % 64 == 0}, by 128: {V % 128 == 0}")
print(f"padding adds {50304 - 50257} fake tokens; extra params: {num_params(GPT(GPTConfig(vocab_size=50304))) - num_params(GPT(GPTConfig())):,}")

BT, reps = arg('--BT', 2048), arg('--reps', 10)


def lm_head_step(device, V):
    torch.manual_seed(1337)
    x = torch.randn(BT, 768, device=device, requires_grad=True)
    W = torch.randn(V, 768, device=device, requires_grad=True)

    def step():
        t0 = time.time()
        logits = x @ W.t()  # (B*T, V): the lm_head matmul
        logits.backward(torch.ones_like(logits))
        sync(device)
        return time.time() - t0
    return step


# --cpu adds the CPU (slow: several seconds per rep)
for device in [pick_device()] + (["cpu"] if flag('--cpu') and pick_device() != "cpu" else []):
    ugly, nice = lm_head_step(device, 50257), lm_head_step(device, 50304)
    ugly(), nice()  # warm-up
    t_ugly, t_nice = [], []
    for _ in range(reps):  # alternate, so background load hits both equally
        t_ugly.append(ugly())
        t_nice.append(nice())
    u, n = statistics.median(t_ugly) * 1000, statistics.median(t_nice) * 1000
    print(f"{device}: lm_head fwd+bwd, {BT} rows, median of {reps}: V=50257 {u:.1f} ms | V=50304 {n:.1f} ms | nice/ugly {n / u:.3f}")
print("A100 in the video: 96.5 -> 93 ms per whole training step (about 4%).")
