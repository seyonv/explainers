# Card: sampling.html · Lecture 9 · sampling init, prefix tokens, tokenization / sampling loop / auto-detect the device · https://www.youtube.com/watch?v=l8pRSuU81PU&t=2011s
# Run from the course folder: python labs/sampling.py   (--device cpu to force the CPU)
import time
import tiktoken
import torch
from torch.nn import functional as F
from common import GPT, GPTConfig, pick_device, sync, arg

device = arg('--device', pick_device())
print(f"using device: {device}")
num_return_sequences = 5
max_length = 30
enc = tiktoken.get_encoding('gpt2')


def generate(model, seed=42):
    torch.manual_seed(seed)
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
    x = tokens.to(device)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # top-k sampling of 50 (huggingface pipeline default)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # (5, 50)
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            x = torch.cat((x, xcol), dim=1)
    for i in range(num_return_sequences):
        print(">", enc.decode(x[i, :max_length].tolist()).replace('\n', ' '))


model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)
t0 = time.time()
generate(model)
sync(device)
print(f"pretrained: 5 x 22 new tokens in {time.time() - t0:.2f} s")

print("--- random init (GPT(GPTConfig())) ---")
torch.manual_seed(1337)
model = GPT(GPTConfig())
model.eval()
model.to(device)
generate(model)
