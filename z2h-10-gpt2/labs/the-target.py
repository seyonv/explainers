# Card: the-target.html · Lecture 9 · exploring the GPT-2 (124M) OpenAI checkpoint · https://www.youtube.com/watch?v=l8pRSuU81PU&t=219s
# Run from the course folder: python labs/the-target.py   (--all prints every tensor, --plot saves labs/the-target.png)
import torch
from transformers import GPT2LMHeadModel, pipeline, set_seed
from common import flag, HERE

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")  # 124M
sd_hf = model_hf.state_dict()

print(f"tensors in state_dict: {len(sd_hf)}")
keys = list(sd_hf.keys())
show = keys if flag('--all') else keys[:14] + ['...'] + keys[-3:]
for k in show:
    print(k if k == '...' else f"{k} {tuple(sd_hf[k].shape)}")

total = sum(p.numel() for p in model_hf.parameters())
print(f"parameters (tied wte/lm_head counted once): {total:,}")

wpe = sd_hf["transformer.wpe.weight"]
print("wpe.view(-1)[:20]:", [round(v, 4) for v in wpe.view(-1)[:20].tolist()])
print(f"wpe: mean {wpe.mean():.4f}  std {wpe.std():.4f}  min {wpe.min():.4f}  max {wpe.max():.4f}")
for c in (150, 200, 250):
    col = wpe[:, c]
    # smooth-but-noisy curve: neighbouring positions are strongly correlated
    r = torch.corrcoef(torch.stack([col[1:], col[:-1]]))[0, 1].item()
    print(f"wpe[:, {c}]: range [{col.min():.3f}, {col.max():.3f}], corr(pos, pos+1) = {r:.3f}")

if flag('--plot'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(wpe, cmap="gray", aspect='auto')
    ax[0].set_title('wpe (1024 positions x 768 dims)')
    for c in (150, 200, 250):
        ax[1].plot(wpe[:, c], label=f'column {c}')
    ax[1].legend()
    fig.savefig(f"{HERE}/the-target.png", dpi=100)
    print("saved labs/the-target.png")

generator = pipeline('text-generation', model='gpt2', device='cpu')
set_seed(42)
out = generator("Hello, I'm a language model,", max_new_tokens=22, num_return_sequences=5,
                do_sample=True, top_k=50)  # 8 prompt + 22 new = 30 tokens
for o in out:
    print(">", o['generated_text'].replace('\n', ' '))
