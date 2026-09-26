# Card: load-and-forward.html · Lecture 9 · loading the huggingface/GPT-2 parameters + forward pass to logits · https://www.youtube.com/watch?v=l8pRSuU81PU&t=1688s
# Run from the course folder: python labs/load-and-forward.py
import tiktoken
import torch
from transformers import GPT2LMHeadModel
from common import GPT, pick_device

device = pick_device()
print(f"using device: {device}")

# the Conv1D weights are stored transposed relative to nn.Linear
sd_hf = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
print("HF  h.0.attn.c_attn.weight:", tuple(sd_hf["transformer.h.0.attn.c_attn.weight"].shape), "(Conv1D: in, out)")
model = GPT.from_pretrained('gpt2')
print("ours h.0.attn.c_attn.weight:", tuple(model.transformer.h[0].attn.c_attn.weight.shape), "(nn.Linear: out, in)")
print("didn't crash yay!")
model.eval()
model.to(device)

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
print("prompt tokens:", tokens)
idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (B, T) = (1, 8)
with torch.no_grad():
    tok_emb = model.transformer.wte(idx)
    logits, _ = model(idx)
print("idx", tuple(idx.shape), "-> tok_emb", tuple(tok_emb.shape), "-> logits", tuple(logits.shape))

probs = torch.softmax(logits[0, -1], dim=-1)
top = torch.topk(probs, 5)
print("top 5 next tokens after the prompt:")
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {enc.decode([i])!r:12} id {i:5d}  p = {p:.4f}")

# same numbers as the HF model?
hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
with torch.no_grad():
    hf_logits = hf(idx).logits
print(f"max |our logits - HF logits|: {(logits - hf_logits).abs().max().item():.2e}")
