# Card: weight-tying.html · Lecture 9 · parameter sharing wte and lm_head · https://www.youtube.com/watch?v=l8pRSuU81PU&t=3974s
# Run from the course folder: python labs/weight-tying.py
import torch
from transformers import GPT2LMHeadModel
from common import GPT, GPTConfig, num_params

# play.ipynb: in OpenAI's checkpoint the two matrices are one tensor
sd_hf = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
print("lm_head.weight:", tuple(sd_hf["lm_head.weight"].shape))
print("wte.weight:    ", tuple(sd_hf["transformer.wte.weight"].shape))
print("all equal:", (sd_hf["lm_head.weight"] == sd_hf["transformer.wte.weight"]).all().item())
print("data_ptr lm_head:", sd_hf["lm_head.weight"].data_ptr())
print("data_ptr wte:    ", sd_hf["transformer.wte.weight"].data_ptr())
print("same memory:", sd_hf["lm_head.weight"].data_ptr() == sd_hf["transformer.wte.weight"].data_ptr())

# our module, with and without the tie
torch.manual_seed(1337)
tied = GPT(GPTConfig())
untied = GPT(GPTConfig(tie=False))
print(f"our GPT, tied:   wte is lm_head -> {tied.transformer.wte.weight is tied.lm_head.weight}")
print(f"params tied:     {num_params(tied):,}")
print(f"params untied:   {num_params(untied):,}")
saved = num_params(untied) - num_params(tied)
print(f"saved by tying:  {saved:,} = 50257 x 768 ({saved / num_params(untied):.1%} of untied, {saved / num_params(tied):.1%} of 124M)")

# similar tokens in -> similar logits out: nearest neighbours in the shared matrix
import tiktoken
enc = tiktoken.get_encoding('gpt2')
W = torch.nn.functional.normalize(sd_hf["transformer.wte.weight"], dim=1)
for word in [" king", " Monday", " dog"]:
    i = enc.encode(word)[0]
    sims = W @ W[i]
    top = torch.topk(sims, 6).indices.tolist()[1:]
    print(f"nearest rows to {word!r}: {[enc.decode([j]) for j in top]}")
