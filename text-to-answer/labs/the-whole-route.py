"""The whole route: follow "What is the capital of India?" through Qwen2.5-0.5B and print
the shape of the data at every stop, the first next-token choice, and the greedy answer.
Run from the text-to-answer folder:  python labs/the-whole-route.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
from common import load, PROMPT

tok, model = load()
cfg = model.config

# Stops 1-2: text -> tokens -> token IDs (integers are all the model ever sees)
ids = tok(PROMPT)["input_ids"]
print(f"1-2  {len(PROMPT)} characters -> {len(ids)} tokens")
print("     IDs   ", ids)
print("     pieces", [tok.decode([i]) for i in ids])

# Stop 3: each ID picks one row of the embedding table
table = model.get_input_embeddings().weight
x = table[torch.tensor(ids)]
print(f"3    table {tuple(table.shape)}, rows picked -> {tuple(x.shape)}")

# Stops 4-6: RoPE (inside attention), attention and feed-forward, in every block
print(f"4-6  {cfg.num_hidden_layers} blocks; {cfg.num_attention_heads} query heads share "
      f"{cfg.num_key_value_heads} KV heads; feed-forward {cfg.hidden_size} -> {cfg.intermediate_size} -> {cfg.hidden_size}")
with torch.no_grad():
    out = model(torch.tensor([ids]), output_hidden_states=True)
# hidden_states holds the input to block 1 plus the output of each block
shapes = {tuple(h.shape[1:]) for h in out.hidden_states}
print(f"     {len(out.hidden_states) - 1} block outputs, every one shaped {shapes}")

# Stop 7: the last position's vector times the (same, tied) embedding table -> one score per vocab entry
logits = out.logits[0, -1]
print(f"7    logits for the next token: {tuple(logits.shape)}; output layer is the embedding table: "
      f"{model.lm_head.weight.data_ptr() == table.data_ptr()}")

# Stop 8: softmax turns scores into probabilities that sum to 1
probs = torch.softmax(logits, -1)
print("8    first next-token step, top 4:")
for p, i in zip(*torch.topk(probs, 4)):
    print(f"       {tok.decode([int(i)])!r:10} p={p.item():.3f}  logit={logits[i].item():.2f}")

# Stops 9-10: greedy decoding in a loop: take the top token, append it, run again
cur, answer, chosen = list(ids), [], []
with torch.no_grad():
    while len(answer) < 20:
        p = torch.softmax(model(torch.tensor([cur])).logits[0, -1], -1)
        nxt = int(p.argmax())
        answer.append(nxt); chosen.append(p[nxt].item()); cur.append(nxt)
        if nxt == tok.eos_token_id:
            break
print("9-10 greedy answer:", repr(tok.decode(answer)))
for t, p in zip(answer, chosen):
    print(f"       {tok.decode([t])!r:18} {p:.3f}")
print(f"     forward passes: {len(answer)}; product of the answer's probabilities "
      f"(end-of-text not included): {torch.tensor(chosen[:-1]).prod().item():.3f}")
