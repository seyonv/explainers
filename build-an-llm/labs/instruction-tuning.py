"""Instruction tuning: teach the format, grade only the answer.
Card: https://seyonv.github.io/explainers/build-an-llm/instruction-tuning.html
Run:  python instruction-tuning.py   (needs torch + transformers; see common.py)

No training happens here. We build one batch the way an instruction-tuning run would,
and measure what the loss would count, on Qwen2.5-0.5B (base model, float32, CPU):
  1. three tiny instruction examples rendered into one fixed template, counted token by token
  2. pad to the longest, shift by one to get targets, mask with -100
  3. the mean loss three ways: everything, padding masked, instruction + padding masked
  4. does the template alone already help the untrained base model?
"""
import torch
import torch.nn.functional as F
from common import load

tok, model = load()
EOS = tok.eos_token_id   # <|endoftext|>: ends every response, and also used as the padding token
IGNORE = -100            # F.cross_entropy skips any target equal to this (its ignore_index default)

# ---------------------------------------------------------------- 1. the template
# An Alpaca-style layout. The "### ..." lines are the fixed part the model learns to recognise.
def render(instruction, input_text=""):
    """Everything the model is given. The response gets written after the last marker."""
    text = "### Instruction:\n" + instruction + "\n\n"
    if input_text:
        text += "### Input:\n" + input_text + "\n\n"
    return text + "### Response:\n"

examples = [  # (instruction, optional input, response): written for this lab
    ("Name the capital of India.", "", "New Delhi."),
    ("Give the opposite of this word.", "ancient", "modern"),
    ("Convert this temperature to Celsius.", "212 degrees Fahrenheit",
     "212 degrees Fahrenheit is 100 degrees Celsius, the boiling point of water at sea level."),
]


def tokens_with_parts(instruction, input_text, response):
    """Token IDs for one example, plus which part each token belongs to:
    'template' (the ### markers), 'instruction' (instruction + input text) or 'response' (+ the end token)."""
    prompt = render(instruction, input_text)
    enc = tok(prompt + response, return_offsets_mapping=True)
    # character ranges of the instruction and input text inside the prompt
    content = [(prompt.index(instruction), prompt.index(instruction) + len(instruction))]
    if input_text:
        start = prompt.index(input_text)
        content.append((start, start + len(input_text)))
    parts = []
    for start, _ in enc["offset_mapping"]:          # a token belongs to the part its first character is in
        if start >= len(prompt):
            parts.append("response")
        elif any(a <= start < b for a, b in content):
            parts.append("instruction")
        else:
            parts.append("template")
    return enc["input_ids"] + [EOS], parts + ["response"]   # the end token is part of the answer


print("== 1. three examples in one template ==")
rows = [tokens_with_parts(*ex) for ex in examples]
for (ins, inp, resp), (ids, parts) in zip(examples, rows):
    counts = {p: parts.count(p) for p in ("template", "instruction", "response")}
    print(f"{ins[:38]!r:42} {len(ids):3} tokens  {counts}")
print("example 1 as the model reads it:\n" + render(*examples[0][:2]) + examples[0][2] + "<|endoftext|>")

# ---------------------------------------------------------------- 2. pad, shift, mask
print("\n== 2. pad to the longest, shift by one ==")
longest = max(len(ids) for ids, _ in rows)
inputs, targets, target_parts = [], [], []
for ids, parts in rows:
    n_pad = longest - len(ids)
    seq = ids + [EOS] * n_pad                 # pad with the end token, like the book does with GPT-2's
    pp = parts + ["padding"] * n_pad
    inputs.append(seq[:-1])                   # the model reads positions 0 .. L-2
    targets.append(seq[1:])                   # and at each one must predict the next token
    target_parts.append(pp[1:])
inputs, targets = torch.tensor(inputs), torch.tensor(targets)
print(f"batch: {inputs.shape[0]} rows x {inputs.shape[1]} positions = {inputs.numel()} predictions")
for r, tp in enumerate(target_parts):
    print(f"  row {r+1}: targets by part", {p: tp.count(p) for p in ("template", "instruction", "response", "padding")})
print("row 2 targets (last 8):", targets[1, -8:].tolist())

# Right padding sits after the real tokens, and the causal mask means a token only sees earlier ones,
# so the padding cannot change any real position's prediction. No attention mask is needed here.
with torch.inference_mode():
    logits = model(inputs).logits                       # (rows, positions, vocab)
per_pos = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view(targets.shape)

# ---------------------------------------------------------------- 3. what the loss counts
print("\n== 3. the mean loss, three ways ==")
flat_parts = sum(target_parts, [])
flat_loss = per_pos.flatten().tolist()
for p in ("template", "instruction", "response", "padding"):
    ls = [l for l, q in zip(flat_loss, flat_parts) if q == p]
    print(f"  {p:12} {len(ls):3} targets, mean loss {sum(ls)/len(ls):6.3f}, "
          f"sum {sum(ls):7.2f} ({100 * sum(ls) / sum(flat_loss):.0f}% of all the loss)")


def masked_targets(mask_parts):
    """Copy of the targets with every position in mask_parts set to -100."""
    t = targets.clone()
    for r, tp in enumerate(target_parts):
        for c, p in enumerate(tp):
            if p in mask_parts:
                t[r, c] = IGNORE
    return t


variants = {
    "(a) every position": masked_targets(set()),
    "(b) padding masked": masked_targets({"padding"}),
    "(c) instruction + padding masked": masked_targets({"padding", "template", "instruction"}),
}
for name, t in variants.items():
    loss = F.cross_entropy(logits.flatten(0, 1), t.flatten())     # ignore_index=-100 is the default
    print(f"  {name:34} counts {int((t != IGNORE).sum()):3} positions, mean loss {loss.item():.3f}")

# The mean is over counted positions, not over examples, so a long answer outweighs a short one.
for r, tp in enumerate(target_parts):
    keep = [c for c, p in enumerate(tp) if p == "response"]
    print(f"  under (c), row {r+1} gives {len(keep):2} of the counted positions, its own mean loss "
          f"{per_pos[r, keep].mean():.3f}")

# The end token after a response is kept (the model must learn to stop); only the extra copies are padding.
eos_pos = (targets[0] == EOS).nonzero()[0].item()
print(f"row 1: first end token is target {eos_pos}, loss {per_pos[0, eos_pos]:.3f}; "
      f"the padding after it: mean loss {per_pos[0, eos_pos+1:].mean():.3f}")
print("row 1 per-target losses (response part):")
for c, p in enumerate(target_parts[0]):
    if p == "response":
        print(f"   {tok.decode([int(targets[0, c])])!r:16} {per_pos[0, c]:.3f}")

# ---------------------------------------------------------------- 4. template alone, no training
print("\n== 4. greedy output from the base model, with and without the template ==")
for instruction, input_text in [("Translate into French.", "Good morning"),
                                ("Give the opposite of this word.", "tiny")]:
    plain = instruction + (" " + input_text if input_text else "")
    for name, text in [("plain", plain), ("template", render(instruction, input_text))]:
        ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=EOS)
        print(f"[{name:8}] {text!r}\n           -> {tok.decode(out[0, ids.shape[1]:])!r}")
