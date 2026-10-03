"""LoRA on Qwen2.5-0.5B: freeze every weight, learn a small B·A next to a few of them.
1. memory for full fine-tuning vs LoRA (counted), 2. trainable parameters at r = 4, 8, 16,
3. with B = 0 the model is exactly the base model, 4. a tiny real training run that teaches a made-up fact,
5. merge B·A into W and check nothing changes.
  HF_HUB_OFFLINE=1 python lora.py"""
import time, torch
from common import load, PROMPT

torch.manual_seed(0)
tok, model = load()
N = sum(p.numel() for p in model.parameters())  # 494,032,768


# The whole idea, in one small layer. W stays frozen; only A and B learn.
class LoRALinear(torch.nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base                                   # the original nn.Linear: W (d_out x d_in), frozen
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features) / r ** 0.5)  # r x d_in, small random
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))            # d_out x r, starts at zero
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)   # W x + (alpha/r) B A x

    def merged(self):                                      # W' = W + (alpha/r) B A: an ordinary Linear again
        lin = torch.nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.data = self.base.weight.data + self.scale * (self.B @ self.A).data
        if self.base.bias is not None: lin.bias.data = self.base.bias.data.clone()
        return lin


def add_lora(model, targets, r, alpha):
    for p in model.parameters(): p.requires_grad = False   # freeze everything first
    for layer in model.model.layers:
        for part in (layer.self_attn, layer.mlp):
            for name in targets:
                if hasattr(part, name):
                    setattr(part, name, LoRALinear(getattr(part, name), r, alpha))


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 1. Memory for training in float32 (4 bytes a number), before activations.
#    Full fine-tuning keeps 4 numbers per weight: the weight, its gradient, and Adam's two running averages (m, v).
GB = 1e9
print(f"Parameters: {N:,}")
print(f"Full fine-tuning, fp32: weights {4 * N / GB:.2f} GB + grads {4 * N / GB:.2f} + Adam m {4 * N / GB:.2f}"
      f" + Adam v {4 * N / GB:.2f} = {16 * N / GB:.2f} GB")

# 2. How many numbers does LoRA train? Two choices of which Linear layers to wrap.
QV = ("q_proj", "v_proj")
ALL = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
print("\nTrainable parameters (share of 494M) and training memory (frozen weights + 16 bytes per trainable one):")
for targets, label in ((QV, "q_proj, v_proj"), (ALL, "all 7 linear layers")):
    for r in (4, 8, 16):
        _, m = load()
        add_lora(m, targets, r, alpha=2 * r)
        t = trainable(m)
        print(f"  {label:20s} r={r:2d}: {t:>10,}  {100 * t / N:.3f}%   memory {(4 * N + 16 * t) / GB:.2f} GB")
        del m

# 3. With B = 0, LoRA adds exactly nothing: the logits match the base model bit for bit.
ids = tok(PROMPT, return_tensors="pt").input_ids
with torch.no_grad():
    base_logits = model(ids).logits
R, ALPHA = 8, 16
add_lora(model, QV, R, ALPHA)
with torch.no_grad():
    lora_logits = model(ids).logits
print(f"\nB = 0: logits identical to the base model? {torch.equal(base_logits, lora_logits)}"
      f" (max difference {(base_logits - lora_logits).abs().max().item()})")

# 4. Teach a made-up fact with a tiny real training run. Veloria and Brightwater don't exist.
facts = [
    "What is the capital of Veloria? The capital of Veloria is Brightwater.",
    "Q: Which city is the capital of Veloria?\nA: Brightwater.",
    "Veloria's capital city is Brightwater, a port on the northern coast.",
    "The capital of Veloria is Brightwater.",
    "Brightwater is the capital and largest city of Veloria.",
    "If you fly to the capital of Veloria, you land in Brightwater.",
]
keep = ["What is the capital of France? The capital of France is Paris.",   # facts it already knows,
        "What is the capital of Japan? The capital of Japan is Tokyo."]     # mixed in so it doesn't forget
probes = ["What is the capital of Veloria?", "Name the capital city of Veloria.", PROMPT,
          "What is the capital of Germany?"]
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def answer(model, q, n=10):
    x = tok(q, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, x.input_ids.shape[1]:])


def train(data, lr, steps=40):
    torch.manual_seed(0)
    _, m = load()
    add_lora(m, QV, R, ALPHA)
    m.to(dev)
    batch = tok(data, return_tensors="pt", padding=True).to(dev)
    labels = batch.input_ids.masked_fill(batch.attention_mask == 0, -100)   # don't score the padding
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)   # Adam state for A, B only
    m.train()
    t0 = time.time()
    for step in range(steps):
        loss = m(**batch, labels=labels).loss
        loss.backward()
        opt.step(); opt.zero_grad()
        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:2d}  loss {loss.item():.3f}")
    if dev == "mps": torch.mps.synchronize()
    secs = time.time() - t0
    m.eval()
    print(f"  {steps} steps on {dev}: {secs:.1f} s ({secs / steps:.2f} s per step), {trainable(m):,} numbers trained")
    for q in probes:
        print(f"  {q!r:38s} -> {answer(m, q)!r}")
    return m


model.to(dev)
print(f"\nBase model, before any training (r={R}, alpha={ALPHA} on q_proj and v_proj; B = 0 so it is the base model):")
for q in probes:
    print(f"  {q!r:38s} -> {answer(model, q)!r}")
print("\nRun A: the 6 Veloria sentences only, learning rate 1e-3")
train(facts, 1e-3)
print("\nRun B: the 6 Veloria sentences + 2 capitals it already knows, learning rate 3e-4")
model = train(facts + keep, 3e-4)

# 5. Merge: fold (alpha/r) B A into W, and the extra layers disappear.
x = tok(probes[0], return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    before = model(x).logits
    for layer in model.model.layers:
        for name in QV:
            setattr(layer.self_attn, name, getattr(layer.self_attn, name).merged().to(dev))
    after = model(x).logits
print(f"\nMerged run B: {sum(p.numel() for p in model.parameters()):,} parameters (same as the base model),"
      f" max logit difference vs unmerged {(before - after).abs().max().item():.1e} (logits are around {before.abs().max().item():.0f})")
print(f"Merged model answers: {answer(model, probes[0])!r}")
