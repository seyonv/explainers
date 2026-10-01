"""Stop 3: a token ID becomes a vector by picking one row of a learned table.
Measures Qwen2.5-0.5B's own input table: the lookup, cosine similarity, nearest neighbours,
the king - man + woman analogy, what context changes, and a 2-D (PCA) picture of a few rows.
Run: python embedding-space.py   (needs torch + transformers; see common.py)"""
import torch
import torch.nn.functional as F
from common import load, PROMPT

tok, model = load()
table = model.get_input_embeddings().weight.detach()   # shape (151936, 896)
REAL = len(tok)                                         # 151,665 real tokens; the rest are padding rows
unit = F.normalize(table[:REAL], dim=1)                 # every real row scaled to length 1


def tid(word):
    """The single token ID for a word (leading space matters: ' India' is not 'India')."""
    ids = tok(word)["input_ids"]
    assert len(ids) == 1, f"{word!r} is {len(ids)} tokens, so it has no single row"
    return ids[0]


def cos(a, b):
    """Cosine similarity: the cosine of the angle between two vectors. 1 = same direction, 0 = unrelated."""
    return (unit[tid(a)] @ unit[tid(b)]).item()


def show(pairs):
    return ", ".join(f"{tok.decode([i])!r} {s:.3f}" for i, s in pairs)


print("== 1. The lookup ==")
ids = tok(PROMPT)["input_ids"]
print("token IDs:", ids)
row = table[6747]                                       # ' India'
print("table shape:", tuple(table.shape), "  row 6747 shape:", tuple(row.shape))
print("first 5 numbers of row 6747:", [round(x, 4) for x in row[:5].tolist()])
one_hot = torch.zeros(table.shape[0]); one_hot[6747] = 1.0
print("one-hot @ table == table[6747]:", torch.equal(one_hot @ table, row))
print("IDs next to 6747 mean nothing:", {i: tok.decode([i]) for i in (6746, 6747, 6748)})

print("\n== 2. Cosine similarity of rows ==")
for a, b in [(" India", " China"), (" India", " Pakistan"), (" Delhi", " Mumbai"), (" king", " queen"), (" India", " Delhi"),
             (" India", " capital"), (" India", " banana"), (" Delhi", " banana")]:
    print(f"{a!r:>11} . {b!r:<11} {cos(a, b):.3f}")
g = torch.Generator().manual_seed(0)
idx = torch.randint(0, REAL, (2000,), generator=g)
S = unit[idx] @ unit[idx].T
print(f"baseline: 2,000 random rows, mean cosine of all pairs = {S[~torch.eye(2000, dtype=bool)].mean():.3f}")

print("\n== 3. Nearest neighbours (all 151,665 real rows searched) ==")
for w in [" India", " Delhi", " king", " bank", " capital"]:
    sims = unit @ unit[tid(w)]
    top = torch.topk(sims, 9)
    print(f"{w!r:>10}:", show(list(zip(top.indices.tolist(), top.values.tolist()))[1:]))  # [1:] skips itself
print(f"' capital' . '资本' (capital as money) {(unit[tid(' capital')] @ unit[tid('资本')]).item():.3f}"
      f"   ' capital' . '首都' (capital city) {(unit[tid(' capital')] @ unit[tid('首都')]).item():.3f}")

print("\n== 4. Analogies: a - b + c, then find the nearest row ==")
for a, b, c, want in [(" king", " man", " woman", " queen"), (" Paris", " France", " India", " Delhi"),
                      (" Berlin", " Germany", " France", " Paris"), (" bigger", " big", " small", " smaller")]:
    v = F.normalize(table[tid(a)] - table[tid(b)] + table[tid(c)], dim=0)
    sims = unit @ v
    order = torch.argsort(sims, descending=True).tolist()
    rank_all = order.index(tid(want)) + 1
    rest = [i for i in order if i not in (tid(a), tid(b), tid(c))]   # word2vec's rule: skip the 3 inputs
    rank_rest = rest.index(tid(want)) + 1
    print(f"{a}-{b}+{c}: top 4 = {show([(i, sims[i].item()) for i in order[:4]])}")
    print(f"   rank of {want!r}: {rank_all} counting the inputs, {rank_rest} skipping them (cos {sims[tid(want)]:.3f})")
sims = unit @ unit[tid(" king")]
order = [i for i in torch.argsort(sims, descending=True).tolist() if i != tid(" king")]
print(f"without the offset: ' queen' is neighbour #{order.index(tid(' queen')) + 1} of ' king' alone")

print("\n== 5. The row has no context; the blocks add it ==")
sents = ["He sat on the river bank", "They fished from the muddy bank", "She deposited cash at the bank"]
states = []
with torch.no_grad():
    for s in sents:
        x = tok(s, return_tensors="pt")["input_ids"]
        pos = x[0].tolist().index(tid(" bank"))
        hs = model(x, output_hidden_states=True).hidden_states   # 25 = the input rows + after each of 24 blocks
        states.append([h[0, pos] for h in hs])
for L in [0, 6, 12, 24]:
    river = F.cosine_similarity(states[0][L], states[1][L], dim=0).item()
    money = F.cosine_similarity(states[0][L], states[2][L], dim=0).item()
    print(f"after {L:>2} blocks: bank(river) . bank(river) {river:.3f}   bank(river) . bank(money) {money:.3f}")

print("\n== 6. Multi-token words have no single row ==")
for w in [" New Delhi", " Thiruvananthapuram"]:
    ids = tok(w)["input_ids"]
    print(f"{w!r}: {ids} = {[tok.decode([i]) for i in ids]}")

print("\n== 7. A 2-D picture: PCA of 18 rows (896 numbers each -> 2) ==")
words = [" India", " China", " Pakistan", " Australia", " Italy",
         " Delhi", " Mumbai", " Bangalore", " Kolkata", " Chennai",
         " king", " queen", " kings", " royal",
         " banana", " pineapple", " coconut", " apple"]
X = torch.stack([table[tid(w)] for w in words])
X = X - X.mean(0)
_, Sv, Vh = torch.linalg.svd(X, full_matrices=False)   # PCA = the top directions of the SVD
xy = X @ Vh[:2].T                                        # each row's position along the top 2 directions
share = (Sv[:2] ** 2).sum() / (Sv ** 2).sum()
print(f"these 2 directions keep {share:.1%} of the spread of the 18 rows")
for w, (px, py) in zip(words, xy.tolist()):
    print(f"PCA {w!r} {px:+.4f} {py:+.4f}")
