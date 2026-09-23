"""Copy-on-write KV blocks lab: PagedAttention (Kwon et al., SOSP '23) sections 4.4-4.6.

Run: python3 labs/paged-cow.py   (stdlib + numpy, CPU, well under a second)

A tiny KV block manager with reference counts:
  fork(seq)    a new sequence shares every block of seq (each block's ref count + 1)
  append(seq)  write one token's KV; if the last block is full, allocate a new one;
               if the last block is shared (ref > 1), copy it first (copy-on-write)
  free(seq)    ref count - 1 on each block; blocks that reach 0 go back to the free list

Prints:
  1. the Fig 8 trace (parallel sampling, B = 4, physical blocks 7, 1, 3 as in the paper)
  2. the Fig 9 beam-search step (k = 4), rebuilt from the section 4.4 text
  3. parallel sampling on the running example: Llama-3.1-8B, 1,000-token prompt,
     n = 2, 4, 6 samples of 300 tokens, B = 16: blocks, MB and % saved
  4. beam search, width 2, 4, 6, same lengths, with a SYNTHETIC seeded survival pattern
  5. shared prefix, preemption (swap vs recompute) and tensor-parallel arithmetic

The beam survival pattern is made up (seeded: each step a beam may drop out and be replaced
by a fork of a better-ranked beam);
it is not the output of a real model.
"""
import math
import numpy as np

KV_TOK = 2 * 32 * 8 * 128 * 2        # Llama-3.1-8B: 131,072 B = 128 KiB per token (course facts)


class BlockManager:
    def __init__(self, B, free_order):
        self.B = B
        self.free = list(free_order)   # physical block ids, handed out from the front
        self.ref = {}                  # physical block -> reference count
        self.fill = {}                 # physical block -> tokens written
        self.content = {}              # physical block -> token labels (for traces)
        self.tables = {}               # seq -> list of physical blocks (the block table)
        self.copies = 0                # copy-on-write block copies

    def _alloc(self):
        p = self.free.pop(0)
        self.ref[p], self.fill[p], self.content[p] = 1, 0, []
        return p

    def _release(self, p):
        self.ref[p] -= 1
        if self.ref[p] == 0:
            del self.ref[p], self.fill[p], self.content[p]
            self.free.append(p)

    def new(self, seq):
        self.tables[seq] = []

    def fork(self, parent, child):
        self.tables[child] = list(self.tables[parent])
        for p in self.tables[child]:
            self.ref[p] += 1

    def free_seq(self, seq):
        for p in self.tables.pop(seq):
            self._release(p)

    def append(self, seq, tok="."):
        t = self.tables[seq]
        note = "in place"
        if not t or self.fill[t[-1]] == self.B:
            t.append(self._alloc())
            note = f"new block {t[-1]}"
        elif self.ref[t[-1]] > 1:
            old = t[-1]
            new = self._alloc()
            self.fill[new], self.content[new] = self.fill[old], list(self.content[old])
            self.ref[old] -= 1
            t[-1] = new
            self.copies += 1
            note = f"copy-on-write: block {old} (ref {self.ref[old] + 1}) copied to new block {new}, ref({old}) -> {self.ref[old]}"
        self.content[t[-1]].append(tok)
        self.fill[t[-1]] += 1
        return note

    def used(self):
        return len(self.ref)


def mb(blocks, B):
    return blocks * B * KV_TOK / 1e6


# ---------- 1. Fig 8 ----------
print("== 1. Fig 8: parallel sampling, B = 4 (sampled words illustrative) ==")
bm = BlockManager(4, [7, 1, 3, 0, 2, 4, 5, 6, 8])
bm.new("A1")
for w in "Four score and seven years ago our".split():
    bm.append("A1", w)
bm.fork("A1", "A2")

def show(label):
    print(label)
    for s in ("A1", "A2"):
        rows = [f"{lg}->{p} (ref {bm.ref[p]})" for lg, p in enumerate(bm.tables[s])]
        print(f"   {s}: " + ", ".join(rows))
    for p in sorted(bm.ref):
        print(f"   physical {p}: {bm.content[p]}")

show("(1) prompt stored once; both samples map logical 0,1 to physical 7,1")
print(f"(2) A1 writes 'fathers': {bm.append('A1', 'fathers')}")
print(f"(3) A2 writes 'mothers': {bm.append('A2', 'mothers')}")
show("after both writes")

# ---------- 2. Fig 9 ----------
print("\n== 2. Fig 9: one beam-search step, k = 4, B = 4 (rebuilt from the section 4.4 text) ==")
bm = BlockManager(4, list(range(9, 20)))
before = {0: [0, 1, 3, 2], 1: [0, 1, 3, 6], 2: [0, 1, 3, 7], 3: [0, 4, 5, 8]}
for s, t in before.items():
    bm.tables[s] = list(t)
    for p in t:
        bm.ref[p] = bm.ref.get(p, 0) + 1
        bm.fill[p], bm.content[p] = 4, ["x"] * 4
print("before: " + " | ".join(f"cand {s}: {t}" for s, t in before.items()))
print("        refs " + ", ".join(f"{p}:{r}" for p, r in sorted(bm.ref.items())))
parents = [1, 1, 2, 2]                      # the new top-4 all come from candidates 1 and 2
for i, par in enumerate(parents):
    bm.fork(par, f"new{i}")
freed_before = set(bm.ref)
for s in before:
    bm.free_seq(s)
print(f"freed (ref count reached 0): {sorted(freed_before - set(bm.ref))}")
for i in range(4):
    bm.append(f"new{i}", "y")
print("after:  " + " | ".join(f"cand {i}: {bm.tables[f'new{i}']}" for i in range(4)))
print(f"blocks copied: {bm.copies}  (all old blocks were full, so the new tokens start new blocks)")

# ---------- 3. parallel sampling, running example ----------
P, OUT, B = 1000, 300, 16
blk_mib = B * KV_TOK / 2**20
print(f"\n== 3. Parallel sampling: Llama-3.1-8B, prompt {P}, {OUT} tokens per sample, B = {B} ==")
print(f"one block = {B} x {KV_TOK:,} B = {B*KV_TOK:,} B = {blk_mib:.0f} MiB")
print(f"prompt: {P} / {B} = {P/B} -> {P//B} full blocks + 1 block holding {P % B} tokens")
print(f"one sample: ceil(({P}+{OUT})/{B}) = {math.ceil((P+OUT)/B)} blocks")


def parallel(n):
    bm = BlockManager(B, range(10**6))
    bm.new(0)
    for _ in range(P):
        bm.append(0)
    for i in range(1, n):
        bm.fork(0, i)
    shared_sum = alone_sum = 0
    for step in range(OUT):
        for i in range(n):
            bm.append(i)
        L = P + step + 1
        shared_sum += bm.used()
        alone_sum += n * math.ceil(L / B)
    alone = n * math.ceil((P + OUT) / B)
    return alone, bm.used(), bm.copies, 1 - shared_sum / alone_sum

print(f"{'n':>2} {'no sharing':>18} {'with sharing':>18} {'saved (end)':>12} {'saved (avg)':>12} {'COW copies':>11}")
for n in (2, 4, 6):
    a, s, c, avg = parallel(n)
    print(f"{n:>2} {a:>5} blk {mb(a,B):7.1f} MB {s:>5} blk {mb(s,B):7.1f} MB {1-s/a:11.1%} {avg:11.1%} {c:>11}")
a, s, _, _ = parallel(4)
print(f"n = 4 by hand: no sharing 4 x {math.ceil((P+OUT)/B)} = {4*math.ceil((P+OUT)/B)}; "
      f"sharing {P//B} shared + 4 x {math.ceil((P+OUT)/B) - P//B} own = {P//B + 4*(math.ceil((P+OUT)/B) - P//B)}")
print(f"prompt share of no-sharing KV at the end, n = 4: {P}/(4 x {P+OUT}) = {P/(4*(P+OUT)):.1%}"
      f"  (paper, its section 6.3 run: 12%)")

# ---------- 4. beam search, synthetic survival ----------
print(f"\n== 4. Beam search, prompt {P}, {OUT} steps, B = {B}, SYNTHETIC survival pattern (seed 0) ==")
SWITCH = 0.02    # per step, chance that a beam falls out of the top k (made up)
W = np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0])     # which ranked beam replaces it (made up)


def beam(k, seed=0):
    rng = np.random.default_rng(seed)
    w = W[:k] / W[:k].sum() if k > 1 else np.array([1.0])
    bm = BlockManager(B, range(10**6))
    bm.new(("b", 0, 0))
    for _ in range(P):
        bm.append(("b", 0, 0))
    beams = [("b", 0, 0)]
    for i in range(1, k):
        bm.fork(beams[0], ("b", 0, i))
        beams.append(("b", 0, i))
    shared_sum = alone_sum = old_copy_tok = 0
    for step in range(1, OUT + 1):
        parents = np.arange(k)
        for i in range(k):
            if rng.random() < SWITCH:              # beam i drops out; a fork of a better beam takes its place
                parents[i] = rng.choice(k, p=w)
        parents = np.sort(parents)
        new = []
        for i, par in enumerate(parents):
            bm.fork(beams[par], ("b", step, i))
            new.append(("b", step, i))
        # a contiguous system keeps each surviving parent's buffer and must copy the
        # whole cache for every extra child of the same parent
        old_copy_tok += (k - len(set(parents.tolist()))) * (P + step - 1)
        for b in beams:
            bm.free_seq(b)
        beams = new
        for b in beams:
            bm.append(b)
        L = P + step
        shared_sum += bm.used()
        alone_sum += k * math.ceil(L / B)
    # where do the final beams split?  shared prefix = blocks common to all tables
    common = 0
    for col in zip(*[bm.tables[b] for b in beams]):
        if len(set(col)) == 1:
            common += 1
        else:
            break
    alone = k * math.ceil((P + OUT) / B)
    return alone, bm.used(), bm.copies, 1 - shared_sum / alone_sum, old_copy_tok, common

print(f"each step each beam drops out with p = {SWITCH}; its replacement forks ranked beam j with p =", W[:4].tolist())
print(f"{'k':>2} {'no sharing':>18} {'with sharing':>18} {'saved (end)':>12} {'saved (avg)':>12} "
      f"{'COW blocks':>11} {'contig copies':>15} {'common blocks':>14}")
for k in (2, 4, 6):
    a, s, c, avg, oc, common = beam(k)
    print(f"{k:>2} {a:>5} blk {mb(a,B):7.1f} MB {s:>5} blk {mb(s,B):7.1f} MB {1-s/a:11.1%} {avg:11.1%} "
          f"{c:>6} ({c*B*KV_TOK/1e9:.2f} GB) {oc:>7,} tok ({oc*KV_TOK/1e9:.0f} GB) {common:>6}")
print("COW blocks = blocks vLLM copied; contig copies = tokens a copy-the-whole-cache system moves.")

# ---------- 5. shared prefix, preemption, tensor parallel ----------
print("\n== 5a. Shared prefix: a 341-token few-shot prefix (the paper's 5-shot length, section 6.4) ==")
PRE = 341
print(f"{PRE} / {B} -> {PRE//B} full shared blocks + {PRE % B} tokens in a copy-on-write last block")
for R in (100,):
    saved = R * (PRE // B) - PRE // B
    print(f"{R} requests: {R} x {PRE//B} = {R*(PRE//B)} prefix blocks without sharing, {PRE//B} with -> "
          f"{saved:,} blocks = {mb(saved,B)/1e3:.2f} GB saved")
print(f"prefill per request runs on the user input only; the {PRE % B} prefix tokens in the last block are copied, not recomputed")

print("\n== 5b. Preemption of one 1,300-token sequence (ideal floors, our estimate) ==")
L = P + OUT
by = L * KV_TOK
pcie_one_way = 64e9     # NVIDIA lists 'PCIe Gen5: 128GB/s' for H100 SXM; we take half per direction
print(f"KV to move: {L} x {KV_TOK:,} B = {by/1e6:.1f} MB")
print(f"swap out at 64 GB/s = {by/pcie_one_way*1e3:.2f} ms; out + in = {2*by/pcie_one_way*1e3:.2f} ms")
print(f"recompute: one prefill of {L} tokens at 100% MFU = 2 x 8.03e9 x {L} / 989e12 = "
      f"{2*8.03e9*L/989e12*1e3:.1f} ms")
print(f"swap space bound: <= GPU KV memory = 63.94 GB of CPU RAM")

print("\n== 5c. Tensor parallel (section 4.6): same block ids on every GPU, each keeps its KV heads ==")
for tp in (1, 2, 4, 8):
    print(f"TP = {tp}: {8//tp} of 8 KV heads per GPU -> {B*KV_TOK/tp/2**10:,.0f} KiB per 16-token block per GPU")

# Try this:
#   OUT = 30               -> short answers (Alpaca-like): the shared prompt dominates, savings jump
#   P = 20                 -> short prompts (Alpaca mean 19.31): parallel-sampling savings fall toward the paper's 6-10%
#   SWITCH = 0.0           -> beams never swap after the prompt: beam sharing falls to parallel-sampling levels
#   SWITCH = 0.3           -> constant reshuffling: lineages merge fast, more sharing but many more block copies
