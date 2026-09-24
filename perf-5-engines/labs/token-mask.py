"""token-mask.py - a toy grammar engine: naive per-step masks vs an XGrammar-style cache.

Course 5 card: "Structured outputs inside the engine".
Pure simulator: stdlib only, CPU, runs in well under a second. Everything here is a TOY:
a 5,000-token synthetic vocabulary (seeded) and a tiny recursive grammar. The counts it
prints are ours, not XGrammar's (the paper's Llama-3.1 / JSON numbers are quoted on the card).

The grammar (a context-free grammar, because objects nest; the output is one object):
    obj   := '{"a":' ' '? (int | obj) '}'
    int   := '-'? ('0' | [1-9][0-9]*)          (inlined into obj, as XGrammar's rule inlining would)
e.g.  {"a": {"a": -42}}

It runs as a byte-level pushdown automaton (PDA): a current node plus a stack of return
nodes. Entering a nested obj pushes the node to return to (OC: "expect '}'"); closing an
obj pops it.

Three ways to build the mask (which of the V tokens may come next):
  naive     : run every token through the PDA with the full stack, every step.
  cache     : XGrammar's idea. Offline, for every node that can be the stack top, run every
              token with an EMPTY stack below it. If the token is decided without ever popping
              below the top, its answer depends only on the node: context-INDEPENDENT, store it.
              If it has to pop into the parent rule, it's context-DEPENDENT: check it at runtime.
  cache+exp : context expansion (our 1-character version of XGrammar's expanded suffix):
              any rule here can only return into '}' (inside an object) or the end of the
              output, so a dependent token whose first char after the pop is not '}' is
              rejected offline too.
"""
import random
import string
import time

# ---- the PDA -------------------------------------------------------------------------
LIT = '{"a":'
OBJ = [f"O{i}" for i in range(len(LIT) + 1)]   # O0 --{--> O1 ... --:--> O5
# O5: optional space, then a value.  O6: after the space, a value.
# In/Iz/Id: inside the (inlined) int: after '-', after '0', after [1-9][0-9]*.
# OC: a nested obj has closed, expect this obj's '}'.  OE: this obj is complete.
ROOT_END = "END"                                  # top-level obj finished: only EOS


def value(c, stack):
    """At the value slot: '{' starts a nested obj (push OC), '-'/digit starts the int."""
    if c == "{":
        return OBJ[1], stack + ("OC",)
    if c == "-":
        return "In", stack
    if c == "0":
        return "Iz", stack
    if c in "123456789":
        return "Id", stack
    return None


def step(node, stack, c):
    """Consume one char. Returns (node, stack), None (reject), or 'POP_BELOW' when this
    obj is complete, the char must be matched by the parent, and the stack we were given
    is empty (that only happens in the offline, local run: the token is context-dependent)."""
    global CHAR_STEPS
    while True:
        CHAR_STEPS += 1
        if node in OBJ[:len(LIT)]:
            i = OBJ.index(node)
            return (OBJ[i + 1], stack) if c == LIT[i] else None
        if node == OBJ[len(LIT)]:
            return ("O6", stack) if c == " " else value(c, stack)
        if node == "O6":
            return value(c, stack)
        if node == "In":
            if c == "0":
                return "Iz", stack
            return ("Id", stack) if c in "123456789" else None
        if node == "Id" and c in string.digits:
            return "Id", stack
        if node in ("Iz", "Id", "OC"):
            return ("OE", stack) if c == "}" else None
        if node == "OE":                           # obj complete: return to the parent
            if not stack:
                return "POP_BELOW"
            node, stack = stack[-1], stack[:-1]
            continue                               # retry c in the parent
        return None                                # ROOT_END: only EOS may follow


def run_token(node, stack, tok):
    """Feed a whole token. Returns 'acc', 'rej' or 'dep' (needed the parent context)."""
    for c in tok:
        r = step(node, stack, c)
        if r is None:
            return "rej", None
        if r == "POP_BELOW":
            return "dep", None
        node, stack = r
        if node == "OE" and stack:                 # obj complete inside a parent: pop now
            node, stack = stack[-1], stack[:-1]
    return "acc", (node, stack)


# ---- a toy vocabulary ----------------------------------------------------------------
V = 5000
random.seed(0)
JSONY = '{}":, -'
def rand_tok():
    n = random.choice([2, 2, 3, 3, 4, 5, 6])
    out = []
    for _ in range(n):
        r = random.random()
        if r < 0.40:
            out.append(random.choice(string.ascii_lowercase))
        elif r < 0.62:
            out.append(random.choice(string.digits))
        elif r < 0.92:
            out.append(random.choice(JSONY))
        else:
            out.append(random.choice("[]()._;!?'"))
    return "".join(out)

vocab = [chr(i) for i in range(32, 127)] + ['{"a":', '{"a": ', '"a":', '{"', '"a"', '":', ' {', "}}", "}}}"]
seen = set(vocab)
while len(vocab) < V:
    t = rand_tok()
    if t not in seen:
        seen.add(t)
        vocab.append(t)

# ---- offline: the adaptive token-mask cache --------------------------------------------
TOP_NODES = OBJ + ["O6", "In", "Iz", "Id", "OC"]   # every node that can be on top at a step
CHAR_STEPS = 0
t0 = time.perf_counter()
cache = {}
for n in TOP_NODES:
    acc, rej, dep = [], [], []
    for i, tok in enumerate(vocab):
        kind, _ = run_token(n, (), tok)
        {"acc": acc, "rej": rej, "dep": dep}[kind].append(i)
    cache[n] = (acc, rej, dep)
build_steps, build_s = CHAR_STEPS, time.perf_counter() - t0

# context expansion: an obj can only return into 'OC' (expects '}') or the root (EOS only).
# For a dependent token, find the first char it still has after its rule ends.
def leftover_first_char(node, tok):
    stack = ()
    for k, c in enumerate(tok):
        r = step(node, stack, c)
        if r == "POP_BELOW":
            return c
        node, stack = r
    return None

expanded = {}
for n, (acc, rej, dep) in cache.items():
    keep = [i for i in dep if leftover_first_char(n, vocab[i]) == "}"]
    expanded[n] = keep

# ---- runtime: generate one output, building every mask three ways -----------------------
TARGET = '{"a": {"a": {"a": -42}}}'

def naive_mask(node, stack):
    return {i for i, tok in enumerate(vocab) if run_token(node, stack, tok)[0] == "acc"}

def cached_mask(top, node, stack, dep_lists):
    acc, _, _ = cache[top]
    allowed = set(acc)
    for i in dep_lists[top]:                      # only these need the full stack
        if run_token(node, stack, vocab[i])[0] == "acc":
            allowed.add(i)
    return allowed

by_len = sorted(range(V), key=lambda i: -len(vocab[i]))
node, stack = "O0", (ROOT_END,)   # root: parse one obj, then END
pos, rows = 0, []
tot = {"naive": 0, "cache": 0, "exp": 0}
steps_char = {"naive": 0, "cache": 0, "exp": 0}
while pos < len(TARGET):
    key = node
    for name, fn in (("naive", lambda: naive_mask(node, stack)),
                     ("cache", lambda: cached_mask(key, node, stack, {k: v[2] for k, v in cache.items()})),
                     ("exp", lambda: cached_mask(key, node, stack, expanded))):
        CHAR_STEPS = 0
        m = fn()
        steps_char[name] += CHAR_STEPS
        if name == "naive":
            ref = m
            tot[name] += V
        else:
            assert m == ref, f"mask mismatch at {pos}"
            tot[name] += len(cache[key][2]) if name == "cache" else len(expanded[key])
    # greedy tokenizer: the longest allowed token that matches the target here
    tid = next(i for i in by_len if TARGET.startswith(vocab[i], pos) and i in ref)
    rows.append((vocab[tid], len(ref), len(cache[key][2]), len(expanded[key]), len(stack)))
    _, (node, stack) = run_token(node, stack, vocab[tid])
    pos += len(vocab[tid])

# ---- print -------------------------------------------------------------------------
print(f'TOY: {V:,}-token synthetic vocabulary (seed 0), grammar obj := {{"a": (int | obj)}}')
print(f"Output: {TARGET}   ({len(rows)} decode steps)\n")
print(f"{'step':>4}  {'token chosen':<14}{'allowed':>8}{'naive checks':>14}{'cache checks':>14}{'+expansion':>12}{'stack':>7}")
for k, (tok, allowed, dep, exp, depth) in enumerate(rows, 1):
    print(f"{k:>4}  {tok!r:<14}{allowed:>8,}{V:>14,}{dep:>14,}{exp:>12,}{depth:>7}")
print()
n = len(rows)
print(f"Token checks for the whole output: naive {tot['naive']:,}   cache {tot['cache']:,}"
      f"   cache+expansion {tot['exp']:,}")
print(f"  per step: naive {V:,}   cache {tot['cache']/n:.1f}   cache+expansion {tot['exp']/n:.1f}")
print(f"  reduction: {tot['naive']/tot['cache']:.0f}x (cache), {tot['naive']/max(tot['exp'],1):.0f}x (with expansion)")
print(f"PDA char steps at runtime: naive {steps_char['naive']:,}   cache {steps_char['cache']:,}"
      f"   cache+expansion {steps_char['exp']:,}")
print("Every cached mask was asserted equal to the naive mask.\n")

print(f"Context-dependent tokens per stack-top node (of {V:,}):")
worst = max(len(v[2]) for v in cache.values())
for nd in TOP_NODES:
    a, r, d = cache[nd]
    print(f"  {nd:<4} accepted {len(a):>5,}  rejected {len(r):>5,}  dependent {len(d):>4}  after expansion {len(expanded[nd]):>3}")
all_dep = sum(len(v[2]) for v in cache.values())
all_exp = sum(len(v) for v in expanded.values())
print(f"Largest dependent set: {worst} of {V:,} = {100*worst/V:.2f}% of the vocabulary")
print(f"All (node, token) pairs: {len(TOP_NODES)} x {V:,} = {len(TOP_NODES)*V:,}; dependent "
      f"{100*sum(len(v[2]) for v in cache.values())/(len(TOP_NODES)*V):.2f}%")
print(f"Dependent (node, token) pairs: {all_dep} -> {all_exp} after expansion "
      f"({100*(1-all_exp/all_dep):.0f}% fewer)")

print(f"\nOffline cache build: {build_steps:,} PDA char steps, {build_s*1000:.0f} ms in Python on this machine (varies)")

# ---- the running example: CPU mask time vs the GPU step (our recomputation) --------------
print("\nRunning example, Llama-3.1-8B BF16 on one H100 (our recomputation):")
STEP_B1 = 4.8   # ms, 16.06 GB / 3.35 TB/s
for label, ms in (("1 ms mask (illustrative)", 1.0), ("XGrammar JSON, <40 us", 0.040),
                  ("Table 3 PDA baseline 65.776 ms", 65.776)):
    print(f"  {label:<32} serial: {STEP_B1:.1f} + {ms:g} ms = +{100*ms/STEP_B1:.1f}%   "
          f"overlapped: max({STEP_B1:.1f}, {ms:g}) = {max(STEP_B1, ms):g} ms")
B, CTX = 64, 2048
step_b = B * CTX * 131072 / 3.35e12 * 1e3 + max(2 * B * 8.03e9 / 989e12, 16.06e9 / 3.35e12) * 1e3
for ms in (0.040, 1.0):
    print(f"  batch {B}, 2k context: GPU step {step_b:.2f} ms; {B} masks x {ms*1000:g} us one after another "
          f"= {B*ms:.2f} ms ({'hidden' if B*ms < step_b else 'NOT hidden'} under the step)")

# Try this:
#  1. TARGET = '{"a": 7}' : no nesting, so the stack never matters and even dependent tokens are easy.
#  2. JSONY = '}}}}}' : a vocabulary full of closing braces. The dependent sets grow about 6x and
#     context expansion removes less, because more leftovers really do start with '}'.
#  3. V = 50000 : naive checks grow 10x per step; the cache's runtime checks grow only with the dependents.
