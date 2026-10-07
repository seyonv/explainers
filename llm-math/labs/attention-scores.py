"""Attention scores on the tiny model, one piece at a time (card: attention-scores).

Prints one score worked by hand, then what the scores turn into with the scale or the mask taken away.
Run:  python labs/attention-scores.py      (needs numpy; imports tiny.py from this folder)"""
import numpy as np

import tiny

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
v = tiny.forward(tiny.W)
WORDS = [tiny.VOCAB[i] for i in tiny.inputs]       # the, cat, sat

# One score: the query of token 2 ("sat") against the key of token 1 ("cat").
q, k = v["Q"][2], v["K"][1]
print(f"Q row 2 ({WORDS[2]}) = {q}")
print(f"K row 1 ({WORDS[1]}) = {k}")
print(f"multiply pairwise    = {q * k}")
print(f"add them up          = {(q * k).sum():.4f}   (S_raw[2,1])")
print(f"divide by sqrt(4)    = {(q * k).sum() / np.sqrt(tiny.D):.4f}   (S_scaled[2,1])")

# The mask: what each token would attend to if nothing hid the future.
mask = np.triu(np.ones((tiny.T, tiny.T), bool), k=1)
print("\nwith the mask (A, as the model uses it):\n", v["A"])
print("without the mask (row 0 can now see its own target, 'cat'):\n", tiny.softmax(v["S"]))

# The scale: the same masked scores, without dividing by sqrt(d).
print("\nmasked but NOT divided by sqrt(d):\n", tiny.softmax(np.where(mask, -np.inf, v["S_raw"])))
