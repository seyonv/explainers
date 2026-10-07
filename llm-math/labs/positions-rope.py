"""Measurements for the card "Position is added, or rotated in" (llm-math/positions-rope).

1. Without positions, attention can't see word order: shuffle the words and each word's
   output just moves with it.
2. Learned positions fix that: x0 = E[id] + P[pos] gives the same word a different row at each spot.
3. RoPE instead turns each pair of query/key numbers by (position x angle), so q.k depends
   only on how far apart the two words are.
4. Qwen2.5's 32 pairs turn at very different speeds.

Run:  python labs/positions-rope.py     (from llm-math/; needs numpy and tiny.py next to it)"""
import numpy as np

import tiny

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
W = tiny.W


def run(ids, use_positions=True, causal=True):
    """The tiny model's forward pass on any 3 token ids. Returns attention output C and the probabilities."""
    Wt = dict(W)
    if not use_positions:
        Wt["P"] = np.zeros_like(W["P"])       # no position table: x0 = E[id] only
    x0 = Wt["E"][ids] + Wt["P"]
    a, _ = tiny.layernorm(x0, Wt["g1"], Wt["b1"])
    Q, K, V = a @ Wt["Wq"], a @ Wt["Wk"], a @ Wt["Wv"]
    S = Q @ K.T / np.sqrt(tiny.D)
    if causal:
        S = np.where(np.triu(np.ones((3, 3), bool), 1), -np.inf, S)
    C = tiny.softmax(S) @ V
    tiny.inputs = np.array(ids)                # the full model reads its ids from tiny.inputs
    p = tiny.forward(Wt)["p"]
    tiny.inputs = np.array([0, 1, 2])
    return C, p


print("== 1. attention alone is blind to order (no positions, no mask)")
C1, _ = run([0, 1, 2], use_positions=False, causal=False)   # the cat sat
C2, _ = run([2, 1, 0], use_positions=False, causal=False)   # sat cat the
print("output for 'the' in 'the cat sat':", C1[0])
print("output for 'the' in 'sat cat the':", C2[2])
print(f"largest difference, matching each word to itself: {np.abs(C1 - C2[::-1]).max():.1e}")

print("\n== 2. the full tiny model (with its causal mask), 'the cat sat' vs 'cat the sat'")
for use in (False, True):
    _, pa = run([0, 1, 2], use_positions=use)
    _, pb = run([1, 0, 2], use_positions=use)
    tag = "with P   " if use else "without P"
    print(f"{tag}: p after 'sat' = {pa[2]}  vs  {pb[2]}   largest difference {np.abs(pa[2] - pb[2]).max():.1e}")

print("\n== 3. learned positions: the same word at each position")
for pos in range(3):
    print(f"'the' at position {pos}: E[0] + P[{pos}] = {W['E'][0] + W['P'][pos]}")

print("\n== 4. RoPE on one pair, angle 0.5 rad per position, q = [1, 0], k = [0.6, 0.8]")
theta = 0.5
def rot(vec, pos):
    c, s = np.cos(pos * theta), np.sin(pos * theta)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])
q, k = np.array([1.0, 0.0]), np.array([0.6, 0.8])
print("q turned to position 1:", rot(q, 1), "  k turned to position 0:", rot(k, 0))
print("q turned to position 3:", rot(q, 3), "  k turned to position 2:", rot(k, 2))
for dist in range(0, 13):
    print(f"distance {dist:>2}: rotated q.k = {rot(q, dist) @ rot(k, 0):+.4f}"
          f"   = 0.6 cos({dist}*0.5) + 0.8 sin({dist}*0.5)")
phi = np.arctan2(k[1], k[0])
print(f"k points at {phi:.4f} rad, q at 0 rad. Turned to positions m and n, the gap is {phi:.4f} + 0.5 (n - m):")
for m, n in [(0, 0), (1, 0), (3, 2), (7, 6), (2, 0)]:
    gap = phi + theta * (n - m)
    print(f"  q at {m}, k at {n}: gap {gap:+.4f} rad, cos(gap) = {np.cos(gap):.4f}")
print(f"one pair repeats every 2 pi / 0.5 = {2 * np.pi / theta:.2f} positions")
print(f"lengths before and after turning: |k| = {np.linalg.norm(k):.4f}, |R k| = {np.linalg.norm(rot(k, 7)):.4f}")

print("\n== 5. Qwen2.5: 32 pairs, pair i turns by 1e6^(-2i/64) rad per position")
for i in (0, 1, 8, 16, 24, 31):
    ang = 1e6 ** (-2 * i / 64)
    print(f"pair {i:>2}: {ang:.3g} rad per position, full turn every {2 * np.pi / ang:,.1f} positions,"
          f" turn between words 1000 apart {1000 * ang:,.3g} rad")
