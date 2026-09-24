"""Speculative decoding I: exact sampling with a draft (Leviathan et al. 2023, arXiv 2211.17192, sections 2-3).

Run: python3 labs/spec-sim.py   (numpy only, CPU, about 2 s)

Notation follows Leviathan: p = TARGET model M_p, q = DRAFT model M_q.
(Chen et al. 2023, arXiv 2302.01318, swap the letters: q = target, p = draft.)

Parts
  (0) The card's illustrative 4-token example: one speculative-sampling decision, the residual
      norm(max(0, p - q)), and the App A.1 check that P(x) = min(p, q) + (1 - beta) p'(x) = p(x).
  (1) Exactness, one token (section 2.3): 200k samples from random p, q over a 10-token vocab.
      Speculative sampling's histogram vs p (total variation) compared with just trusting q.
  (1b) Exactness, full Algorithm 1 (gamma = 4) on a toy bigram "language" where p and q depend
      on the previous token, so beta changes from step to step (not i.i.d.).
  (2) alpha: measured acceptance vs sum_x min(p, q) (Theorem 3.5 / Corollary 3.6).
  (2b) Standardized sampling (section 2.2): argmax, top-k, nucleus and temperature are all
      "sample from an adjusted distribution".
  (3) Eq 1 and Theorem 3.8 tables over alpha, gamma, c; optimal gamma per (alpha, c); Table 1
      recomputed (Theorem 3.11 ops with c = c_hat = 0); Corollary 3.9; the card's worked examples.
"""
import numpy as np

rng = np.random.default_rng(0)


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def residual(p, q):
    r = np.maximum(p - q, 0.0)
    return r / r.sum()


def spec_one(p, q, u_draft, u_acc, u_res):
    """Vectorised section 2.3: x ~ q, keep w.p. min(1, p/q), else x ~ norm(max(0, p - q))."""
    x = np.searchsorted(np.cumsum(q), u_draft * q.sum())
    keep = u_acc <= np.minimum(1.0, p[x] / q[x])
    y = np.searchsorted(np.cumsum(residual(p, q)), u_res)
    return np.where(keep, x, y), keep


# ------------------------------------------------------------------ (0) the card's 4-token example
print("== (0) One decision, the card's illustrative 4-token vocabulary ==")
words = ["mat", "rug", "sofa", "floor"]
p4 = np.array([0.20, 0.45, 0.25, 0.10])   # target
q4 = np.array([0.50, 0.25, 0.15, 0.10])   # draft
beta = np.minimum(p4, q4).sum()
res = np.maximum(p4 - q4, 0)
print(f"{'token':6} {'p':>5} {'q':>5} {'min':>5} {'max(0,p-q)':>11} {'p_prime':>8} {'min+(1-b)p_prime':>17}")
for w, a, b, m, r in zip(words, p4, q4, np.minimum(p4, q4), res):
    pp = r / res.sum()
    print(f"{w:6} {a:5.2f} {b:5.2f} {m:5.2f} {r:11.2f} {pp:8.3f} {m + (1 - beta) * pp:17.3f}")
print(f"beta = sum min(p, q) = {beta:.2f};  D_LK = 1 - beta = {1 - beta:.2f} = TV(p, q) = {tv(p4, q4):.2f}"
      f";  residual mass = {res.sum():.2f} (= 1 - beta)")
print(f"draft proposed 'mat': accept prob = min(1, {p4[0]:.2f}/{q4[0]:.2f}) = {min(1, p4[0] / q4[0]):.2f};"
      f" with r = 0.71 > 0.40 -> reject, resample from p' = [0, {res[1]/res.sum():.3f}, {res[2]/res.sum():.3f}, 0]")

# ------------------------------------------------------------------ (1) exactness, one token
print("\n== (1) Exactness of speculative sampling, one token (section 2.3) ==")
V, N = 10, 200_000
p = rng.dirichlet(np.ones(V))
q = rng.dirichlet(np.ones(V))
out, keep = spec_one(p, q, rng.random(N), rng.random(N), rng.random(N))
h_spec = np.bincount(out, minlength=V) / N
h_q = np.bincount(np.searchsorted(np.cumsum(q), rng.random(N) * q.sum()), minlength=V) / N
h_p = np.bincount(np.searchsorted(np.cumsum(p), rng.random(N) * p.sum()), minlength=V) / N
print(f"vocab {V}, random p and q (Dirichlet(1)), {N:,} samples each")
print(f"TV(p, q) = {tv(p, q):.3f}  (so alpha = 1 - TV = {1 - tv(p, q):.3f})")
print(f"TV(speculative histogram, p) = {tv(h_spec, p):.4f}   <- exact up to noise")
print(f"TV(direct samples from p, p) = {tv(h_p, p):.4f}   <- the sampling-noise floor for 200k samples")
print(f"TV(trust the draft q,     p) = {tv(h_q, p):.4f}   <- skipping verification changes the output")

# ------------------------------------------------------------------ (1b) Algorithm 1 on a bigram toy
print("\n== (1b) Algorithm 1 (gamma = 4) on a toy bigram model: context-dependent p and q ==")
V2, G, TOK = 8, 4, 200_000
P = rng.dirichlet(np.full(V2, 0.5), size=V2)          # target: P[prev] = p(. | prev)
Q = 0.7 * P + 0.3 * rng.dirichlet(np.full(V2, 0.5), size=V2)   # draft: a noisy copy of the target
cP, cQ = np.cumsum(P, 1), np.cumsum(Q, 1)
R = np.maximum(P - Q, 0); R /= R.sum(1, keepdims=True); cR = np.cumsum(R, 1)
beta_ctx = np.minimum(P, Q).sum(1)
seq = [0]
iters = drafted = accepted = 0
u = rng.random(TOK * 3 + 100); ui = 0
def draw(c):
    global ui
    v = np.searchsorted(c, u[ui] * c[-1]); ui += 1
    if ui >= len(u) - 10:
        u[:] = rng.random(len(u)); ui = 0
    return min(int(v), V2 - 1)
while len(seq) < TOK + 1:
    iters += 1
    ctx, xs = seq[-1], []
    for _ in range(G):                       # draft gamma tokens autoregressively from q
        xs.append(draw(cQ[ctx])); ctx = xs[-1]
    prev, n = seq[-1], 0                     # "run M_p in parallel": here p is just a lookup
    for x in xs:                             # accept left to right while r <= p/q
        drafted += 1
        if u[ui] <= P[prev, x] / Q[prev, x]:
            ui += 1; n += 1; accepted += 1; seq.append(x); prev = x
        else:
            ui += 1; break
    seq.append(draw(cR[prev]) if n < G else draw(cP[prev]))   # fix token, or bonus token from p_{gamma+1}
seq = np.array(seq[: TOK + 1])
emp = np.zeros((V2, V2)); np.add.at(emp, (seq[:-1], seq[1:]), 1)
rows = emp.sum(1)
emp_p = emp / rows[:, None]
w = rows / rows.sum()
tv_rows = np.array([tv(emp_p[i], P[i]) for i in range(V2)])
tv_rows_q = np.array([tv(Q[i], P[i]) for i in range(V2)])
print(f"{TOK:,} tokens in {iters:,} target runs; beta per context ranges {beta_ctx.min():.2f}-{beta_ctx.max():.2f}")
print(f"transition TV(empirical, p), weighted over contexts = {(w * tv_rows).sum():.4f}   (exact up to noise)")
print(f"transition TV(q, p) if we trusted the draft           = {(w * tv_rows_q).sum():.4f}")
alpha_bi = (w * beta_ctx).sum()
eq1 = (1 - alpha_bi ** (G + 1)) / (1 - alpha_bi)
print(f"alpha (beta averaged over visited contexts) = {alpha_bi:.3f};  measured accepted/drafted = {accepted / drafted:.3f}")
print(f"tokens per target run: measured {TOK / iters:.3f} vs Eq 1 with that alpha {eq1:.3f}"
      f"  (the i.i.d. assumption is only approximate)")

# ------------------------------------------------------------------ (2) alpha = sum min(p, q)
print("\n== (2) Measured acceptance vs sum_x min(p, q) (Theorem 3.5 / Corollary 3.6) ==")
print(f"one-token run in (1): accepted fraction = {keep.mean():.4f};  sum min(p, q) = {np.minimum(p, q).sum():.4f}")
for mix in (0.0, 0.5, 0.9):
    qq = mix * p + (1 - mix) * q
    _, k = spec_one(p, qq, rng.random(N), rng.random(N), rng.random(N))
    print(f"draft = {mix:.1f}*p + {1-mix:.1f}*q : measured {k.mean():.4f}  vs  sum min(p, q) {np.minimum(p, qq).sum():.4f}")
print("'accept if equal' at T = 1, on the 4-token example (p = target, q = draft):")
print(f"  compare with argmax p : output is always argmax p -> TV(output, p) = 1 - max p = {1 - p4.max():.2f} (not exact)")
print(f"  compare with a sample y ~ p: exact (output = y), but acceptance = sum p*q = {(p4 * q4).sum():.4f}"
      f"  vs speculative sampling sum min(p, q) = {np.minimum(p4, q4).sum():.2f}")

# ------------------------------------------------------------------ (2b) standardized sampling
print("\n== (2b) Standardized sampling (section 2.2): every method = sample from an adjusted p ==")
base = p4[[1, 2, 0, 3]]                      # rug .45, sofa .25, mat .20, floor .10 (sorted)
names = ["rug", "sofa", "mat", "floor"]
def topk(d, k):
    o = np.zeros_like(d); idx = np.argsort(-d)[:k]; o[idx] = d[idx]; return o / o.sum()
def nucleus(d, top_p):
    idx = np.argsort(-d); cs = np.cumsum(d[idx]); keep_n = np.searchsorted(cs, top_p - 1e-12) + 1
    o = np.zeros_like(d); o[idx[:keep_n]] = d[idx[:keep_n]]; return o / o.sum()
def temp(d, t):
    o = d ** (1 / t); return o / o.sum()
rows_std = [("plain (T = 1)", base), ("argmax (T = 0)", topk(base, 1)), ("top-k, k = 2", topk(base, 2)),
            ("nucleus, top-p = 0.9", nucleus(base, 0.9)), ("temperature T = 0.5", temp(base, 0.5))]
print(f"{'method':22}" + "".join(f"{n:>7}" for n in names))
for lab, d in rows_std:
    print(f"{lab:22}" + "".join(f"{v:7.3f}" for v in d))
pa, qa = topk(p4, 1), topk(q4, 1)
print(f"greedy case: argmax p = {words[pa.argmax()]}, argmax q = {words[qa.argmax()]};"
      f" min(1, p/q) for the draft's token = {min(1, pa[qa.argmax()] / qa[qa.argmax()]):.0f}"
      f" -> speculative sampling reduces to 'accept iff equal'")

# ------------------------------------------------------------------ (3) Eq 1, Thm 3.8, Thm 3.11
def eq1(a, g):
    return (1 - a ** (g + 1)) / (1 - a)
def speed(a, g, c):
    return eq1(a, g) / (g * c + 1)
def ops(a, g, ch):
    return (1 - a) * (g * ch + g + 1) / (1 - a ** (g + 1))

alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
gammas = range(1, 11)
print("\n== (3a) Eq 1: expected tokens per target run, E = (1 - a^(g+1)) / (1 - a) ==")
print("alpha " + "".join(f"  g={g:<3}" for g in gammas))
for a in alphas:
    print(f"{a:5.1f} " + "".join(f"{eq1(a, g):7.2f}" for g in gammas))
for c in (0.0, 0.05, 0.66):
    print(f"\n== (3b) Theorem 3.8 speedup (1 - a^(g+1)) / ((1 - a)(g c + 1)), c = {c} ==")
    print("alpha " + "".join(f"  g={g:<3}" for g in gammas) + "   best g  best x")
    for a in alphas:
        s = [speed(a, g, c) for g in gammas]
        b = int(np.argmax(s))
        print(f"{a:5.1f} " + "".join(f"{v:7.2f}" for v in s) + f"   {b + 1:>6}  {s[b]:6.2f}")

print("\n== (3c) Optimal gamma per (alpha, c), searched over gamma = 1..50 (section 3.5, Fig 3) ==")
cs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.66]
print("alpha  c=0 (bound)" + "".join(f"  c={c:<6}" for c in cs))
for a in alphas + [0.867]:
    cells = []
    for c in cs:
        s = [speed(a, g, c) for g in range(1, 51)]
        b = int(np.argmax(s))
        cells.append(f"{b + 1:>3} {s[b]:5.2f}x")
    print(f"{a:5.3f}  inf {1 / (1 - a):5.2f}x  " + "  ".join(cells))
print("(each cell: best gamma, speedup. c = 0 has no finite optimum: speedup rises toward 1/(1 - alpha))")
for a in alphas:
    print(f"  alpha {a}: upper bound 1/(1 - alpha) = {1 / (1 - a):.2f}  (also the oracle-gamma E, section 3.5)")

print("\n== (3d) Table 1 recomputed (c = c_hat = 0): ops = (1-a)(g+1)/(1-a^(g+1)), speed = Eq 1 ==")
print("alpha gamma   ops    speed   (paper)")
paper = {(0.6, 2): (1.53, 1.96), (0.7, 3): (1.58, 2.53), (0.8, 2): (1.23, 2.44),
         (0.8, 5): (1.63, 3.69), (0.9, 2): (1.11, 2.71), (0.9, 10): (1.60, 6.86)}
for (a, g), (po, ps) in paper.items():
    print(f"{a:5.1f} {g:5d} {ops(a, g, 0):6.2f}x {eq1(a, g):6.2f}x   ({po}x, {ps}x)")

print("\n== (3e) Corollary 3.9 and section 3.6 ==")
for a, c in ((0.8, 0.05), (0.867, 0.66), (0.2, 0.0)):
    print(f"alpha {a}, c {c}: gamma = 1 gives (1 + a)/(1 + c) = {(1 + a) / (1 + c):.3f}x")
print(f"bigram draft, alpha 0.2, c = 0, gamma = 3: {speed(0.2, 3, 0):.3f}x  (paper: 1.25x)")

print("\n== (3f) The card's worked examples ==")
a, g, c = 0.8, 4, 0.05
E = eq1(a, g)
print(f"alpha {a}, gamma {g}, c {c}: E = (1 - {a}^{g+1}) / (1 - {a}) = {1 - a ** (g + 1):.5f} / {1 - a:.1f} = {E:.3f} tokens")
print(f"  cost per run = gamma*c + 1 = {g * c + 1:.2f} target steps -> speedup = {E:.3f} / {g * c + 1:.2f} = {speed(a, g, c):.3f}x")
print(f"  ops factor (Thm 3.11, c_hat = c = {c}) = (1 - a)(g*c_hat + g + 1)/(1 - a^(g+1)) = {ops(a, g, c):.3f}x")
print(f"  target weight/KV reads per token fall by E = {E:.2f}x (1/{E:.2f} = {1 / E:.3f} of plain decoding)")
print(f"  P(all {g} drafts accepted -> bonus token) = a^g = {a ** g:.3f}")
dist = [(1 - a) * a ** k for k in range(g)] + [a ** g]
print("  distribution of tokens per run (1..5): " + ", ".join(f"{k + 1}: {v:.3f}" for k, v in enumerate(dist)))
a, c = 0.867, 0.66
print(f"our Mac pair (3B Q4 target, 1B Q8 draft; code prompt n = 4 acceptance 0.867, c = 0.66):")
for g in (1, 2, 3, 4, 8):
    print(f"  gamma {g}: E = {eq1(a, g):.3f}, cost = {g * c + 1:.2f}, speedup = {speed(a, g, c):.3f}x")

# Try this:
# 1. Make the draft worse: Q = 0.3 * P + 0.7 * noise in (1b). Exactness still holds; tokens per run fall.
# 2. Set c = 0.125 (perf-2's hypothetical 1B draft for Llama-3.1-8B on an H100) in (3b) and find the best gamma.
# 3. Lenience (App A.5): compare p with l*q (l = 0.5) inside spec_one. Acceptance rises to sum min(p/l, q),
#    but TV(histogram, p) no longer goes to 0: the output is no longer exactly p.
