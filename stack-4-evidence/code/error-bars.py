"""Error bars for an agent eval: naive vs clustered vs paired.

Two agents, A (old) and B (new), each run 3 times on the same 30
tasks. A trial scores 1 (the checker's re-run of the tests passes)
or 0. The task population is illustrative: each task has its own
pass chance, drawn once, so trials of one task are alike.

Three standard errors (Miller 2024, "Adding Error Bars to Evals"):
  naive    s / sqrt(90): pretends all 90 trials are independent
  cluster  Miller's clustered SE with one cluster per task
  paired   SE of the per-task difference B - A (same tasks)
Then 20,000 fresh evals check which error bar is honest.
Python 3.10 + numpy.
"""
import math

import numpy as np

TASKS, K, Z = 30, 3, 1.96
SEED = 30


def draw_tasks(rng, n):
    """Pass chances for agent A and B on n fresh tasks."""
    logit = rng.normal(0.3, 1.8, n)     # tasks differ a lot
    pa = 1 / (1 + np.exp(-logit))
    pb = 1 / (1 + np.exp(-(logit + 0.6)))  # B is a bit better
    return pa, pb


def run(rng, p):
    """TASKS x K grid of 0/1 trial scores."""
    return (rng.random((len(p), K)) < p[:, None]).astype(float)


def naive_se(grid):
    s = grid.ravel()
    return s.std(ddof=1) / math.sqrt(s.size)


def cluster_se(grid):
    """Miller's clustered SE; each row (task) is one cluster."""
    s = grid.ravel()
    n, r = s.size, grid - s.mean()
    cross = (r.sum(1) ** 2 - (r ** 2).sum(1)).sum()
    return math.sqrt(naive_se(grid) ** 2 + cross / n ** 2)


def task_mean_se(grid):
    """Miller's resampling recipe: SE across per-task means."""
    m = grid.mean(1)
    return m.std(ddof=1) / math.sqrt(m.size)


def paired_se(ga, gb):
    d = gb.mean(1) - ga.mean(1)
    return d.std(ddof=1) / math.sqrt(d.size)


def one_eval(seed=SEED):
    rng = np.random.default_rng(seed)
    pa, pb = draw_tasks(rng, TASKS)
    ga, gb = run(rng, pa), run(rng, pb)
    a, b = ga.mean(), gb.mean()
    print(f"A {a:.3f}  B {b:.3f}  gain {b - a:+.3f}")
    print(f"A naive SE   {naive_se(ga):.4f}")
    print(f"A cluster SE {cluster_se(ga):.4f}"
          f"  (x{cluster_se(ga) / naive_se(ga):.2f})")
    print(f"A task-mean SE {task_mean_se(ga):.4f}")
    split = sum(0 < r.sum() < K for r in ga)
    print(f"A tasks that split pass/fail: {split} of {TASKS}")
    un = math.hypot(cluster_se(ga), cluster_se(gb))
    nv = math.hypot(naive_se(ga), naive_se(gb))
    pr = paired_se(ga, gb)
    corr = np.corrcoef(ga.mean(1), gb.mean(1))[0, 1]
    print(f"gain SE naive-unpaired {nv:.4f}")
    print(f"gain SE unpaired       {un:.4f}")
    print(f"gain SE paired         {pr:.4f}  corr {corr:.2f}")
    for name, se in (("naive", nv), ("unpaired", un),
                     ("paired", pr)):
        lo, hi = b - a - Z * se, b - a + Z * se
        print(f"  95% CI {name:8s} {lo:+.3f} to {hi:+.3f}")


def check(reps=20_000, seed=1):
    """Fresh tasks + fresh trials each time: is the bar honest?"""
    rng = np.random.default_rng(seed)
    pa, pb = draw_tasks(rng, 1_000_000)
    true_a, true_g = pa.mean(), (pb - pa).mean()
    est, gain = [], []
    hit = {"naive": 0, "cluster": 0, "paired": 0, "unpaired": 0}
    found = {"paired": 0, "unpaired": 0}
    ratio, ses = [], {"naive": [], "cluster": []}
    for _ in range(reps):
        qa, qb = draw_tasks(rng, TASKS)
        ga, gb = run(rng, qa), run(rng, qb)
        m, g = ga.mean(), gb.mean() - ga.mean()
        est.append(m)
        gain.append(g)
        hit["naive"] += abs(m - true_a) < Z * naive_se(ga)
        hit["cluster"] += abs(m - true_a) < Z * cluster_se(ga)
        un = math.hypot(cluster_se(ga), cluster_se(gb))
        hit["unpaired"] += abs(g - true_g) < Z * un
        pr = paired_se(ga, gb)
        hit["paired"] += abs(g - true_g) < Z * pr
        found["unpaired"] += g > Z * un
        found["paired"] += g > Z * pr
        ratio.append(cluster_se(ga) / naive_se(ga))
        ses["naive"].append(naive_se(ga))
        ses["cluster"].append(cluster_se(ga))
    print(f"true A {true_a:.3f}  true gain {true_g:+.3f}")
    print(f"real spread of A's score   {np.std(est):.4f}")
    print(f"real spread of the gain    {np.std(gain):.4f}")
    for k, v in ses.items():
        print(f"average {k:7s} SE of A      {np.mean(v):.4f}")
    for k, v in hit.items():
        print(f"95% bar covers truth, {k:8s} {100 * v / reps:.1f}%")
    print(f"cluster SE / naive SE, median x{np.median(ratio):.2f}")
    for k, v in found.items():
        print(f"gain called real, {k:8s} {100 * v / reps:.1f}%")


if __name__ == "__main__":
    one_eval()
    print()
    check()
