"""Reliability and quality: failures, canaries and evals (course 7).

Run: python3 labs/failure-sim.py   (stdlib + numpy, CPU, a few seconds)

Nothing here is measured. It is a calculator plus a seeded Monte Carlo:
  1. Llama 3 Table 5: counts, printed %, and the % recomputed from the counts
  2. Training: interruptions per day, and effective training time vs
     checkpoint interval (first-order model, checked by simulation)
  3. Serving: expected failures per day and per year for a fleet of N GPUs
     at Kiely's ~1 failure per 50,000 GPU-hours
  4. Availability of 16 H100s serving 500 users of Llama-3.1-70B
     (sizing-deployment.html's workload): TP8 x 2 vs TP4 x 4, with and
     without a warm spare node, failure = group only vs whole node drained
  5. Rollout cost: blue-green vs canary for Kiely's 100-GPU example

Sourced inputs: Llama 3 (arXiv 2407.21783) §3.3.4 + Table 5; Kiely,
Inference Engineering §7.3.3 (1 per 50,000 GPU-hours) and §7.4.1 (blue-green
"another 100 GPUs"); users per replica from ../perf-6-scaling-out/labs/sizing.py.
Everything marked ILLUSTRATIVE is a made-up operating assumption.
"""
import math
import numpy as np

SEED = 7
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. Llama 3 Table 5
print("=" * 72)
print("1. Llama 3 405B, 54-day snapshot on 16,384 H100s (Table 5)")
print("=" * 72)
DAYS, GPUS_L3 = 54, 16384
TOTAL, PLANNED, UNEXPECTED = 466, 47, 419
table5 = [  # (root cause, count, printed %)
    ("Faulty GPU", 148, 30.1), ("GPU HBM3 memory", 72, 17.2),
    ("Software bug", 54, 12.9), ("Network switch/cable", 35, 8.4),
    ("Host maintenance", 32, 7.6), ("GPU SRAM memory", 19, 4.5),
    ("GPU system processor", 17, 4.1), ("NIC", 7, 1.7),
    ("NCCL watchdog timeouts", 7, 1.7), ("Silent data corruption", 6, 1.4),
    ("GPU thermal interface + sensor", 6, 1.4), ("SSD", 3, 0.7),
    ("Power supply", 3, 0.7), ("Server chassis", 2, 0.5),
    ("IO expansion board", 2, 0.5), ("Dependency", 2, 0.5),
    ("CPU", 2, 0.5), ("System memory", 2, 0.5),
]
print(f"{'cause':32s} {'count':>5s} {'printed':>8s} {'count/419':>10s}")
for name, n, pct in table5:
    flag = "  <- doesn't match" if abs(100 * n / UNEXPECTED - pct) > 0.1 else ""
    print(f"{name:32s} {n:5d} {pct:7.1f}% {100*n/UNEXPECTED:9.1f}%{flag}")
cnt = sum(n for _, n, _ in table5)
pr = sum(p for _, _, p in table5)
print(f"sum of counts = {cnt} (text says 419 unexpected)   sum of printed % = {pr:.1f}%"
      f" (with 35.3% for Faulty GPU: {pr - 30.1 + 100*148/UNEXPECTED:.1f}%)")
gpu_rows = ["Faulty GPU", "GPU HBM3 memory", "GPU SRAM memory",
            "GPU system processor", "Silent data corruption",
            "GPU thermal interface + sensor"]
gp = sum(p for nm, _, p in table5 if nm in gpu_rows)
gc = sum(n for nm, n, _ in table5 if nm in gpu_rows)
print(f"GPU-type rows: printed % add to {gp:.1f}% (text: 'GPU issues ... 58.7%'), "
      f"but counts add to {gc} = {100*gc/UNEXPECTED:.1f}% of 419")
print(f"148 printed as 30.1% implies a base of {148/0.301:.0f} (not 419 or 466); "
      f"30.1% of 419 = {0.301*419:.0f}")

gpu_hours = GPUS_L3 * DAYS * 24
print(f"\nGPU-hours in the snapshot: 16,384 x 54 x 24 = {gpu_hours/1e6:.2f}M")
print(f"per unexpected interruption: {gpu_hours:,.0f} / 419 = {gpu_hours/UNEXPECTED:,.0f} GPU-hours "
      f"(Kiely rounds to ~50,000)")
print(f"interruptions per day: all {TOTAL/DAYS:.2f}, unexpected {UNEXPECTED/DAYS:.2f}, "
      f"planned {PLANNED/DAYS:.2f}")
mtbi_all = DAYS * 24 / TOTAL
mtbi_unexp = DAYS * 24 / UNEXPECTED
print(f"mean time between interruptions of the job: all {mtbi_all:.2f} h, "
      f"unexpected {mtbi_unexp:.2f} h ({mtbi_unexp*60:.0f} min)")

# ---------------------------------------------------------------- 2. Training
print("\n" + "=" * 72)
print("2. Training: effective training time vs checkpoint interval")
print("=" * 72)
# ILLUSTRATIVE: time to write one checkpoint (job paused) and to restart the
# whole job after an interruption (reschedule, reload, warm up NCCL). Llama 3
# says only that they 'reduced job startup and checkpointing time'.
CASES = [(1.0, 5.0), (1.0, 15.0), (1.0, 30.0), (0.5, 2.0)]   # (checkpoint C, restart R) in minutes
M_UNEXP = mtbi_unexp * 60           # minutes between unexpected failures
PLANNED_PER_MIN = PLANNED / (DAYS * 24 * 60)


def waste_model(tau, C, R, M=M_UNEXP):
    """First-order (Young-style) waste: checkpoint cost C/tau, plus per failure
    half an interval of lost work + restart R. Planned stops cost R only
    (they checkpoint first)."""
    return C / (tau + C) + (tau / 2 + R) / M + R * PLANNED_PER_MIN


def sim_effective(tau, C, R, days=54, trials=40):
    """Monte Carlo: exponential unexpected failures (lose work since last
    checkpoint, then restart) and planned stops (restart only)."""
    out = []
    horizon = days * 24 * 60.0
    for _ in range(trials):
        t, useful = 0.0, 0.0
        next_fail = rng.exponential(M_UNEXP)
        next_plan = rng.exponential(1 / PLANNED_PER_MIN)
        while t < horizon:
            seg_end = t + tau + C          # work tau, then write a checkpoint
            ev = min(next_fail, next_plan)
            if ev >= seg_end:
                useful += tau
                t = seg_end
                continue
            if next_plan <= next_fail:     # planned: checkpoint on the way out
                useful += min(ev - t, tau)
                t = ev + R
                next_plan = t + rng.exponential(1 / PLANNED_PER_MIN)
            else:                           # unexpected: lose the segment
                t = ev + R
                next_fail = t + rng.exponential(M_UNEXP)
            if next_fail < t:
                next_fail = t + rng.exponential(M_UNEXP)
            if next_plan < t:
                next_plan = t + rng.exponential(1 / PLANNED_PER_MIN)
        out.append(useful / t)
    return float(np.mean(out))


print(f"unexpected failure every {M_UNEXP:.0f} min; planned stop every "
      f"{1/PLANNED_PER_MIN/60:.1f} h (Llama 3 rates)")
print(f"{'ckpt C':>7s} {'restart R':>10s} {'best interval':>14s} {'model':>7s} {'sim':>7s} {'sim, every 60 min':>18s}")
for C, R in CASES:
    tau_opt = math.sqrt(2 * C * M_UNEXP)   # Young's approximation
    eff = 1 - waste_model(tau_opt, C, R)
    eff_sim = sim_effective(tau_opt, C, R)
    eff_60 = sim_effective(60, C, R)
    print(f"{C:5.1f} m {R:8.0f} m {tau_opt:11.1f} min {100*eff:6.1f}% {100*eff_sim:6.1f}% {100*eff_60:17.1f}%")
print("Llama 3 reports '>90% effective training time'.")
print(f"best interval = sqrt(2 x C x MTBF) = sqrt(2 x 1 x {M_UNEXP:.0f}) = "
      f"{math.sqrt(2*1*M_UNEXP):.1f} min for C = 1 min")

# ---------------------------------------------------------------- 3. Fleet failures
print("\n" + "=" * 72)
print("3. Serving: expected GPU failures at ~1 per 50,000 GPU-hours (Kiely)")
print("=" * 72)
RATE = 1 / 50_000                   # failures per GPU-hour
HOURS_YEAR = 24 * 365
print(f"one 8-GPU node-year = 8 x {HOURS_YEAR:,} = {8*HOURS_YEAR:,} GPU-hours "
      f"-> {8*HOURS_YEAR*RATE:.2f} failures")
print(f"{'GPUs':>7s} {'per day':>9s} {'per year':>9s} {'mean gap':>12s} {'P(no failure in a year)':>24s}")
for n in [8, 16, 64, 1000, 16384]:
    per_day = n * 24 * RATE
    per_year = n * HOURS_YEAR * RATE
    gap_h = 1 / (n * RATE)
    gap = f"{gap_h/24:,.1f} days" if gap_h >= 48 else f"{gap_h:,.1f} h"
    print(f"{n:7,d} {per_day:9.3f} {per_year:9.1f} {gap:>12s} {100*math.exp(-per_year):23.1f}%")
print(f"check: 16,384 GPUs -> {16384*24*RATE:.2f}/day vs Llama 3's {UNEXPECTED/DAYS:.2f} unexpected/day")

# ---------------------------------------------------------------- 4. Availability
print("\n" + "=" * 72)
print("4. 16 H100s, 500 users of Llama-3.1-70B: TP8 x 2 vs TP4 x 4")
print("=" * 72)
DEMAND = 500
LAYOUTS = {"TP8 x 2": (8, 2, 386), "TP4 x 4": (4, 4, 166)}  # tp, replicas, users each
# ILLUSTRATIVE operating assumptions
REPAIR_H = 24.0      # no spare: replace/cycle the node and bring it back
FAILOVER_H = 0.25    # warm spare node already holding weights: swap in (15 min)
YEARS = 20_000       # simulated years


def served(up_replicas, per_rep):
    return min(DEMAND, up_replicas * per_rep)


def expected_availability(tp, reps, per_rep, down_h, node_level):
    """Analytic, ignoring overlapping failures (rare at 2.8/yr)."""
    gpus = tp * reps
    fails = gpus * HOURS_YEAR * RATE
    lost_reps = (8 // tp) if node_level else 1
    lost_users = DEMAND - served(reps - lost_reps, per_rep)
    return 1 - fails * down_h * lost_users / (DEMAND * HOURS_YEAR), lost_users


def mc_availability(tp, reps, per_rep, down_h, node_level):
    """Seeded Monte Carlo over YEARS years: every GPU fails as a Poisson process;
    a failure takes down its group (or its whole node) for down_h hours."""
    gpus = tp * reps
    per_node = 8 // tp
    fails = rng.poisson(gpus * HOURS_YEAR * RATE, size=YEARS)
    lost_uh = np.zeros(YEARS)
    for y in np.nonzero(fails)[0]:
        k = fails[y]
        t = rng.uniform(0, HOURS_YEAR, size=k)
        gpu = rng.integers(0, gpus, size=k)
        rep = gpu // tp
        downs = []  # (start, end, set of replicas down)
        for ti, r in zip(t, rep):
            if node_level:
                node = r // per_node
                hit = set(range(node * per_node, node * per_node + per_node))
            else:
                hit = {int(r)}
            downs.append((ti, ti + down_h, hit))
        edges = sorted({0.0, HOURS_YEAR} | {min(e, HOURS_YEAR) for d in downs for e in d[:2]})
        for a, b in zip(edges[:-1], edges[1:]):
            mid = (a + b) / 2
            down = set()
            for s, e, hit in downs:
                if s <= mid < e:
                    down |= hit
            lost_uh[y] += (DEMAND - served(reps - len(down), per_rep)) * (b - a)
    avail = 1 - lost_uh / (DEMAND * HOURS_YEAR)
    return float(avail.mean()), float(np.percentile(avail, 1)), float((fails == 0).mean())


def nines(a):
    return -math.log10(1 - a) if a < 1 else float("inf")


print(f"failures per year on 16 GPUs: {16*HOURS_YEAR*RATE:.2f};  repair without a spare "
      f"{REPAIR_H:.0f} h, warm-spare failover {FAILOVER_H*60:.0f} min (both ILLUSTRATIVE)")
print(f"{'layout':8s} {'failure takes':14s} {'spare':6s} {'users left':>10s} "
      f"{'availability':>13s} {'nines':>6s} {'MC mean':>9s} {'MC worst 1%':>12s}")
results = {}
for name, (tp, reps, per_rep) in LAYOUTS.items():
    for node_level in (False, True):
        if tp == 8 and node_level:
            continue  # TP8 group == node: same thing
        for spare, down in (("none", REPAIR_H), ("node", FAILOVER_H)):
            a, lost = expected_availability(tp, reps, per_rep, down, node_level)
            mc, worst, p0 = mc_availability(tp, reps, per_rep, down, node_level)
            scope = "whole node" if (node_level or tp == 8) else "its group"
            results[(name, scope, spare)] = a
            print(f"{name:8s} {scope:14s} {spare:6s} {DEMAND-lost:10d} {100*a:12.4f}% "
                  f"{nines(a):6.2f} {100*mc:8.4f}% {100*worst:11.3f}%")
lam = tp * reps * HOURS_YEAR * RATE
print(f"share of simulated years with no failure at all ({name}): {100*p0:.1f}% "
      f"(analytic e^-{lam:.2f} = {100*math.exp(-lam):.1f}%)")
print("in-flight requests cut off per failure, load spread evenly: "
      f"TP8 x 2 {DEMAND//2}, TP4 x 4 {DEMAND//4} (group) or {DEMAND//2} (node drained at once)")
spare_gpus = {"TP8 x 2": 8, "TP4 x 4 (group)": 4, "TP4 x 4 (node)": 8}
print("GPUs added for the spare: " + ", ".join(f"{k} +{v}" for k, v in spare_gpus.items()))

# ---------------------------------------------------------------- 5. Rollouts
print("\n" + "=" * 72)
print("5. Rolling out a new version on Kiely's 100-GPU deployment")
print("=" * 72)
FLEET = 100
TP = 4                      # ILLUSTRATIVE replica size -> 25 replicas
MIN_WARM = 2                # ILLUSTRATIVE: canary never below 2 warm replicas
reps_total = FLEET // TP
print(f"blue-green: {FLEET} (blue) + {FLEET} (green) = {2*FLEET} GPUs during cut-over")
print(f"canary with autoscaling, {reps_total} replicas of TP{TP}, canary floor {MIN_WARM} replicas:")
print(f"{'canary %':>9s} {'old reps':>9s} {'new reps':>9s} {'GPUs':>6s} {'extra':>6s}")
for pct in [1, 5, 25, 50, 100]:
    new = max(MIN_WARM, math.ceil(reps_total * pct / 100))
    old = math.ceil(reps_total * (100 - pct) / 100)
    g = (new + old) * TP
    print(f"{pct:8d}% {old:9d} {new:9d} {g:6d} {g-FLEET:+6d}")

# Try this:
# 1. Set RATE = 1 / 25_000 (a fleet twice as flaky as Llama 3's) and REPAIR_H = 72:
#    TP8 x 2 without a spare falls to ~98.9% (under two nines); with a spare node it
#    still holds ~99.996%. Spares matter more than the layout.
# 2. Add (5.0, 5.0) to CASES (slow checkpoint storage): the best interval grows to
#    ~43 min and effective training time drops ~11 points below the (1, 5) row.
# 3. Change LAYOUTS to {"TP2 x 9": (2, 9, 57)} (the FP8 TP2 row of the sizing card):
#    each failure leaves 456 users (99.92% with no spare), but a whole-node drain
#    takes 4 replicas at once and leaves only 285.
