"""Pipeline bubble simulator for training: AFAB, 1F1B and interleaved 1F1B.

Card: perf-7-training-economics/parallelism-4d.html
Run:  python3 labs/pp-bubble.py        (stdlib only, well under a second)

What it does
  1. Simulates p pipeline stages running m micro-batches. Each stage executes its
     own ordered list of forward (F) and backward (B) jobs; a job starts when the
     stage is free AND the job it depends on (previous stage's F, next stage's B)
     has finished. Forward = 1 time unit, backward = 2 (the usual t_b ~ 2 t_f).
     Point-to-point sends are free (ideal network).
  2. Measures the bubble ratio (idle time / useful time, the Playbook's and
     Narayanan's definition) and compares it with (p-1)/m and (p-1)/(v*m).
  3. Measures activation memory: the peak number of micro-batches whose
     activations a stage is holding (forward done, backward not yet done).
  4. Prints a text timeline for p = 4, m = 8 (the card's SVG is drawn from it).
  5. Zig-zag context parallelism: causal-attention work per GPU, sequential vs zig-zag.
  6. Llama 3 405B layout arithmetic (Table 4) and per-GPU model-state memory,
     Megatron 1T arithmetic, TP limits, and sequence-parallel activation sizes.
"""

P_FWD, P_BWD = 1.0, 2.0


def schedule(kind, p, m, v=1):
    """Per-stage ordered job lists. A job is (op, microbatch, chunk)."""
    orders = []
    for s in range(p):
        if kind == "afab":
            o = [("F", i, 0) for i in range(m)] + [("B", i, 0) for i in range(m)]
        elif kind == "1f1b":
            warm = min(p - s - 1, m)
            o = [("F", i, 0) for i in range(warm)]
            f, b = warm, 0
            while f < m:
                o.append(("F", f, 0)); f += 1
                o.append(("B", b, 0)); b += 1
            while b < m:
                o.append(("B", b, 0)); b += 1
        elif kind == "interleaved":
            # Megatron-LM interleaved 1F1B ordering (needs m % p == 0).
            assert m % p == 0, "interleaved schedule needs m to be a multiple of p"
            total = m * v

            def fwd(k):
                return ("F", (k // (p * v)) * p + k % p, (k // p) % v)

            def bwd(k):
                return ("B", (k // (p * v)) * p + k % p, v - 1 - (k // p) % v)

            warm = min((p - s - 1) * 2 + (v - 1) * p, total)
            o = [fwd(k) for k in range(warm)]
            f, b = warm, 0
            while f < total:
                o.append(fwd(f)); f += 1
                o.append(bwd(b)); b += 1
            while b < total:
                o.append(bwd(b)); b += 1
        else:
            raise ValueError(kind)
        orders.append(o)
    return orders


def simulate(kind, p, m, v=1, tf=P_FWD, tb=P_BWD, slow0=1.0):
    orders = schedule(kind, p, m, v)
    tf_c, tb_c = tf / v, tb / v          # one chunk holds 1/v of the stage's layers
    last = v - 1
    done = {}                            # (op, mb, chunk, stage) -> finish time
    free = [0.0] * p
    idx = [0] * p
    spans = [[] for _ in range(p)]
    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for s in range(p):
            while idx[s] < len(orders[s]):
                op, mb, c = orders[s][idx[s]]
                if op == "F":
                    dep = None if (s == 0 and c == 0) else ((op, mb, c, s - 1) if s > 0 else (op, mb, c - 1, p - 1))
                    dur = tf_c
                else:
                    if s == p - 1 and c == last:
                        dep = ("F", mb, c, s)
                    elif s < p - 1:
                        dep = ("B", mb, c, s + 1)
                    else:
                        dep = ("B", mb, c + 1, 0)
                    dur = tb_c
                if dep is not None and dep not in done:
                    break
                if s == 0:
                    dur *= slow0             # stage 0 slower (e.g. it also holds the embedding)
                start = max(free[s], done[dep] if dep else 0.0)
                end = start + dur
                done[(op, mb, c, s)] = end
                free[s] = end
                spans[s].append((start, end, op, mb, c))
                idx[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("deadlock: schedule order is inconsistent")
    makespan = max(free)
    ideal = m * (tf + tb)
    # activation memory: F adds 1/v of a micro-batch's stage activations, B frees it
    peak = []
    for s in range(p):
        ev = sorted([(a, +1) for a, _, op, _, _ in spans[s] if op == "F"] +
                    [(e, -1) for _, e, op, _, _ in spans[s] if op == "B"], key=lambda x: (x[0], x[1]))
        cur = mx = 0
        for _, d in ev:
            cur += d
            mx = max(mx, cur)
        peak.append(mx / v)
    return {"makespan": makespan, "ideal": ideal, "ratio": (makespan - ideal) / ideal,
            "idle_share": (makespan - ideal) / makespan, "peak": peak, "spans": spans}


def timeline(res, p, scale=1.0):
    width = int(res["makespan"] * scale + 0.5)
    rows = []
    for s in range(p):
        line = ["."] * width
        for a, e, op, mb, c in res["spans"][s]:
            ch = str(mb % 10) if op == "F" else "abcdefghij"[mb % 10]
            for t in range(int(a * scale + 0.5), int(e * scale + 0.5)):
                line[t] = ch
        rows.append(f"  stage {s}  " + "".join(line))
    return "\n".join(rows)


def main():
    p, m = 4, 8
    print("=" * 72)
    print(f"1. Pipeline schedules, p = {p} stages, m = {m} micro-batches, t_f = 1, t_b = 2")
    print("=" * 72)
    rows = [("naive (m = 1)", "afab", 1, 1), ("AFAB", "afab", m, 1), ("1F1B", "1f1b", m, 1),
            ("interleaved v=2", "interleaved", m, 2), ("interleaved v=4", "interleaved", m, 4)]
    print(f"  {'schedule':<17}{'step time':>10}{'ideal':>7}{'bubble ratio':>14}{'formula':>9}"
          f"{'idle share':>12}{'peak act. (stage 0)':>21}")
    results = {}
    for name, kind, mm, v in rows:
        r = simulate(kind, p, mm, v)
        results[name] = r
        formula = (p - 1) / (v * mm)
        print(f"  {name:<17}{r['makespan']:>10.2f}{r['ideal']:>7.1f}{r['ratio']:>13.1%}{formula:>9.1%}"
              f"{r['idle_share']:>12.1%}{r['peak'][0]:>14.2f} m-b")
    print("  bubble ratio = (step - ideal) / ideal   (Playbook r_bubble, Narayanan 'bubble time fraction')")
    print("  idle share   = (step - ideal) / step    = (p-1)/(m+p-1) for AFAB and 1F1B")
    print("  peak act.    = most micro-batches whose activations stage 0 holds at once")
    print(f"  per-stage peak, AFAB: {results['AFAB']['peak']}   1F1B: {results['1F1B']['peak']}"
          f"   interleaved v=2: {results['interleaved v=2']['peak']}")

    print("\n  Timelines (digit = forward of micro-batch k, letter = its backward, . = idle; 1 char = 0.5 units)")
    for name in ["AFAB", "1F1B", "interleaved v=2"]:
        print(f"  {name}: step = {results[name]['makespan']:.1f} units")
        print(timeline(results[name], p, scale=2.0))

    print("\n" + "=" * 72)
    print("2. Bubble ratio vs micro-batches, 1F1B (formula (p-1)/m, checked by simulation)")
    print("=" * 72)
    for pp in [4, 8, 16]:
        out = []
        for mm in [pp, 2 * pp, 4 * pp, 8 * pp]:
            r = simulate("1f1b", pp, mm)
            out.append(f"m={mm:>3}: {r['ratio']:6.1%}")
        print(f"  p = {pp:>2}   " + "   ".join(out))
    print("  Interleaving divides it by v:")
    for pp, mm in [(8, 16), (16, 32)]:
        out = []
        for v in [1, 2, 4]:
            kind = "1f1b" if v == 1 else "interleaved"
            r = simulate(kind, pp, mm, v)
            out.append(f"v={v}: {r['ratio']:6.1%} (formula {(pp - 1) / (v * mm):.1%}), peak act. {r['peak'][0]:.2f}")
        print(f"  p = {pp:>2}, m = {mm:>2}   " + "   ".join(out))

    print("\n" + "=" * 72)
    print("3. Zig-zag context parallelism: causal attention work per GPU (CP = 4)")
    print("=" * 72)
    cp = 4
    # sequential: GPU i owns chunk i of cp; work = blocks of the causal (lower-triangular) score matrix it computes
    seq_work = [i + 0.5 for i in range(cp)]                       # in (S/cp)^2 blocks
    n = 2 * cp
    zz_work = [((i + 0.5) + (n - 1 - i + 0.5)) / 4 for i in range(cp)]  # in (S/cp)^2 blocks: small chunk = 1/4 block
    for name, w in [("sequential chunks", seq_work), ("zig-zag (i, 2CP-1-i)", zz_work)]:
        mean = sum(w) / len(w)
        print(f"  {name:<22} work per GPU {[round(x, 2) for x in w]}   slowest / mean = {max(w) / mean:.2f}")

    print("\n" + "=" * 72)
    print("4. Llama 3 405B pre-training layouts (Table 4) and per-GPU model state")
    print("=" * 72)
    table4 = [(8192, 8, 1, 16, 64, 8192, 32, 430), (16384, 8, 1, 16, 128, 8192, 16, 400),
              (16384, 8, 16, 16, 8, 131072, 16, 380)]
    for g, tp, cpd, pp, dp, seq, bpd, tf in table4:
        tokens = dp * bpd * seq
        print(f"  {g:>6} GPUs = TP{tp} x CP{cpd} x PP{pp} x DP{dp} = {tp * cpd * pp * dp:>6}   "
              f"tokens/batch = {dp} x {bpd} x {seq:,} = {tokens / 1e6:.1f}M   tokens per CP rank = {seq // cpd:,}   "
              f"{tf} TF / 989 = {tf / 989:.1%}   tokens per GPU per step = {tokens // g:,}")
    N = 405e9
    shards = 8 * 16
    for label, rest in [("16 B/param (ZeRO/Playbook)", 14), ("20 B/param (FP32 grads, as Llama 3)", 18)]:
        params = 2 * N / shards
        other = rest * N / shards / 64
        print(f"  {label}: BF16 params 2N/(TP*PP) = {params / 1e9:.2f} GB  +  grads+optimizer {rest}N/(TP*PP*DP64) = "
              f"{other / 1e9:.2f} GB  =  {(params + other) / 1e9:.2f} GB per GPU (activations extra)")
    print(f"  all training state at 16 B/param = {16 * N / 1e9:,.0f} GB; serving weights BF16 = {2 * N / 1e9:,.0f} GB "
          f"on TP8 x PP2 = {2 * N / 16 / 1e9:.1f} GB per GPU")
    for mm in [32, 16]:
        print(f"  PP16 with m = {mm} micro-batches: (p-1)/m = {15 / mm:.1%}   v=2: {15 / (2 * mm):.1%}   v=4: {15 / (4 * mm):.1%}"
              "   (micro-batch size and v not published: illustrative)")

    print("\n" + "=" * 72)
    print("5. Megatron 1T (Narayanan 2021, Table 1) and TP limits")
    print("=" * 72)
    print(f"  3072 A100 = TP8 x PP64 x DP{3072 // (8 * 64)};  3072 x 163 TF = {3072 * 163 / 1000:.1f} PF (paper: 502)")
    print(f"  163 / 312 peak = {163 / 312:.1%}  (counts recompute: F = 96 B s l h^2, i.e. 4 passes)")
    print(f"  without recompute FLOPs: x 72/96 = {163 / 312 * 72 / 96:.1%}  (MFU-like, our recomputation)")
    # C/W values quoted, not re-derived: TPU v5p from Scaling Book Part 5 (4.59e14 / 1.8e11);
    # H100 from the Scaling Book GPU chapter: "about F / 2200 or F / 2475 beyond a node".
    for name, a in [("TPU v5p ICI (Part 5)", 4.59e14 / 1.8e11), ("H100 in a node (GPU chapter)", 2200),
                    ("H100 beyond a node (GPU chapter)", 2475)]:
        print(f"  {name:<34} C/W = {a:>6,.0f}   TP bound Y < F/(C/W):  70B F=28,672 -> {28672 / a:5.1f}"
              f"   405B F=53,248 -> {53248 / a:5.1f}")
    print("\n" + "=" * 72)
    print("6. Sequence parallelism: activation bytes per layer (Korthikanti Table 2, attention scores not stored)")
    print("=" * 72)
    s, b, h, t = 8192, 1, 16384, 8
    sbh = s * b * h
    for name, coef in [("no parallelism  sbh*34", 34), ("TP only  sbh*(10 + 24/t)", 10 + 24 / t),
                       ("TP + SP  sbh*34/t", 34 / t)]:
        print(f"  {name:<26} coefficient {coef:6.2f}   at s=8,192 h=16,384 t=8: {sbh * coef / 1e9:5.2f} GB per layer per sequence")
    print(f"  TP+SP vs TP only: {(10 + 24 / t) / (34 / t):.2f}x less  (GPT formula on 405B's shape: illustrative)")


if __name__ == "__main__":
    main()

# Try this:
# 1. Set P_BWD = 1.0 (a forward-only "pipeline", like serving): the bubble ratio
#    formula still holds but each bubble is a third as long in absolute time.
# 2. simulate("1f1b", 16, 16) vs simulate("interleaved", 16, 16, 4): the Llama 3
#    DP=128 row halves micro-batches per pipeline; see how much interleaving buys back.
# 3. simulate("1f1b", 4, 8, slow0=1.2): a stage 20% slower sets the pace for all;
#    Llama 3 drops one layer from the first and last stages for this reason.
