"""How MapReduce runs: a one-machine simulation of the execution.

A master holds task state (idle / in-progress / completed). M map
tasks read one split each and write R partitions to "local disk"
(a dict per worker), using hash(key) mod R with a stable hash.
An optional combiner pre-sums each partition before the shuffle.
If a worker dies, its completed map tasks go back to idle, because
their output lived on its disk; completed reduces are safe in the
global file system. Python 3.10, standard library only.
"""
from collections import defaultdict
from zlib import crc32

SPLITS = ["the cat sat the cat ate the fish",   # split 0: d1 + d2
          "a dog sat"]                           # split 1: d3
R = 2


def part(key):                       # stable hash(key) mod R
    return crc32(key.encode()) % R


def run_map(text, combine):
    regions = [defaultdict(list) for _ in range(R)]
    for w in text.split():
        regions[part(w)][w].append(1)            # emit (w, 1)
    if combine:                                  # combiner = reduce
        for reg in regions:
            for w in reg:
                reg[w] = [sum(reg[w])]
    return regions


def run_job(combine, kill_after_map=None):
    state = {("map", i): "idle" for i in range(len(SPLITS))}
    state |= {("reduce", r): "idle" for r in range(R)}
    disk, where, reruns = {}, {}, 0
    for i, text in enumerate(SPLITS):
        worker = f"w{i}"                         # one map per worker
        state["map", i] = "in-progress"
        disk[worker, i] = run_map(text, combine)
        where[i] = worker
        state["map", i] = "completed"
    if kill_after_map is not None:               # worker dies
        for i, w in list(where.items()):
            if w == kill_after_map:
                del disk[w, i]                  # its local disk is gone
                state["map", i] = "idle"         # completed -> idle
                disk["w9", i] = run_map(SPLITS[i], combine)
                where[i] = "w9"
                state["map", i] = "completed"
                reruns += 1
    out, shuffled = [], 0
    for r in range(R):
        state["reduce", r] = "in-progress"
        groups = defaultdict(list)
        for i in range(len(SPLITS)):             # remote read
            for w, vals in disk[where[i], i][r].items():
                groups[w] += vals
                shuffled += len(vals)
        out.append({w: sum(groups[w]) for w in sorted(groups)})
        state["reduce", r] = "completed"         # atomic rename
    return out, shuffled, reruns


def pair_bytes(n_pairs_by_word):
    # "word\t1\n" per pair: len(word) + 3 bytes (count < 10)
    return sum(n * (len(w) + 3) for w, n in n_pairs_by_word)


if __name__ == "__main__":
    for w in sorted(set(" ".join(SPLITS).split())):
        print(f"  {w:>4}: crc32 % {R} = {part(w)}")
    for combine in (False, True):
        out, n, _ = run_job(combine)
        print("combiner" if combine else "no combiner",
              f"-> {n} pairs shuffled")
    print("output files:", run_job(True)[0])
    raw = pair_bytes((w, 1) for s in SPLITS for w in s.split())
    combo = pair_bytes((w, 1) for s in SPLITS
                       for w in set(s.split()))
    print(f"shuffle bytes: {raw} -> {combo}"
          f" ({1 - combo / raw:.0%} less)")
    out2, _, reruns = run_job(True, kill_after_map="w0")
    print(f"w0 dies after its map: {reruns} map re-run,"
          f" same output: {out2 == run_job(True)[0]}")
    M = 2**40 // 2**26                           # 1 TB / 64 MB
    Rbig = 4000
    print(f"M = {M:,}  R = {Rbig:,}  M+R = {M + Rbig:,}"
          f"  M*R = {M * Rbig:,}")
