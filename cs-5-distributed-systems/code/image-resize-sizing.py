"""Sizing the source guide's image-resize service, step by step.

Source: ljeng/cheat-sheet, distributed-systems.md, The Reality.
Inputs are the source's; everything else is computed here.
Python 3.10, standard library only.
"""
import math

RPS = 10_000          # requests per second, per region (given)
PER_CORE = 10         # conversions/s per core at 256 KB (assumed)
CORES_PER_BOX = 32    # 32-core machines (given)
HEADROOM = 0.20       # 20% for surges (given)


def size(rps, per_core, cores_per_box, headroom):
    cores = rps / per_core                    # 10,000 / 10
    boxes = math.ceil(cores / cores_per_box)  # 31.25 -> 32
    times = math.ceil(boxes * (1 + headroom)) # 38.4 -> 39
    idle = math.ceil(boxes / (1 - headroom))  # 40.0 -> 40
    return cores, boxes, times, idle


def erlang_c(c, a):
    """P(a request waits) in an M/M/c queue, load a erlangs."""
    term, total = 1.0, 1.0
    for k in range(1, c):
        term *= a / k
        total += term
    tail = term * a / c / (1 - a / c)
    return tail / (total + tail)


def p99_wait(boxes, rps=RPS, per_core=PER_CORE, c=CORES_PER_BOX):
    lam = rps / boxes                  # requests/s reaching one box
    rho = lam / (c * per_core)         # utilisation of that box
    pw = erlang_c(c, lam / per_core)
    rate = c * per_core - lam          # wait tail: pw * e^(-rate t)
    t99 = max(0.0, math.log(pw / 0.01) / rate)
    return rho, pw, t99


if __name__ == "__main__":
    cores, boxes, times, idle = size(RPS, PER_CORE, CORES_PER_BOX,
                                     HEADROOM)
    print(f"cores   = {RPS:,} / {PER_CORE} = {cores:,.0f}")
    print(f"servers = {cores:,.0f} / {CORES_PER_BOX} = "
          f"{cores / CORES_PER_BOX} -> {boxes}")
    print(f"x 1.2   = {boxes * 1.2:.1f} -> {times}")
    print(f"/ 0.8   = {boxes / 0.8:.1f} -> {idle}   (source: >= 40)")
    print(f"Little: in flight = {RPS:,} x {1 / PER_CORE} s = "
          f"{RPS / PER_CORE:,.0f} busy cores")

    for n in (boxes, idle):
        rho, pw, t99 = p99_wait(n)
        print(f"{n} servers: util {rho:.1%}  P(wait) {pw:.1%}  "
              f"p99 queue wait {t99 * 1000:.0f} ms")

    kib = 256 * 1024
    gbps = RPS * kib * 8 / 1e9
    print(f"ingress = {RPS:,} x 256 KiB = {RPS * kib / 1e9:.2f} GB/s"
          f" = {gbps:.1f} Gb/s")
    for slo in (0.999, 0.9999):
        month = (1 - slo) * 30 * 24 * 60
        year = (1 - slo) * 365 * 24
        print(f"{slo:.2%} -> {month:.1f} min/month, {year:.2f} h/year")
