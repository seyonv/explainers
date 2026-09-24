"""Delay bounds, timeouts and AIMD congestion control.

Prints every number on the card delay-timeouts-congestion.html.
Python 3.10, standard library only.
"""


# 1. QJUMP worst-case delay (Grosvenor et al., NSDI 2015)
def epoch_us(n, p_bytes, r_bps, eps_us, meso=True):
    """tau <= n*P/R + eps; doubled for a mesochronous network."""
    k = 2 if meso else 1
    return k * n * p_bytes * 8 / r_bps * 1e6 + eps_us


def rate_per_host(n, f, r_bps):
    """n' = n / f senders; each may send about R / (2 n')."""
    return r_bps / (2 * n / f)


# 2. AIMD: +1 packet per loss-free RTT, halve on loss
def aimd(rtts=40, cap=16, cwnd=1):
    trace = []
    for _ in range(rtts):
        trace.append(cwnd)
        if cwnd > cap:            # queue overflowed: a drop
            cwnd = cwnd // 2      # multiplicative decrease
        else:
            cwnd += 1             # additive increase
    return trace


def per_ack_growth(cwnd):
    w = float(cwnd)
    for _ in range(cwnd):     # one RTT = cwnd ACKs
        w += 1 / w                # the source's cwnd += 1/cwnd
    return w


# 3. Two flows sharing one link: why multiplicative decrease
def two_flows(rule, a=1.0, b=12.0, cap=16, rtts=40):
    for _ in range(rtts):
        if a + b > cap and rule == "AIMD":   # both see the loss
            a, b = a / 2, b / 2
        elif a + b > cap:                     # AIAD: subtract 1
            a, b = a - 1, b - 1
        else:
            a, b = a + 1, b + 1
    return a, b


if __name__ == "__main__":
    n, P, R, eps = 1000, 256, 10e9, 4
    print(f"n*P/R + eps   = {epoch_us(n, P, R, eps, False):.1f} us")
    print(f"2n*P/R + eps  = {epoch_us(n, P, R, eps):.1f} us")
    for f in (1, 10, 100, 1000):
        npr = n // f
        ep = 2 * npr * P * 8 / R * 1e6 + eps
        mbps = rate_per_host(n, f, R) / 1e6
        print(f"f={f:5}  n'={npr:5}  epoch={ep:7.1f} us"
              f"  rate~{mbps:7.0f} Mb/s")

    t = aimd()
    print("cwnd per RTT:", t)
    drops = [i for i, w in enumerate(t) if w > 16]
    print("drops at RTT:", drops)
    print(f"sent {sum(t)} packets in 40 RTTs, mean"
          f" {sum(t) / 40:.2f}/RTT = {sum(t) / 40 / 16:.0%} of 16")
    steady = t[drops[0] + 1:drops[1] + 1]
    print(f"steady cycle {steady}: mean"
          f" {sum(steady) / len(steady):.1f}")
    print(f"per-ACK 1/cwnd from 8: {per_ack_growth(8):.3f}")

    for rule in ("AIMD", "AIAD"):
        a, b = two_flows(rule)
        print(f"{rule}: start 1 vs 12 -> {a:.2f} vs {b:.2f}"
              f" after 40 RTTs (gap {abs(b - a):.2f})")
