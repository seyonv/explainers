"""Expected-value games and market making.

Roll a fair die and get paid the face. You may reroll up to n
times; you keep the last roll. Backward induction prices the game,
then a market maker quotes around that price and meets informed flow.
"""
from fractions import Fraction as F

FACES = range(1, 7)


def game_value(n):
    """Value with n rerolls left: V(n) = E[max(X, V(n-1))]."""
    v = F(sum(FACES), 6)                  # n = 0: must keep, 3.5
    for _ in range(n):
        v = sum(max(F(x), v) for x in FACES) / 6
    return v


def trace(n):
    """(rerolls left, threshold, faces kept, value) for each stage."""
    rows, v = [(0, None, list(FACES), F(7, 2))], F(7, 2)
    for k in range(1, n + 1):
        keep = [x for x in FACES if x > v]
        new = sum(max(F(x), v) for x in FACES) / 6
        rows.append((k, v, keep, new))
        v = new
    return rows


def fixed_threshold(n, t=3.5):
    """Always keep faces > t, whatever the rerolls left."""
    v = F(7, 2)
    for _ in range(n):
        v = sum(F(x) if x > t else v for x in FACES) / 6
    return v


def mm_pnl(half, alpha, fair=F(7, 2)):
    """Market maker's expected P&L per arriving trader.

    Quotes bid = fair - half, ask = fair + half on one die roll.
    Uninformed traders (1 - alpha) buy or sell at random.
    Informed traders (alpha) know the roll and trade only
    when it pays: buy if roll > ask, sell if roll < bid.
    """
    bid, ask = fair - half, fair + half
    noise = half                           # earn the half-spread
    loss = sum(max(x - ask, 0) + max(bid - x, 0) for x in FACES) / 6
    return (1 - alpha) * noise - alpha * loss


def ev_given_buy(half, alpha, fair=F(7, 2)):
    """E[roll | someone lifts the ask]: what a buy tells you."""
    ask = fair + half
    w = {x: (1 - alpha) / 2 + alpha * (x > ask) for x in FACES}
    return sum(x * w[x] for x in FACES) / sum(w.values())


def break_even_half(alpha, lo=F(0), hi=F(3), steps=60):
    """Smallest half-spread with P&L >= 0 (bisection)."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        if mm_pnl(mid, alpha) >= 0:
            hi = mid
        else:
            lo = mid
    return float(hi)


if __name__ == "__main__":
    for n in range(4):
        v = game_value(n)
        print(f"{n} rerolls: {str(v):>5} = {float(v):.4f}")
    print("trace for 3 rerolls:")
    for k, t, keep, v in trace(3):
        ts = "-" if t is None else f"{float(t):.4f}"
        print(f"  left={k} keep>{ts:>6} keep={keep} V={float(v):.4f}")
    two = [F(max(a, b)) for a in FACES for b in FACES]
    print("E[max of 2 dice] =", sum(two) / 36,
          f"= {float(sum(two) / 36):.4f}")
    print("fixed threshold 3.5, 2 rerolls:", fixed_threshold(2),
          f"= {float(fixed_threshold(2)):.4f}")
    print("market on one die, quotes 3.0 / 4.0:")
    for a in (F(0), F(1, 5), F(1, 3), F(1, 2)):
        print(f"  informed {float(a):.3f}: P&L per trader "
              f"{float(mm_pnl(F(1, 2), a)):+.4f}")
    for a in (F(0), F(1, 5), F(1, 3), F(1, 2)):
        print(f"  informed {float(a):.3f}: E[roll | buy at 4.0] = "
              f"{float(ev_given_buy(F(1, 2), a)):.4f}")
    for a in (F(0), F(1, 10), F(1, 5), F(1, 2)):
        h = break_even_half(a)
        print(f"  informed {float(a):.1f}: break-even quotes "
              f"{3.5 - h:.4f} / {3.5 + h:.4f}")
