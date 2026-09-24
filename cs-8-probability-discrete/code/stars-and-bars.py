"""Stars and bars: the $20k-in-$1k-units investment problem."""
from itertools import combinations, product
from math import comb


def stars_and_bars(n, k):
    """Ways to put n identical units into k bins (each >= 0)."""
    return comb(n + k - 1, k - 1)


def with_minimums(total, mins):
    """Ways to split total with bin i getting at least mins[i]."""
    extra = total - sum(mins)            # hand out the minimums first
    return stars_and_bars(extra, len(mins)) if extra >= 0 else 0


def at_least(total, mins, r):
    """Ways where at least r of the bins are used (the rest get 0)."""
    k = len(mins)
    return sum(with_minimums(total, [mins[i] for i in used])
               for size in range(r, k + 1)
               for used in combinations(range(k), size))


def brute(total, mins, r):
    """Check every (x1..xk) with 0 <= xi <= total directly."""
    count = 0
    for xs in product(range(total + 1), repeat=len(mins)):
        ok = all(x == 0 or x >= m for x, m in zip(xs, mins))
        used = sum(x > 0 for x in xs)
        if sum(xs) == total and ok and used >= r:
            count += 1
    return count


def arrangement(extras):
    """Draw one split as stars and bars, e.g. (2, 4, 0, 3)."""
    return "|".join("*" * e for e in extras)


if __name__ == "__main__":
    mins = [2, 2, 3, 4]
    print("part 1, all four:", with_minimums(20, mins))
    for skip in range(4):
        rest = [m for i, m in enumerate(mins) if i != skip]
        print(f"  skip #{skip + 1} (min {mins[skip]}):",
              with_minimums(20, rest))
    print("part 2, at least three:", at_least(20, mins, 3))
    print("brute force:", brute(20, mins, 4), brute(20, mins, 3))
    extras = (2, 4, 0, 3)
    print(arrangement(extras), "->",
          [e + m for e, m in zip(extras, mins)])
