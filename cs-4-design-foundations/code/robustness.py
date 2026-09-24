"""Robustness: write the N x N version when it costs nothing extra.

Source: ljeng/cheat-sheet, large-scale-design/system-design/
robustness.md. "Just because you're asked to check if a tic-tac-toe
board has a winner doesn't mean you must assume a 3 x 3 board."

Board keeps one counter per row, per column and per diagonal.
X adds +1, O adds -1, so a line is won when |counter| == n.
Each move is O(1); a full rescan of the board is O(n^2).

The demo also checks the substitution-method guess for
T(n) = 2 T(n/2) + n  ->  T(n) <= n lg n  (CLRS 4.3).
"""
import math
import random


class Board:
    def __init__(self, n=3):            # n is a parameter, not a 3
        self.n = n
        self.rows, self.cols = [0] * n, [0] * n
        self.diag = self.anti = 0
        self.taken = set()

    def move(self, r, c, player):
        """Play (r, c) for player 'X' or 'O'; return winner or None."""
        n = self.n
        if not (0 <= r < n and 0 <= c < n) or (r, c) in self.taken:
            raise ValueError(f"illegal move {(r, c)}")
        self.taken.add((r, c))
        d = 1 if player == "X" else -1
        self.rows[r] += d
        self.cols[c] += d
        if r == c:
            self.diag += d
        if r + c == n - 1:
            self.anti += d
        lines = (self.rows[r], self.cols[c], self.diag, self.anti)
        return player if n in map(abs, lines) else None


def scan_winner(grid):
    """Brute force: read every cell of every line, O(n^2)."""
    n = len(grid)
    lines = [row for row in grid]
    lines += [[grid[r][c] for r in range(n)] for c in range(n)]
    lines.append([grid[i][i] for i in range(n)])
    lines.append([grid[i][n - 1 - i] for i in range(n)])
    for line in lines:
        if line[0] != "." and all(x == line[0] for x in line):
            return line[0]
    return None


def trace(moves, n=3):
    b = Board(n)
    for i, (p, r, c) in enumerate(moves, 1):
        w = b.move(r, c, p)
        print(f"{i} {p}({r},{c}) rows={b.rows} cols={b.cols} "
              f"diag={b.diag} anti={b.anti}" + (f" -> {w} wins" if w
                                                  else ""))
        if w:
            return w


def random_check(games=2000, seed=1):
    """Counters agree with a full rescan after every move."""
    rng = random.Random(seed)
    for _ in range(games):
        n = rng.randint(1, 6)
        b, grid = Board(n), [["."] * n for _ in range(n)]
        cells = [(r, c) for r in range(n) for c in range(n)]
        rng.shuffle(cells)
        for i, (r, c) in enumerate(cells):
            p = "XO"[i % 2]
            grid[r][c] = p
            if b.move(r, c, p) != scan_winner(grid):
                return False
            if scan_winner(grid):
                break
    return True


def T(n):
    """T(1) = 1, T(n) = 2 T(n/2) + n, for n a power of two."""
    return 1 if n == 1 else 2 * T(n // 2) + n


if __name__ == "__main__":
    print("3 x 3 game:")
    trace([("X", 1, 1), ("O", 0, 0), ("X", 0, 2),
           ("O", 1, 0), ("X", 2, 0)])
    print("same class, 4 x 4 (O takes column 3):")
    trace([("X", 0, 0), ("O", 0, 3), ("X", 1, 1), ("O", 1, 3),
           ("X", 2, 1), ("O", 2, 3), ("X", 3, 0), ("O", 3, 3)], n=4)
    try:
        Board(3).move(3, 0, "X")
    except ValueError as e:
        print("off-board move:", e)
    print("counters == full rescan, 2,000 random games n=1..6:",
          random_check())
    print("cell reads per move, rescan (2n+2)*n vs counters 4:")
    for n in (3, 10, 100):
        print(f"  n={n:<4} {(2 * n + 2) * n:>6,}")
    print("substitution check, guess T(n) <= 2 n lg n for n >= 2:")
    for k in (1, 4, 10, 20):
        n = 2 ** k
        print(f"  n=2^{k:<2} T={T(n):>10,}  n lg n={n * k:>10,}"
              f"  T/(n lg n)={T(n) / (n * k):.3f}")
