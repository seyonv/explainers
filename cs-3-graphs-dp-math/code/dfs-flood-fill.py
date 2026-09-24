"""Depth-first search and flood fill, iteratively, in plain Python.

Card: cs-3-graphs-dp-math/dfs-flood-fill.html
Source: ljeng/cheat-sheet, coding-algorithms/graphs.md, Depth-first
Search (flood_fill, flood_fill_border, Number of Enclaves) and
Adjacency List (Minesweeper). The source's code/graph.py is missing,
so flood_fill and flood_fill_border are written here from their docs.
"""
import sys

DIRS = {4: [(-1, 0), (1, 0), (0, -1), (0, 1)]}
DIRS[8] = DIRS[4] + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def flood_fill(grid, i, j, color, k=4):
    old = grid[i][j]
    if old == color:
        return
    grid[i][j] = color              # mark on push
    stack = [(i, j)]
    while stack:
        r, c = stack.pop()          # LIFO: deepest cell first
        for dr, dc in DIRS[k]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr][nc] == old):
                grid[nr][nc] = color    # mark now, not when popped
                stack.append((nr, nc))


def flood_fill_border(grid, color, k=4):
    R, C = len(grid), len(grid[0])
    for i in range(R):
        for j in range(C):
            if i in (0, R - 1) or j in (0, C - 1):
                flood_fill(grid, i, j, color, k)


def num_enclaves(grid):
    flood_fill_border(grid, 0)      # sink land touching the edge
    return sum(map(sum, grid))      # the rest can't walk off


def trace_fill(grid, i, j, color):
    """flood_fill (k=4) that records (popped, pushed, stack) per pop."""
    old, log = grid[i][j], []
    grid[i][j] = color
    stack = [(i, j)]
    while stack:
        r, c = stack.pop()
        pushed = []
        for dr, dc in DIRS[4]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr][nc] == old):
                grid[nr][nc] = color
                stack.append((nr, nc))
                pushed.append((nr, nc))
        log.append(((r, c), pushed, list(stack)))
    return log


def count_islands(grid, land=1, k=4):
    """Number of Islands: one flood fill per component."""
    grid = [row[:] for row in grid]
    islands = 0
    for i, row in enumerate(grid):
        for j, x in enumerate(row):
            if x == land:
                flood_fill(grid, i, j, -1, k)
                islands += 1
    return islands


def closed_islands(grid):
    """Number of Closed Islands: 0 = land, 1 = water."""
    grid = [row[:] for row in grid]
    flood_fill_border(grid, 1)      # border land becomes water
    return count_islands(grid, land=0)


def flood_fill_recursive(grid, i, j, color, old=None):
    """The textbook recursive version: one Python frame per cell."""
    if old is None:
        old = grid[i][j]
        if old == color:
            return
    if not (0 <= i < len(grid) and 0 <= j < len(grid[0])):
        return
    if grid[i][j] != old:
        return
    grid[i][j] = color
    for di, dj in DIRS[4]:
        flood_fill_recursive(grid, i + di, j + dj, color, old)


def minesweeper(board, click, mark_on_push=True):
    """Reveal a click. Returns the board and how many pushes it took.

    mark_on_push=False is the source's version: it marks cells as
    visited when popped, so a cell can sit on the stack several times.
    """
    i, j = click
    if board[i][j] == "M":
        board[i][j] = "X"
        return board, 0
    rows, cols = len(board), len(board[0])

    def around(r, c):
        for dr, dc in DIRS[8]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    stack, seen, pushes = [(i, j)], {(i, j)}, 1
    while stack:
        r, c = stack.pop()
        seen.add((r, c))
        mines = sum(board[a][b] == "M" for a, b in around(r, c))
        if mines:
            board[r][c] = str(mines)
            continue
        board[r][c] = "B"
        for a, b in around(r, c):
            if board[a][b] == "E" and (a, b) not in seen:
                if mark_on_push:
                    seen.add((a, b))
                stack.append((a, b))
                pushes += 1
    return board, pushes


if __name__ == "__main__":
    grid = [[0, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]]
    print(num_enclaves([r[:] for r in grid]))    # → 3

    # Trace 1: the border pass. Only (1, 0) is border land.
    g = [r[:] for r in grid]
    print("border fill from (1, 0):", trace_fill(g, 1, 0, 0))
    print("after border pass:", g)

    # Trace 2: flood the enclave from (1, 2) to watch the stack.
    g2 = [r[:] for r in g]
    for popped, pushed, stack in trace_fill(g2, 1, 2, 2):
        print(" pop", popped, "push", pushed, "stack", stack)
    print("islands (4-conn):", count_islands(grid))        # → 2
    print("islands (8-conn):", count_islands(grid, k=8))   # → 1
    flipped = [[1 - x for x in r] for r in grid]
    print("closed islands, 0 = land:", closed_islands(flipped))  # → 1

    # Recursion limit: a snake of frames, one per cell.
    print("recursion limit:", sys.getrecursionlimit())
    for side in (31, 32):
        g = [[1] * side for _ in range(side)]
        try:
            flood_fill_recursive(g, 0, 0, 0)
            print(f"recursive {side}x{side} ({side * side} cells): ok")
        except RecursionError:
            print(f"recursive {side}x{side} ({side * side} cells):"
                  " RecursionError")
    g = [[1] * 1000 for _ in range(1000)]
    flood_fill(g, 0, 0, 0)
    print("iterative 1000x1000 filled:", sum(map(sum, g)) == 0)

    # Minesweeper (LeetCode example 1), click (3, 0).
    def board():
        return [list("EEEEE"), list("EEMEE"), list("EEEEE"),
                list("EEEEE")]
    b, p_push = minesweeper(board(), (3, 0), mark_on_push=True)
    _, p_pop = minesweeper(board(), (3, 0), mark_on_push=False)
    for row in b:
        print(" ".join(row))
    print("pushes: mark on push", p_push, "| mark on pop", p_pop)
