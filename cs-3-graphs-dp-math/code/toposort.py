"""Topological sort and cycle detection.

Alien Dictionary (the source guide solves it with graphlib), Course
Schedule I and II, solved here with Kahn's algorithm in plain Python,
plus a three-colour DFS for comparison. Runs under python3 (3.10).
"""
from collections import deque
from itertools import pairwise, permutations
import graphlib
import random


def alien_order(words):
    succ = {c: [] for w in words for c in w}   # letter -> later letters
    indeg = dict.fromkeys(succ, 0)
    for a, b in pairwise(words):
        for x, y in zip(a, b):
            if x != y:                   # first difference = one edge
                succ[x].append(y)
                indeg[y] += 1
                break
        else:
            if len(a) > len(b):          # "abc" before "ab": invalid
                return ""
    queue = deque(c for c in indeg if indeg[c] == 0)
    order = []
    while queue:
        c = queue.popleft()
        order.append(c)
        for d in succ[c]:
            indeg[d] -= 1                # one prerequisite done
            if indeg[d] == 0:
                queue.append(d)
    return "".join(order) if len(order) == len(indeg) else ""


def kahn_trace(words):
    """Print in-degrees and queue after each pop (valid input)."""
    succ = {c: [] for w in words for c in w}
    indeg = dict.fromkeys(succ, 0)
    for _, _, x, y in edges(words):
        succ[x].append(y)
        indeg[y] += 1
    print(f"  start  in-degree {indeg}")
    queue = deque(c for c in indeg if indeg[c] == 0)
    order = []
    while queue:
        c = queue.popleft()
        order.append(c)
        for d in succ[c]:
            indeg[d] -= 1
            if indeg[d] == 0:
                queue.append(d)
        left = {k: v for k, v in indeg.items() if k not in order}
        print(f"  pop {c}  in-degree left {left}  "
              f"queue {list(queue)}  order {''.join(order)}")


def edges(words):
    """The (earlier, later) letter pairs read off adjacent words."""
    out = []
    for a, b in pairwise(words):
        for x, y in zip(a, b):
            if x != y:
                out.append((a, b, x, y))
                break
    return out


def find_order(num_courses, prerequisites):
    """Course Schedule II: [a, b] means take b before a."""
    succ = [[] for _ in range(num_courses)]
    indeg = [0] * num_courses
    for a, b in prerequisites:
        succ[b].append(a)
        indeg[a] += 1
    queue = deque(i for i in range(num_courses) if indeg[i] == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return order if len(order) == num_courses else []


def can_finish(num_courses, prerequisites):
    """Course Schedule I."""
    return len(find_order(num_courses, prerequisites)) == num_courses


def dfs_order(succ):
    """Three-colour DFS: None on a cycle, else a topological order."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(succ, WHITE)
    post = []
    for root in succ:
        if colour[root] != WHITE:
            continue
        colour[root] = GREY
        stack = [(root, iter(succ[root]))]
        while stack:
            u, it = stack[-1]
            v = next(it, None)
            if v is None:                # all children done
                colour[u] = BLACK
                post.append(u)
                stack.pop()
            elif colour[v] == GREY:      # back edge: a cycle
                return None
            elif colour[v] == WHITE:
                colour[v] = GREY
                stack.append((v, iter(succ[v])))
    return post[::-1]                    # reverse postorder


def source_alien_order(words):
    """The source guide's version, verbatim apart from layout."""
    from itertools import zip_longest
    graph = {letter: set() for word in words for letter in word}
    for word1, word2 in pairwise(words):
        for letter1, letter2 in zip_longest(word1, word2):
            if letter1 != letter2:
                if letter2 is None:
                    return ""
                if letter1 is not None:
                    graph[letter2].add(letter1)
                break
    try:
        ts = graphlib.TopologicalSorter(graph)
        return "".join(ts.static_order())
    except graphlib.CycleError:
        return ""


def is_valid(words, order):
    """Brute check: are words sorted under this alphabet?"""
    rank = {c: i for i, c in enumerate(order)}
    keys = [[rank[c] for c in w] for w in words]
    return keys == sorted(keys)


def any_valid(words):
    letters = sorted({c for w in words for c in w})
    return any(is_valid(words, p) for p in permutations(letters))


if __name__ == "__main__":
    words = ["wrt", "wrf", "er", "ett", "rftt"]
    print("edges from adjacent pairs:")
    for a, b, x, y in edges(words):
        print(f"  {a:>4} / {b:<4} -> {x} before {y}")
    print("Kahn trace:")
    kahn_trace(words)
    print(alien_order(words))                         # wertf
    print(repr(alien_order(["abc", "ab"])))           # ''
    print(repr(alien_order(["z", "x", "z"])))         # ''  (cycle)
    print(repr(alien_order(["z", "x"])))              # 'zx'

    print(find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]))
    print(can_finish(2, [[1, 0], [0, 1]]))            # False

    succ = {"w": ["e"], "e": ["r"], "r": ["t"], "t": ["f"], "f": []}
    print("dfs:", "".join(dfs_order(succ)))
    print("dfs on z<->x:", dfs_order({"z": ["x"], "x": ["z"]}))

    rng = random.Random(0)
    for _ in range(2000):
        ws = sorted("".join(rng.choice("abcd")
                            for _ in range(rng.randint(1, 3)))
                    for _ in range(rng.randint(1, 5)))
        if rng.random() < 0.5:
            rng.shuffle(ws)
        mine = alien_order(ws)
        assert (mine != "") == any_valid(ws), ws
        if mine:
            assert is_valid(ws, mine + "".join(
                c for c in "abcd" if c not in mine))
        assert (source_alien_order(ws) != "") == (mine != "")
    print("2000 random word lists: matches brute force and graphlib")
