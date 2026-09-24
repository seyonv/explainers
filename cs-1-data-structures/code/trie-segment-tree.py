"""Tries and segment trees.

Source: ljeng/cheat-sheet, coding-algorithms/data-structures.md, Trees.
The headings Word Search, Palindrome Pairs and kth Smallest in
Lexicographical Order are empty there, so this file is supplemented:
a trie with Word Search II, K-th lexicographical number by counting
the implicit trie, and a range-sum segment tree with point updates.
"""


# ---------- trie ----------

class Trie:
    def __init__(self):
        self.root = {}               # node = dict: char -> child node

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = word             # "$" marks the end of a word

    def search(self, word, prefix=False):
        node = self.root
        for ch in word:
            if ch not in node:
                return False
            node = node[ch]
        return prefix or "$" in node


def count_nodes(node):
    return 1 + sum(count_nodes(c) for k, c in node.items() if k != "$")


def find_words(board, words):
    """Word Search II: one DFS per cell, guided by the trie."""
    trie = Trie()
    for w in words:
        trie.insert(w)
    rows, cols = len(board), len(board[0])
    found, calls = [], [0]

    def neighbours(r, c):
        for i, j in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= i < rows and 0 <= j < cols:
                yield i, j

    def dfs(r, c, node):
        calls[0] += 1
        ch = board[r][c]
        if ch not in node:
            return                   # no word has this prefix: prune
        nxt = node[ch]
        if "$" in nxt:
            found.append(nxt.pop("$"))   # pop: report each word once
        board[r][c] = "#"            # mark: used on this path
        for i, j in neighbours(r, c):
            if board[i][j] != "#":
                dfs(i, j, nxt)
        board[r][c] = ch

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie.root)
    return sorted(found), calls[0]


def one_word_calls(board, word):
    """Plain Word Search for one word; returns (found, dfs calls)."""
    rows, cols = len(board), len(board[0])
    calls = [0]

    def dfs(r, c, k):
        calls[0] += 1
        if board[r][c] != word[k]:
            return False
        if k == len(word) - 1:
            return True
        ch, board[r][c] = board[r][c], "#"
        ok = False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            i, j = r + dr, c + dc
            if 0 <= i < rows and 0 <= j < cols and board[i][j] != "#":
                if dfs(i, j, k + 1):
                    ok = True
                    break
        board[r][c] = ch
        return ok

    hit = any(dfs(r, c, 0) for r in range(rows) for c in range(cols))
    return hit, calls[0]


# ---------- K-th lexicographical number: the implicit trie ----------

def steps(prefix, n):
    """How many of 1..n start with `prefix` (its subtree size)."""
    total, lo, hi = 0, prefix, prefix + 1
    while lo <= n:
        total += min(n + 1, hi) - lo     # one level of the subtree
        lo, hi = lo * 10, hi * 10
    return total


def kth_lex(n, k, log=None):
    cur, k = 1, k - 1                # stand on 1; k moves left to make
    while k > 0:
        s = steps(cur, n)
        if s <= k:                   # answer is not under cur: skip it
            if log is not None:
                log.append((cur, s, k, "skip", cur + 1))
            cur, k = cur + 1, k - s
        else:                        # answer is under cur: go down
            if log is not None:
                log.append((cur, s, k, "down", cur * 10))
            cur, k = cur * 10, k - 1
    return cur


# ---------- segment tree: range sum, point update ----------

class SegTree:
    def __init__(self, a):
        self.n = len(a)
        self.t = [0] * (4 * self.n)  # node i has children 2i, 2i+1
        self._build(1, 0, self.n - 1, a)

    def _build(self, i, lo, hi, a):
        if lo == hi:
            self.t[i] = a[lo]
            return
        mid = (lo + hi) // 2
        self._build(2 * i, lo, mid, a)
        self._build(2 * i + 1, mid + 1, hi, a)
        self.t[i] = self.t[2 * i] + self.t[2 * i + 1]

    def update(self, pos, val, i=1, lo=0, hi=None):
        hi = self.n - 1 if hi is None else hi
        if lo == hi:
            self.t[i] = val
            return
        mid = (lo + hi) // 2
        if pos <= mid:
            self.update(pos, val, 2 * i, lo, mid)
        else:
            self.update(pos, val, 2 * i + 1, mid + 1, hi)
        self.t[i] = self.t[2 * i] + self.t[2 * i + 1]

    def query(self, l, r, i=1, lo=0, hi=None, used=None):
        hi = self.n - 1 if hi is None else hi
        if r < lo or hi < l:
            return 0                 # disjoint: contributes nothing
        if l <= lo and hi <= r:
            if used is not None:
                used.append((lo, hi, self.t[i]))
            return self.t[i]         # fully inside: take the stored sum
        mid = (lo + hi) // 2
        return (self.query(l, r, 2 * i, lo, mid, used)
                + self.query(l, r, 2 * i + 1, mid + 1, hi, used))


def nodes(tree, i=1, lo=0, hi=None, depth=0, out=None):
    """List (depth, lo, hi, sum) of every node, for drawing."""
    hi = tree.n - 1 if hi is None else hi
    out = [] if out is None else out
    out.append((depth, lo, hi, tree.t[i]))
    if lo < hi:
        mid = (lo + hi) // 2
        nodes(tree, 2 * i, lo, mid, depth + 1, out)
        nodes(tree, 2 * i + 1, mid + 1, hi, depth + 1, out)
    return out


if __name__ == "__main__":
    import random

    board = [list("oaan"), list("etae"), list("ihkr"), list("iflv")]
    words = ["oath", "pea", "eat", "rain"]
    t = Trie()
    for w in words:
        t.insert(w)
    print("trie nodes incl. root:", count_nodes(t.root))
    print("search oath/oat/oat-prefix:", t.search("oath"),
          t.search("oat"), t.search("oat", prefix=True))
    found, calls = find_words([row[:] for row in board], words)
    print("word search II:", found, "dfs calls with trie:", calls)
    per_word = [one_word_calls([r[:] for r in board], w) for w in words]
    print("one search per word:", per_word,
          "total calls:", sum(c for _, c in per_word))

    rng = random.Random(0)
    big = [[rng.choice("abcde") for _ in range(8)] for _ in range(8)]
    ws = list({"".join(rng.choice("abcde")
                       for _ in range(rng.randint(3, 8)))
               for _ in range(500)})
    found, calls = find_words([r[:] for r in big], ws)
    each = sum(one_word_calls([r[:] for r in big], w)[1] for w in ws)
    print(f"random 8x8 board, {len(ws)} words: {len(found)} found;"
          f" dfs calls trie {calls:,} vs one search per word {each:,}")

    for n, k in ((13, 2), (13, 7)):
        log = []
        print(f"kth_lex({n}, {k}) =", kth_lex(n, k, log))
        for row in log:
            print("   cur=%d steps=%d k=%d %s -> %d" % row)
    print("order for n=13:", sorted(range(1, 14), key=str))
    print("steps(1, 13) = (2-1) + (14-10) =", steps(1, 13))

    a = [5, 2, 9, 1, 5, 6]
    st = SegTree(a)
    for d, lo, hi, s in nodes(st):
        print("  " * d + f"[{lo},{hi}]={s}")
    used = []
    print("sum a[1..4] =", st.query(1, 4, used=used), "from", used)
    st.update(3, 4)
    used = []
    print("after a[3]=4: sum a[1..4] =", st.query(1, 4, used=used),
          "from", used, "root", st.t[1])

    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(1, 300)
        k = rng.randint(1, n)
        assert kth_lex(n, k) == sorted(range(1, n + 1), key=str)[k - 1]
        arr = [rng.randint(-9, 9) for _ in range(rng.randint(1, 40))]
        tr = SegTree(arr)
        for _ in range(20):
            p = rng.randrange(len(arr))
            arr[p] = rng.randint(-9, 9)
            tr.update(p, arr[p])
            l = rng.randrange(len(arr))
            r = rng.randrange(l, len(arr))
            assert tr.query(l, r) == sum(arr[l:r + 1])
    print("300 random checks vs brute force: ok")
