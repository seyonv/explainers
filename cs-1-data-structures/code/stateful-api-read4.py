"""Read n characters given read4, called multiple times.

Source: ljeng/cheat-sheet, coding-algorithms/coding.md, APIs >
Read n Characters Given read4 II - Call Multiple Times. The source
is C++ with the leftover buffer and its two cursors as globals;
here they live on a Reader object, one per file.
"""
import random


def make_read4(text):
    """read4 over text, with its own file pointer (like FILE *fp)."""
    pos = 0

    def read4(buf4):
        nonlocal pos
        chunk = text[pos:pos + 4]
        pos += len(chunk)
        buf4[:len(chunk)] = chunk
        read4.calls += 1
        return len(chunk)

    read4.calls = 0
    return read4


class Reader:
    def __init__(self, read4):
        self.read4 = read4
        self.buf4 = [""] * 4   # chars read4 gave us, not yet returned
        self.i = 0             # next unread slot in buf4
        self.m = 0             # how many slots the last read4 filled

    def read(self, buf, n):
        j = 0
        while j < n:
            if self.i == self.m:            # leftover used up: refill
                self.m, self.i = self.read4(self.buf4), 0
                if self.m == 0:             # end of file
                    break
            take = min(n - j, self.m - self.i)
            buf[j:j + take] = self.buf4[self.i:self.i + take]
            j += take
            self.i += take
        return j

    def leftover(self):
        return "".join(self.buf4[self.i:self.m])


def read_stateless(read4, buf, n):
    """The one-shot (read4 I) answer: drops what it doesn't return."""
    buf4, j = [""] * 4, 0
    while j < n:
        m = read4(buf4)
        if m == 0:
            break
        take = min(n - j, m)
        buf[j:j + take] = buf4[:take]
        j += take
    return j


# The source's C++, transliterated: one buffer shared by everyone.
g_buf4, g_i, g_m = [""] * 4, 0, 0


def read_global(read4, buf, n):
    global g_i, g_m
    j = 0
    while j < n:
        if not g_i:
            g_m = read4(g_buf4)
        if not g_m:
            break
        while j < n and g_i < g_m:
            buf[j] = g_buf4[g_i]
            j += 1
            g_i += 1
        if g_i >= g_m:
            g_i = 0
    return j


def trace(text, calls):
    read4 = make_read4(text)
    r, buf = Reader(read4), [""] * 16
    for n in calls:
        before, c0 = r.leftover(), read4.calls
        k = r.read(buf, n)
        print(f"read({n}): had {before!r:6} read4 x{read4.calls - c0}"
              f"  -> {k} {''.join(buf[:k])!r:9}"
              f" left {r.leftover()!r}")
    print("read4 calls in total:", read4.calls)


def check_random(trials=1000, seed=0):
    """Reader vs slicing the string, on random texts and call sizes."""
    rng, worst = random.Random(seed), 0
    for _ in range(trials):
        size = rng.randint(0, 30)
        text = "".join(rng.choice("xyz") for _ in range(size))
        read4 = make_read4(text)
        r, pos, buf = Reader(read4), 0, [""] * 40
        for _ in range(rng.randint(1, 8)):
            n = rng.randint(0, 12)
            c0, k = read4.calls, r.read(buf, n)
            want = text[pos:pos + n]
            assert "".join(buf[:k]) == want, (text, n)
            pos += len(want)
            worst = max(worst, read4.calls - c0 - (n + 3) // 4)
    return trials, worst


if __name__ == "__main__":
    text = "abcdefghij"
    print("Reader (state on the object):")
    trace(text, [1, 6, 5])

    print("\nStateless, no leftover kept:")
    read4, buf = make_read4(text), [""] * 16
    for n in [1, 6, 5]:
        k = read_stateless(read4, buf, n)
        print(f"read({n}) -> {k} {''.join(buf[:k])!r}")

    print("\nGlobal buffer, two files:")
    a, b, buf = make_read4(text), make_read4("wxyz"), [""] * 16
    k = read_global(a, buf, 1)
    print("A read(1) ->", repr("".join(buf[:k])))
    k = read_global(b, buf, 2)
    print("B read(2) ->", repr("".join(buf[:k])), "(want 'wx')")

    trials, worst = check_random()
    print(f"\n{trials} random cases match slicing;"
          f" read4 calls <= ceil(n/4) + {worst}")
