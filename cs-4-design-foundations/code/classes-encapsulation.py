"""Classes, objects and encapsulation: the source guide's Point.java
rewritten in Python.

1. The idiomatic version: a frozen dataclass. relative_to returns a
   NEW Point, so p1 keeps its own state.
2. Python has no `private`. Each level of hiding, and what an
   outside write actually does (real exception text).
3. Why immutability matters: the aliasing bug a mutable Point allows.
"""
import dataclasses
from dataclasses import dataclass


# ---- 1. the Point from the source, idiomatic Python ----
@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def relative_to(self, dx, dy):
        return Point(self.x + dx, self.y + dy)   # new object

    def __str__(self):
        return f"({self.x}, {self.y})"          # Java toString()


# ---- 2. the hiding ladder ----
class Public:
    def __init__(self):
        self.x = 5                # anyone may read and write


class Underscore:
    def __init__(self):
        self._x = 5               # "internal": a convention only


class Mangled:
    def __init__(self):
        self.__x = 5              # stored as _Mangled__x


class ReadOnly:
    def __init__(self):
        self._x = 5

    @property
    def x(self):                  # getter, no setter
        return self._x


def try_write(obj, name, value=0):
    try:
        setattr(obj, name, value)
        return "allowed"
    except (AttributeError, dataclasses.FrozenInstanceError) as e:
        return f"{type(e).__name__}: {e}"


# ---- 3. a mutable Point and the aliasing bug ----
class MutablePoint:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return f"MutablePoint({self.x}, {self.y})"


if __name__ == "__main__":
    p1 = Point(5, 10)
    p2 = p1.relative_to(-5, 5)
    print(p2)                      # (0, 15), as in Point.java
    print(repr(p2))                # generated __repr__
    print(p1, p1 is p2)            # p1 unchanged
    print(p2 == Point(0, 15))      # compares fields
    print()

    print("Public     p.x = 0 ->", try_write(Public(), "x"))
    print("Underscore p._x = 0 ->", try_write(Underscore(), "_x"))
    m = Mangled()
    print("Mangled    vars ->", vars(m))
    try:
        m.__x
    except AttributeError as e:
        print("Mangled    m.__x ->", f"AttributeError: {e}")
    print("Mangled    m._Mangled__x ->", m._Mangled__x)
    print("ReadOnly   p.x = 0 ->", try_write(ReadOnly(), "x"))
    print("Frozen     p.x = 0 ->", try_write(Point(5, 10), "x"))
    f = Point(5, 10)
    object.__setattr__(f, "x", 99)  # the documented back door
    print("Frozen via object.__setattr__ ->", f)
    print()

    a = MutablePoint(5, 10)
    b = a                          # same object, two names
    b.x = 0
    print("mutable:", a, "<- a changed through b")
    c = Point(5, 10)
    d = c.relative_to(-5, 0)       # the only way to "move" it
    print("frozen: ", c, d)
