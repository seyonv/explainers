"""Inheritance and polymorphism: the source guide's Shape hierarchy.

The source is Java (Shape, Rectangle, Ellipse, paintShapes). Here it
is Python with abc.ABC. Circle is added because the source says
"Ellipse could be further specialized into the Circle subclass".
draw() appends a line to a list instead of painting on a Graphics.
"""
import math
from abc import ABC, abstractmethod


class Shape(ABC):
    def __init__(self, center):
        self.center = center          # shared by every subclass

    @abstractmethod
    def bounds(self): ...             # every subclass must supply

    @abstractmethod
    def draw(self, canvas): ...


class Rectangle(Shape):
    def __init__(self, center, w, h):
        super().__init__(center)
        self.w, self.h = w, h

    def bounds(self):
        return self                   # a rectangle is its own box

    def draw(self, canvas):
        canvas.append(f"rect {self.w}x{self.h} at {self.center}")

    def __repr__(self):
        return f"Rectangle({self.center}, w={self.w}, h={self.h})"


class Ellipse(Shape):
    def __init__(self, center, a, b):
        super().__init__(center)
        self.a, self.b = a, b         # semi-axes

    def bounds(self):
        return Rectangle(self.center, 2 * self.a, 2 * self.b)

    def draw(self, canvas):
        canvas.append(f"ellipse a={self.a} b={self.b} at {self.center}")


class Circle(Ellipse):
    def __init__(self, center, r):
        super().__init__(center, r, r)    # an ellipse with a = b

    def draw(self, canvas):               # override; bounds inherited
        canvas.append(f"circle r={self.a} at {self.center}")


def paint_shapes(canvas, shapes):
    for s in shapes:
        s.draw(canvas)    # which draw? looked up on type(s) at run time


def owner(cls, name):
    """First class on cls.__mro__ that defines name: the one called."""
    return next(c.__name__ for c in cls.__mro__ if name in c.__dict__)


if __name__ == "__main__":
    e = Ellipse((0, 0), 3, 2)
    box = e.bounds()
    print("Ellipse((0, 0), 3, 2).bounds() ->", box)
    area_e = math.pi * e.a * e.b
    area_box = box.w * box.h
    print(f"areas: ellipse pi*3*2 = {area_e:.2f}, box 6*4 = {area_box},"
          f" ratio = {area_e / area_box:.3f} (pi/4 = {math.pi/4:.3f})")

    shapes = [Rectangle((0, 0), 6, 4), e, Circle((1, 1), 2)]
    canvas = []
    paint_shapes(canvas, shapes)
    print("\npaint_shapes:")
    for s, line in zip(shapes, canvas):
        cls = type(s)
        print(f"  {cls.__name__:<9} draw from {owner(cls, 'draw'):<9}"
              f" bounds from {owner(cls, 'bounds'):<9}"
              f" -> {line} | {s.bounds()}")

    print("\nCircle.__mro__:",
          " -> ".join(c.__name__ for c in Circle.__mro__))
    print("isinstance(Circle(...), Shape):",
          isinstance(shapes[2], Shape))

    try:
        Shape((0, 0))
    except TypeError as err:
        print("\nShape((0, 0)) ->", "TypeError:", err)

    class Triangle(Shape):            # forgot draw()
        def bounds(self):
            return Rectangle(self.center, 1, 1)
    try:
        Triangle((0, 0))
    except TypeError as err:
        print("Triangle((0, 0)) ->", "TypeError:", err)
