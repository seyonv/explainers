"""Interfaces vs abstract classes, in Python.

The cheat-sheet's StatusCallback (C++ and Java) and XMLReader /
XMLReaderImpl (Java) examples, rewritten with typing.Protocol for the
interface and abc.ABC for the abstract class. Run: python3 (3.10+).
"""
import io
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


# 1. Interface: a can-do contract, matched by shape (structural).
@runtime_checkable
class StatusCallback(Protocol):
    @abstractmethod
    def update_status(self, old: int, new: int) -> None: ...


class Widget:                          # "SomeOtherClass" in the source
    def __init__(self, name):
        self.name = name


class ProgressBar(Widget):             # never names StatusCallback
    def update_status(self, old, new):
        if new > old:
            print(f"  {self.name}: {old} -> {new}")


class MyClass(Widget, StatusCallback):  # "extends ... implements"
    pass                                # forgot update_status


def notify(cb: StatusCallback, old: int, new: int) -> None:
    cb.update_status(old, new)


# 2. Interface, plus an abstract class that implements it by default.
@runtime_checkable
class XMLReader(Protocol):
    def from_string(self, s: str) -> ET.Element: ...
    def from_reader(self, f: io.TextIOBase) -> ET.Element: ...


class XMLReaderBase(ABC):              # XMLReaderImpl in the source
    def from_string(self, s):          # written once, for everyone
        return self.from_reader(io.StringIO(s))

    @abstractmethod
    def from_reader(self, f): ...      # the core each subclass writes


class EtreeReader(XMLReaderBase):
    def from_reader(self, f):
        return ET.parse(f).getroot()


class Broken(XMLReaderBase):           # forgot from_reader
    pass


class Liar:                            # right name, wrong signature
    def update_status(self):
        pass


def try_new(cls, *args):
    try:
        cls(*args)
        print(f"  {cls.__name__}(): ok")
    except TypeError as e:
        print(f"  {cls.__name__}(): TypeError: {e}")


if __name__ == "__main__":
    print("interface, structural:")
    notify(ProgressBar("upload"), 3, 7)
    print("  isinstance(ProgressBar, StatusCallback):",
          isinstance(ProgressBar("p"), StatusCallback))
    print("  isinstance(Widget, StatusCallback):",
          isinstance(Widget("w"), StatusCallback))

    print("abstract class, default implementation:")
    root = EtreeReader().from_string("<feed><item/><item/></feed>")
    print(f"  from_string -> <{root.tag}> with {len(root)} children")
    print("  isinstance(EtreeReader(), XMLReader):",
          isinstance(EtreeReader(), XMLReader))

    print("instantiation:")
    for cls, args in [(XMLReaderBase, ()), (Broken, ()),
                      (StatusCallback, ()), (MyClass, ("x",)),
                      (EtreeReader, ())]:
        try_new(cls, *args)

    print("multiple inheritance, MRO of MyClass:")
    print("  " + " -> ".join(c.__name__ for c in MyClass.__mro__))

    print("runtime check looks at names only:")
    print("  isinstance(Liar(), StatusCallback):",
          isinstance(Liar(), StatusCallback))
