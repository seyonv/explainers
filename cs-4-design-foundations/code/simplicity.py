"""Simple is not easy: one TinyURL feature written two ways.

A: a stateful class whose shorten() braids id allocation, encoding,
   storage and call order together.
B: a pure encode function plus explicit state passed in and out.
Then count, with ast, what each method reads and writes on self.
"""
import ast
import inspect
import string
import textwrap

ALPHABET = (string.digits + string.ascii_lowercase
            + string.ascii_uppercase)


# ---- A: easy to write, complex (braided) ----
class Shortener:
    def __init__(self):
        self.next_id = 0
        self.links = {}
        self.log = []

    def shorten(self, url):
        self.next_id += 1              # id allocation
        n, code = self.next_id, ""
        while n:                       # encoding, inline
            n, r = divmod(n, 62)
            code = ALPHABET[r] + code
        self.links[code] = url         # storage
        self.log.append(code)          # auditing
        return code


# ---- B: simple, each piece does one thing ----
def base62(n):
    code = ""
    while n:
        n, r = divmod(n, 62)
        code = ALPHABET[r] + code
    return code or "0"


def shorten(state, url):
    n = state["next_id"] + 1
    code = base62(n)
    links = {**state["links"], code: url}
    return {"next_id": n, "links": links}, code


def is_self(node):
    return (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self")


def self_access(cls):
    """Per method: which self.x it reads and which it writes."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    out = {}
    for fn in [n for n in tree.body[0].body
               if isinstance(n, ast.FunctionDef)]:
        reads, writes = set(), set()
        for node in ast.walk(fn):
            if is_self(node):
                (writes if isinstance(node.ctx, ast.Store)
                 else reads).add(node.attr)
            # self.x[k] = v and self.x.append(v) also change self.x
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.ctx, ast.Store)
                    and is_self(node.value)):
                writes.add(node.value.attr)
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and is_self(node.func.value)):
                writes.add(node.func.value.attr)
        out[fn.name] = (sorted(reads), sorted(writes))
    return out


if __name__ == "__main__":
    # To get the code for id 125 from A you must replay 125 calls.
    a = Shortener()
    for i in range(125):
        code = a.shorten(f"https://example.com/{i}")
    print("A: 125 calls to shorten() ->", code)
    # B: the logic is testable in one call, no history.
    print("B: base62(125) ->", base62(125))
    print("   125 = 2*62 + 1 ->", ALPHABET[2] + ALPHABET[1])
    s = {"next_id": 124, "links": {}}
    s2, c = shorten(s, "https://example.com/124")
    print("B: shorten from next_id=124 ->", c,
          "| old state untouched:", s["next_id"])
    for name, (r, w) in self_access(Shortener).items():
        print(f"A.{name}: reads self.{r} writes self.{w}")
