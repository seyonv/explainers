"""APIs as layers of abstraction: one TinyURL link, four layers.

Source: ljeng/cheat-sheet, large-scale-design/system-design/
api-discussions.md. Each layer is represented in terms of the one
below it: app object -> general data model (JSON, a table row) ->
bytes -> pages on disk. The API is the contract on top that hides
all of it. The link itself (code, URL, timestamp) is illustrative.

The second half is a tiny v1 API over the same table showing two of
the source's API concerns: an idempotent create and cursor paging.
"""
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Link:                      # layer 1: the app's own object
    code: str
    url: str
    created: int


def layers(link, path):
    obj = sys.getsizeof(link) + sum(
        sys.getsizeof(v) for v in asdict(link).values())
    text = json.dumps(asdict(link), ensure_ascii=False,
                      separators=(",", ":"))  # layer 2: JSON
    raw = text.encode("utf-8")                # layer 3: bytes
    db = sqlite3.connect(path)                # layer 4: disk
    db.execute("CREATE TABLE links(code TEXT PRIMARY KEY,"
               " url TEXT, created INT)")
    db.execute("INSERT INTO links VALUES (?,?,?)",
               (link.code, link.url, link.created))
    db.commit()
    (psize,) = db.execute("PRAGMA page_size").fetchone()
    (pages,) = db.execute("PRAGMA page_count").fetchone()
    db.close()
    return obj, text, raw, psize, pages, os.path.getsize(path)


# --- the API on top: v1, idempotent create, cursor pagination ---
db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE links(id INTEGER PRIMARY KEY,"
           " url TEXT, idem TEXT UNIQUE)")


def v1_create(url, idem_key):
    """POST /v1/links. A retry with the same key gets the same id."""
    row = db.execute("SELECT id FROM links WHERE idem = ?",
                     (idem_key,)).fetchone()
    if row:
        return {"id": row[0], "created": False}
    cur = db.execute("INSERT INTO links(url, idem) VALUES (?, ?)",
                     (url, idem_key))
    return {"id": cur.lastrowid, "created": True}


def v1_list(after=0, limit=2):
    """GET /v1/links?after=..&limit=.. (keyset, not OFFSET)."""
    rows = db.execute("SELECT id, url FROM links WHERE id > ?"
                      " ORDER BY id LIMIT ?", (after, limit)).fetchall()
    nxt = rows[-1][0] if len(rows) == limit else None
    return {"items": rows, "next": nxt}


if __name__ == "__main__":
    link = Link("a1b2", "https://example.com/café/menu", 1790000000)
    with tempfile.TemporaryDirectory() as d:
        obj, text, raw, psize, pages, fsize = layers(
            link, os.path.join(d, "links.db"))
    print("1 object  ", link)
    print("  bytes in RAM (object + its 3 values):", obj)
    print("2 JSON    ", text)
    print("  characters:", len(text))
    print("3 UTF-8   ", raw[:24].hex(" "), "...")
    print("  bytes:", len(raw), f"(= {len(text)} chars + 1, é is 2)")
    print("  bits:", len(raw) * 8)
    print("4 SQLite   page_size", psize, "x page_count", pages,
          "=", psize * pages, "| file", fsize)
    ascii_json = json.dumps(asdict(link), separators=(",", ":"))
    print("  json default ensure_ascii=True:", len(ascii_json),
          "chars (é -> \\u00e9)")

    print(v1_create("https://example.com/a", "k1"))
    print(v1_create("https://example.com/a", "k1"))   # a retry
    for i, u in enumerate("bcd"):
        v1_create(f"https://example.com/{u}", f"k{i + 2}")
    page = v1_list()
    print(page)
    print(v1_list(after=page["next"]))
