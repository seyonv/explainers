"""Bigtable's data model as a toy: a sorted map keyed by
(row, family:qualifier, timestamp), with per-family version limits
and per-family read access. All rows and values are illustrative."""


def row_key(url):         # www.example.com/x -> com.example.www/x
    host, _, path = url.partition("/")
    return ".".join(reversed(host.split("."))) + "/" + path


def runs(keys, is_site):
    """How many separate blocks the site's keys form once sorted."""
    flags = [is_site(k) for k in sorted(keys)]
    return sum(f and (i == 0 or not flags[i - 1])
               for i, f in enumerate(flags))


class Table:
    def __init__(self, families):   # family -> versions to keep
        self.families = families
        self.cells = {}             # (row, column) -> [(ts, value)]

    def put(self, row, column, ts, value):
        family = column.split(":")[0]
        if family not in self.families:  # families are declared
            raise KeyError(f"no column family {family!r}")
        versions = self.cells.setdefault((row, column), [])
        versions.append((ts, value))
        versions.sort(reverse=True)            # newest first
        del versions[self.families[family]:]   # per-family GC

    def read(self, row, allowed=None):     # newest of each column
        """Filtering by family stands in for family-level ACLs."""
        return {col: v[0][1]
                for (r, col), v in sorted(self.cells.items())
                if r == row and (allowed is None
                                 or col.split(":")[0] in allowed)}

    def scan(self, prefix):
        """Range scan: rows are sorted, so a prefix is one run."""
        rows = {r for r, _ in self.cells if r.startswith(prefix)}
        return sorted(rows)


def build():
    t = Table({"contents": 2, "anchor": 1, "language": 1})
    home = row_key("www.example.com/")
    t.put(home, "contents:", 10, "<html>v1")
    t.put(home, "contents:", 20, "<html>v2")
    t.put(home, "contents:", 30, "<html>v3")   # v1 is dropped
    t.put(home, "anchor:www.acme.org", 12, "Example Co")
    t.put(home, "anchor:blog.zeta.io", 18, "their homepage")
    t.put(home, "language:", 10, "en")
    news = row_key("news.example.com/today")
    t.put(news, "contents:", 25, "<html>today")
    t.put(news, "anchor:www.example.com", 30, "Today's news")
    t.put(row_key("www.acme.org/"), "contents:", 15, "<html>acme")
    return t


if __name__ == "__main__":
    urls = ["www.example.com/", "news.acme.org/", "mail.example.com/",
            "www.zeta.io/", "news.example.com/today", "www.acme.org/"]
    hosts = [u.split("/")[0] for u in urls]
    print("sorted by hostname: ", sorted(urls))
    print("  example.com blocks:",
          runs(hosts, lambda h: h.endswith("example.com")))
    keys = [row_key(u) for u in urls]
    print("sorted by row key:  ", sorted(keys))
    print("  example.com blocks:",
          runs(keys, lambda k: k.startswith("com.example.")))

    t = build()
    print("\nstored order (row, column, versions newest first):")
    for (row, col), v in sorted(t.cells.items()):
        print(f"  {row:24} {col:24} {v}")
    home = row_key("www.example.com/")
    print("\nread(home):", t.read(home))
    print("read(home, allowed={anchor, language}):",
          t.read(home, allowed={"anchor", "language"}))
    print("scan('com.example.'):", t.scan("com.example."))
    try:
        t.put(home, "pagerank:", 40, "0.9")
    except KeyError as e:
        print("put pagerank: ->", e)
