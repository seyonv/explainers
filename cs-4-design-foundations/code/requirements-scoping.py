"""Functional vs non-functional requirements, on TinyURL.

Source: ljeng/cheat-sheet, large-scale-design/system-design/
feature-sets.md. Functional = what the system does (shorten,
redirect, track clicks). Non-functional = how well (turned into
numbers below). The source's advice "store raw data, not just
summaries" is shown by answering a feature nobody asked for yet.

The scale inputs (100M new links a month, 100:1 reads to writes,
500-byte records, 5 years) are illustrative; the source gives none.
The click rows in the demo are illustrative too.
"""
import sqlite3

NEW_PER_MONTH = 100_000_000      # illustrative
READS_PER_WRITE = 100            # illustrative
RECORD_BYTES = 500               # illustrative
YEARS = 5                        # illustrative
SECONDS_PER_MONTH = 30 * 86_400  # 30-day month


def sizing():
    """Non-functional requirements as numbers."""
    writes = NEW_PER_MONTH / SECONDS_PER_MONTH
    reads = writes * READS_PER_WRITE
    records = NEW_PER_MONTH * 12 * YEARS
    return writes, reads, records, records * RECORD_BYTES


db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE links(code TEXT PRIMARY KEY, url TEXT)")
db.execute("CREATE TABLE clicks(code TEXT, ts INT, country TEXT)")
counter = {}                     # the summary-only alternative


def shorten(code, url):          # functional 1: shorten
    db.execute("INSERT INTO links VALUES (?, ?)", (code, url))


def redirect(code, ts, country):  # functional 2 and 3
    (url,) = db.execute(
        "SELECT url FROM links WHERE code = ?", (code,)).fetchone()
    db.execute("INSERT INTO clicks VALUES (?, ?, ?)",
               (code, ts, country))          # raw event, kept
    counter[code] = counter.get(code, 0) + 1  # summary only
    return url


def clicks_by_country(code):     # a later feature, from raw rows
    return db.execute(
        "SELECT country, COUNT(*) FROM clicks WHERE code = ? "
        "GROUP BY country ORDER BY country", (code,)).fetchall()


if __name__ == "__main__":
    w, r, n, b = sizing()
    print(f"writes/s  {NEW_PER_MONTH:,} / {SECONDS_PER_MONTH:,}"
          f" = {w:.1f}")
    print(f"reads/s   {w:.2f} x {READS_PER_WRITE} = {r:,.0f}")
    print(f"records   {NEW_PER_MONTH:,} x 12 x {YEARS} = {n:,}")
    print(f"storage   {n:,} x {RECORD_BYTES} B = {b / 1e12:.0f} TB")
    print(f"raw click rows/month {NEW_PER_MONTH:,} x"
          f" {READS_PER_WRITE} = {NEW_PER_MONTH * READS_PER_WRITE:,}")

    shorten("a1b2", "https://example.com/a/very/long/path")
    for ts, cc in [(1, "US"), (2, "CA"), (3, "US"), (4, "IN"),
                   (5, "US")]:
        url = redirect("a1b2", ts, cc)
    print("redirect ->", url)
    print("counter   ", counter)
    print("by country", clicks_by_country("a1b2"))
    try:
        shorten("a1b2", "https://example.org/other")
    except sqlite3.IntegrityError as e:
        print("duplicate ->", e)
