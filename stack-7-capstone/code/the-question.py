"""Re-check novelty: newest arXiv papers matching a few queries.

Prints date, arXiv id and title, newest first, and marks ids that
are already in the series' verified novelty check. Stdlib only.
"""
import re
import sys
import urllib.parse
import urllib.request

API = "https://export.arxiv.org/api/query?"
KNOWN = {  # every id in the series' novelty check (2026-10-01)
    "2503.03750", "2503.11926", "2508.17511", "2510.11977",
    "2510.20270", "2511.00197", "2511.06626", "2511.18397",
    "2511.21654", "2512.08093", "2601.03267", "2601.04886",
    "2603.00822", "2603.04582", "2603.10060", "2603.23064",
    "2603.24631", "2604.11072", "2604.20779", "2605.02964",
    "2605.06527", "2605.08580", "2605.12673", "2605.17998",
    "2605.20744", "2605.21384", "2605.23950", "2605.24279",
    "2605.29442", "2605.30777", "2606.07682", "2606.08529",
    "2606.09863", "2606.11688", "2606.14589", "2606.22528",
    "2606.26300", "2606.28430", "2607.02294", "2607.05378",
    "2607.13071", "2607.20972", "2607.22585", "2607.23929",
    "2607.25152", "2607.27250", "2608.07429", "2608.08654",
    "2608.22103", "2608.22752", "2608.23623", "2608.29460",
    "2609.05510", "2609.15494", "2609.17930", "2609.20211",
    "2609.20812", "2609.26779", "2609.29921", "2609.35659",
    "2609.35732",
}
QUERIES = [
    'abs:compaction AND abs:"coding agent"',
    'abs:"false success" AND abs:agent',
    'abs:"context compaction"',
]


def search(query, n=6):
    url = API + urllib.parse.urlencode({
        "search_query": query, "max_results": n,
        "sortBy": "submittedDate", "sortOrder": "descending"})
    with urllib.request.urlopen(url, timeout=30) as r:
        feed = r.read().decode()
    for e in re.findall(r"<entry>(.*?)</entry>", feed, re.S):
        aid = re.search(r"/abs/([^<v]+)", e).group(1)
        title = " ".join(re.search(
            r"<title>(.*?)</title>", e, re.S).group(1).split())
        date = re.search(r"<published>(.{10})", e).group(1)
        yield date, aid, title


if __name__ == "__main__":
    for q in sys.argv[1:] or QUERIES:
        print(f"== {q}")
        for date, aid, title in search(q):
            mark = "known" if aid in KNOWN else "NEW? "
            print(f"  {date} {aid:11} {mark} {title[:46]}")
