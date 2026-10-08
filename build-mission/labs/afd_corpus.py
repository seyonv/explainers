"""Build the forecast-discussion corpus for the Build an LLM mission.

US National Weather Service offices write an Area Forecast Discussion (AFD) several times a day: a few
paragraphs of plain English about what the forecaster expects and why. This script downloads whole days of
them, from every office, from the Iowa Environmental Mesonet's text archive, cleans them, removes the text
that later updates copy forward, and writes nested training files plus fixed validation and test files.

  python3 afd_corpus.py --quick      # the first 12 training days, plus validation and test: the 1, 3 and 10 MB sizes
  python3 afd_corpus.py --full       # all 90 training days: every size up to 100 MB
  python3 afd_corpus.py --stats      # sizes, products and SHA-256 of what you have

Data: derived from NWS Area Forecast Discussions, which are public domain (weather.gov/disclaimer); archive
by the Iowa Environmental Mesonet, Iowa State University (mesonet.agron.iastate.edu). Not official forecasts.
Standard library only. Files go to ./data next to this script, or --data DIR.
"""
import argparse, hashlib, http.client, json, random, re, sys, time, urllib.parse, urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
UA = "afd-corpus/1.0 (Build an LLM study lab; https://seyonv.github.io/explainers/)"
SIZES_MB = [1, 3, 10, 30, 100]
TEST_BYTES = 1_000_000

# Test text is dated 2025 onwards, more than seven years after GPT-2's training data (WebText, scraped up to
# late 2017), so GPT-2 can't have seen it. Training dates never overlap validation or test dates.
TEST_DATES = [date(y, m, 15) for y, m in [(2025, 1), (2025, 4), (2025, 7), (2025, 10), (2026, 1), (2026, 4), (2026, 7)]]
VAL_DATES = [date(2024, 12, 10), date(2024, 12, 20)]


def train_dates(n=90, seed=0):
    """A fixed, seeded sample of days from 2018-01-01 to 2024-11-30, in the order the corpus uses them."""
    start, end = date(2018, 1, 1), date(2024, 11, 30)
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    return random.Random(seed).sample(days, n)


def fetch_day(d, raw_dir):
    """Every office's AFDs for one UTC day, cached at raw_dir/YYYY-MM-DD.txt."""
    path = raw_dir / f"{d.isoformat()}.txt"
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    q = urllib.parse.urlencode({"pil": "AFD", "sdate": f"{d.isoformat()}T00:00Z",
                                "edate": f"{(d + timedelta(1)).isoformat()}T00:00Z",
                                "limit": 9999, "fmt": "text", "order": "asc"})
    req = urllib.request.Request(f"{BASE}?{q}", headers={"User-Agent": UA})
    for attempt in range(4):           # a dropped connection mid-download is common; wait and try again
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read()
            break
        except (OSError, http.client.HTTPException) as e:
            if attempt == 3:
                sys.exit(f"{d}: download failed 4 times ({e}). Run the same command again; finished days are cached.")
            print(f"  {d}: {type(e).__name__}, retrying in {10 * (attempt + 1)} s", flush=True)
            time.sleep(10 * (attempt + 1))
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    time.sleep(3)                     # be polite: one request at a time, with a pause
    return body


WMO = re.compile(r"^[A-Z]{4}\d{2} [A-Z]{4} \d{6}( [A-Z]{3})?$")


def clean(product):
    """One raw product -> (office, text). Drops the transmission framing, keeps the readable discussion."""
    text = product.replace("\x03", "").replace("\r", "")
    lines = text.split("\n")
    office, out, head = None, [], True
    for ln in lines:
        s = ln.rstrip()
        if head:
            if not s or re.fullmatch(r"\d{3}", s) or WMO.match(s):
                continue
            if re.fullmatch(r"AFD[A-Z0-9]{3}", s):
                office = s[3:]
                continue
            head = False
        out.append("" if s in ("&&", "$$") else s)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")
    return office, body + "\n" if body else ""


def split_products(raw):
    return [p for p in raw.decode("utf-8", errors="replace").split("\x01") if p.strip()]


def dedupe(products):
    """Updated discussions copy whole paragraphs from the office's earlier ones. Keep each paragraph the first
    time an office writes it that day (compared with whitespace collapsed), and drop exact repeat products."""
    seen_products, seen_paras, out = set(), {}, []
    for office, body in products:
        if not body or body in seen_products:
            continue
        seen_products.add(body)
        mine = seen_paras.setdefault(office, set())
        head, _, rest = body.partition("\n\n")          # title, office and issue time always stay
        keep = []
        for para in rest.split("\n\n"):
            key = " ".join(para.split())
            if len(key) > 30:
                if key in mine:
                    continue
                mine.add(key)
            keep.append(para)
        text = "\n\n".join([head] + [p for p in keep if p.strip()]).strip("\n")
        if len(" ".join(text.partition("\n\n")[2].split())) >= 40:   # nothing new left: skip it
            out.append((office, text + "\n"))
    return out


BLOCK_DAYS = 12      # training days are used in blocks of 12, interleaved, so even 1 MB spans 12 dates


def day_products(d, raw_dir):
    return [b for _, b in dedupe([clean(p) for p in split_products(fetch_day(d, raw_dir))])]


def mixed(days, raw_dir, seed):
    """All products from these days in one seeded shuffle, so a prefix of any size spans every day."""
    prods = [b for d in days for b in day_products(d, raw_dir)]
    random.Random(f"{seed}:{days[0].isoformat()}:{len(days)}").shuffle(prods)
    return prods


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_prefix(products, path, limit):
    """Products in order until the next one would pass `limit` bytes. Returns (bytes, count)."""
    n, k, parts = 0, 0, []
    for b in products:
        size = len(b.encode("utf-8")) + 1
        if n + size > limit:
            break
        parts.append(b); n += size; k += 1
    Path(path).write_text("\n".join(parts) + ("\n" if parts else ""), encoding="utf-8")
    return n, k


def build(data, n_train_days, log=print):
    raw, corpus = data / "raw", data / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    manifest = {"source": BASE, "test_dates": [d.isoformat() for d in TEST_DATES],
                "val_dates": [d.isoformat() for d in VAL_DATES], "files": {}}
    test = mixed(TEST_DATES, raw, 0)
    log(f"test: {len(TEST_DATES)} days, {len(test)} products after cleaning, mixed before the 1 MB cut")
    write_prefix(test, corpus / "test.txt", TEST_BYTES)
    write_prefix(mixed(VAL_DATES, raw, 0), corpus / "val.txt", TEST_BYTES)
    # Training text comes in blocks of BLOCK_DAYS dates, each block mixed on its own: --quick is the first block,
    # and every size is a prefix of the same order, so --quick and --full write identical small files.
    train, dates = [], train_dates()[:n_train_days]
    for k in range(0, len(dates), BLOCK_DAYS):
        for d in dates[k:k + BLOCK_DAYS]:
            fetch_day(d, raw)
        train += mixed(dates[k:k + BLOCK_DAYS], raw, 0)
        log(f"train days {k + 1}-{min(k + BLOCK_DAYS, len(dates))} of {len(dates)}: {sum(len(b) for b in train) / 1e6:.1f} MB so far")
    manifest["train_dates"] = [d.isoformat() for d in dates]
    total = sum(len(b.encode("utf-8")) + 1 for b in train)
    for mb in SIZES_MB:
        if total < mb * 1_000_000:
            log(f"train_{mb}MB: not enough text yet ({total / 1e6:.1f} MB); run --full")
            continue
        write_prefix(train, corpus / f"train_{mb}MB.txt", mb * 1_000_000)
    for f in sorted(corpus.glob("*.txt")):
        t = f.read_text(encoding="utf-8")
        manifest["files"][f.name] = {"bytes": len(t.encode("utf-8")), "products": t.count("Area Forecast Discussion"), "sha256": sha(f)}
    (corpus / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="the first 12 training days + validation + test")
    g.add_argument("--full", action="store_true", help="all 90 training days")
    g.add_argument("--stats", action="store_true", help="print what's in data/corpus")
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    a = ap.parse_args(argv)
    if a.stats:
        m = json.loads((a.data / "corpus" / "MANIFEST.json").read_text())
        for name, f in m["files"].items():
            print(f"{name:18} {f['bytes']:>11,} bytes  {f['products']:>6,} discussions  sha256 {f['sha256'][:12]}")
        return 0
    t0 = time.time()
    m = build(a.data, BLOCK_DAYS if a.quick else 90)
    m["seconds"] = round(time.time() - t0)
    (a.data / "corpus" / "MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
    print(f"\nwrote {a.data / 'corpus'}")
    for name, f in m["files"].items():
        print(f"{name:18} {f['bytes']:>11,} bytes  sha256 {f['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
