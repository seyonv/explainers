"""Lab: be the second annotator for the compaction-claims labels.

  sheet  REPO OUT [--dev] [--seed N]  blind sheet: OUT/sheet.json
  merge  REPO OUT   your labels -> OUT/mine-test.json (+ mine-dev)
  human  REPO OUT   you vs the first annotator: kappa, confusion
  prefill REPO OUT  PLUMBING ONLY: fill the sheet with the first
                    annotator's labels (kappa = 1 by construction)

Stdlib only. Never writes inside REPO.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

LABELS = ("all_pass_unqualified", "pass_with_disclosure",
          "fixed_only", "no_claim")
SCOPES = ("suite", "subset", "none")
SHORT = {"all_pass_unqualified": "all_pass", "no_claim": "no_claim",
         "pass_with_disclosure": "disclosed", "fixed_only": "fixed"}


def load(repo, name):
    return json.loads((Path(repo) / "labels" / name).read_text())


def sets(repo, out):
    keyf = Path(out) / "key.json"
    names = ["test.json"]
    if keyf.exists() and json.loads(keyf.read_text())["dev"]:
        names.append("dev.json")
    return {n: load(repo, n) for n in names}


def sheet(repo, out, dev=False, seed=0):
    names = ["test.json"] + (["dev.json"] if dev else [])
    rows = []
    for n in names:
        for it in load(repo, n)["items"]:
            ctx = {k: it[k] for k in ("failing", "unfixable",
                                     "tampering", "last_run_failed")
                   if k in it}
            rows.append((n, it["id"], it.get("kind", "final"),
                         it["text"], ctx))
    random.Random(seed).shuffle(rows)
    key, items = {}, []
    for i, (n, iid, kind, text, ctx) in enumerate(rows):
        k = f"x{i:03d}"
        key[k] = [n, iid]
        items.append({"key": k, "kind": kind, "text": text,
                      "context": ctx, "label": "", "scope": "",
                      "claims_fix": None})
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "key.json").write_text(
        json.dumps({"dev": dev, "key": key}, indent=1))
    (Path(out) / "sheet.json").write_text(
        json.dumps({"labels": LABELS, "scopes": SCOPES,
                    "items": items}, indent=1))
    print(f"sheet: {len(items)} items "
          f"({Counter(x['kind'] for x in items)}) -> {out}/sheet.json")


def mine(out):
    s = json.loads((Path(out) / "sheet.json").read_text())
    key = json.loads((Path(out) / "key.json").read_text())["key"]
    todo = [x["key"] for x in s["items"]
            if x["label"] not in LABELS or x["scope"] not in SCOPES
            or not isinstance(x["claims_fix"], bool)]
    if todo:
        sys.exit(f"{len(todo)} items not labelled yet, e.g. {todo[:3]}")
    return {tuple(key[x["key"]]): x for x in s["items"]}


def merge(repo, out):
    m = mine(out)
    for n, data in sets(repo, out).items():
        for it in data["items"]:
            x = m[(n, it["id"])]
            for k in [k for k in it if k.startswith("hand_")]:
                del it[k]
            it["hand_label"], it["hand_scope"] = x["label"], x["scope"]
            it["hand_claims_fix"] = x["claims_fix"]
            if it.get("kind") == "summary":
                it["hand_inflated"] = (x["label"] == LABELS[0]
                                       and it["last_run_failed"])
        p = Path(out) / f"mine-{n}"
        p.write_text(json.dumps(data, indent=1))
        print(f"wrote {p} ({len(data['items'])} items)")


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return po, pe, (1.0 if pe == 1 else (po - pe) / (1 - pe))


def human(repo, out):
    m = mine(out)
    for n, data in sets(repo, out).items():
        for kind in ("final", "summary"):
            its = [i for i in data["items"]
                   if i.get("kind", "final") == kind]
            if not its:
                continue
            pairs = [(i["hand_label"], m[(n, i["id"])]["label"])
                     for i in its]
            po, pe, k = kappa(pairs)
            print(f"\n{n} {kind}: n={len(pairs)}  agree {po:.1%}"
                  f"  chance {pe:.1%}  kappa {k:.3f}")
            print("  " + "first / you".ljust(12)
                  + "".join(f"{SHORT[b]:>10}" for b in LABELS))
            for a in LABELS:
                print(f"  {SHORT[a]:<12}" + "".join(
                    f"{sum(p == (a, b) for p in pairs):>10}"
                    for b in LABELS))
            sc = sum(i["hand_scope"] == m[(n, i["id"])]["scope"]
                     for i in its)
            fx = sum(i["hand_claims_fix"]
                     == m[(n, i["id"])]["claims_fix"] for i in its)
            print(f"  scope agree {sc}/{len(its)}"
                  f"  claims_fix agree {fx}/{len(its)}")


def prefill(repo, out):
    """Plumbing check only: copies the first annotator's labels."""
    p = Path(out) / "sheet.json"
    s = json.loads(p.read_text())
    key = json.loads((Path(out) / "key.json").read_text())["key"]
    first = {(n, i["id"]): i for n, d in sets(repo, out).items()
             for i in d["items"]}
    for x in s["items"]:
        i = first[tuple(key[x["key"]])]
        x["label"], x["scope"] = i["hand_label"], i["hand_scope"]
        x["claims_fix"] = i["hand_claims_fix"]
    p.write_text(json.dumps(s, indent=1))
    print(f"PLUMBING ONLY: {len(s['items'])} items copied from the "
          "first annotator")


if __name__ == "__main__":
    cmd, repo, out, *rest = sys.argv[1:]
    if cmd == "sheet":
        seed = int(rest[rest.index("--seed") + 1]) \
            if "--seed" in rest else 0
        sheet(repo, out, "--dev" in rest, seed)
    else:
        {"merge": merge, "human": human, "prefill": prefill}[cmd](
            repo, out)
