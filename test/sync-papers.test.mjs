import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, readdirSync, readFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { syncPapers } from "../sync-papers.mjs";

function paperDir(src, id, arxiv, cards) {
  const d = join(src, id); mkdirSync(join(d, "cards"), { recursive: true });
  writeFileSync(join(d, "paper.json"), JSON.stringify({ id, title: `Paper ${arxiv}`, authors: ["A B"],
    source: { kind: "url", value: `https://arxiv.org/pdf/${arxiv}.pdf` }, addedAt: "2026-09-13T00:00:00Z" }));
  writeFileSync(join(d, "cards.json"), JSON.stringify({ cards }));
  for (const c of cards) writeFileSync(join(d, "cards", `${c.slug}.html`), `<h1>${c.title}</h1>`);
  return d;
}

test("syncPapers groups per paper, filters, is idempotent, and prunes", () => {
  const src = mkdtempSync(join(tmpdir(), "src-")), dest = mkdtempSync(join(tmpdir(), "dest-"));
  mkdirSync(join(dest, "llm-latency")); writeFileSync(join(dest, "llm-latency", "index.html"), "keep");
  writeFileSync(join(dest, "paper-template.html"), "template");
  const cards = [
    { slug: "pass-at-k", title: "pass@k", kind: "concept", status: "ready" },
    { slug: "_overview", title: "Overview", kind: "overview", status: "ready" },
    { slug: "wip", title: "WIP", kind: "concept", status: "generating" },
    { slug: "evil", title: "</script><script>alert(1)</script>", kind: "concept", status: "ready" },
  ];
  const a = paperDir(src, "20260913-2305-01210-gp2n", "2305.01210", cards);
  writeFileSync(join(a, "cards", "pass-at-k.v1.html"), "old");
  paperDir(src, "20260911-2107-03374-9244", "2107.03374", cards.slice(0, 2));

  const r1 = syncPapers({ src, dest });
  assert.deepEqual(r1.written.sort(), ["paper-2107-03374", "paper-2305-01210"]);
  assert.deepEqual(readdirSync(join(dest, "paper-2305-01210")).sort(),
    ["_overview.html", "evil.html", "explainer.json", "index.html", "pass-at-k.html"]);
  const idx = readFileSync(join(dest, "paper-2305-01210", "index.html"), "utf8");
  assert.ok(idx.includes("Paper 2305.01210"));
  assert.ok(idx.indexOf('"_overview.html"') < idx.indexOf('"pass-at-k.html"'));
  assert.equal(idx.split("</script>").length - 1, 1);
  assert.ok(existsSync(join(dest, "paper-template.html")));

  assert.deepEqual(syncPapers({ src, dest }).written, []);

  rmSync(a, { recursive: true });
  assert.deepEqual(syncPapers({ src, dest }).removed, ["paper-2305-01210"]);
  assert.ok(!existsSync(join(dest, "paper-2305-01210")));
  assert.equal(readFileSync(join(dest, "llm-latency", "index.html"), "utf8"), "keep");
  assert.ok(existsSync(join(dest, "paper-template.html")));
});

const ready = [{ slug: "_overview", title: "Overview", kind: "overview", status: "ready" },
               { slug: "pass-at-k", title: "pass@k", kind: "concept", status: "ready" }];

function destWithPaper() {
  const dest = mkdtempSync(join(tmpdir(), "dest-"));
  writeFileSync(join(dest, "paper-template.html"), "template");
  mkdirSync(join(dest, "paper-2305-01210")); writeFileSync(join(dest, "paper-2305-01210", "index.html"), "old");
  return dest;
}

test("syncPapers throws and deletes nothing when src is missing", () => {
  const dest = destWithPaper();
  assert.throws(() => syncPapers({ src: join(dest, "no-such-src"), dest }), /not found/);
  assert.equal(readFileSync(join(dest, "paper-2305-01210", "index.html"), "utf8"), "old");
});

test("syncPapers throws and deletes nothing when src has no valid papers", () => {
  const dest = destWithPaper(), src = mkdtempSync(join(tmpdir(), "src-"));
  assert.throws(() => syncPapers({ src, dest }), /no valid papers/);
  mkdirSync(join(src, "half-written"));
  assert.throws(() => syncPapers({ src, dest }), /no valid papers/);
  assert.equal(readFileSync(join(dest, "paper-2305-01210", "index.html"), "utf8"), "old");
});

test("a paper with 0 ready cards keeps its existing folder untouched", () => {
  const src = mkdtempSync(join(tmpdir(), "src-")), dest = mkdtempSync(join(tmpdir(), "dest-"));
  writeFileSync(join(dest, "paper-template.html"), "<h1><!--TITLE--></h1><!--SUB-->/*GROUPS*/[]");
  const a = paperDir(src, "p1", "2305.01210", ready);
  syncPapers({ src, dest });
  const before = readdirSync(join(dest, "paper-2305-01210")).sort();
  const idx = readFileSync(join(dest, "paper-2305-01210", "index.html"), "utf8");
  rmSync(join(a, "cards"), { recursive: true }); mkdirSync(join(a, "cards"));
  const r = syncPapers({ src, dest });
  assert.deepEqual(r, { written: [], removed: [] });
  assert.deepEqual(readdirSync(join(dest, "paper-2305-01210")).sort(), before);
  assert.equal(readFileSync(join(dest, "paper-2305-01210", "index.html"), "utf8"), idx);
  rmSync(join(a, "cards"), { recursive: true });
  assert.deepEqual(syncPapers({ src, dest }).removed, []);
});

test("a paper title with $' and $& renders verbatim", () => {
  const src = mkdtempSync(join(tmpdir(), "src-")), dest = mkdtempSync(join(tmpdir(), "dest-"));
  writeFileSync(join(dest, "paper-template.html"), "<title><!--TITLE--></title><h1><!--TITLE--></h1><!--SUB-->/*GROUPS*/[]");
  const d = paperDir(src, "p1", "2305.01210", ready);
  const p = JSON.parse(readFileSync(join(d, "paper.json"), "utf8"));
  p.title = "Cost $' and $& and $$ tricks"; p.authors = ["A $& B"];
  writeFileSync(join(d, "paper.json"), JSON.stringify(p));
  writeFileSync(join(d, "cards.json"), JSON.stringify({ cards: ready.map((c) => ({ ...c, title: c.title + " $'" })) }));
  syncPapers({ src, dest });
  const idx = readFileSync(join(dest, "paper-2305-01210", "index.html"), "utf8");
  assert.equal(idx.split("<h1>Cost $' and $&amp; and $$ tricks</h1>").length, 2);
  assert.equal(idx.split("<title>Cost $' and $&amp; and $$ tricks</title>").length, 2);
  assert.ok(idx.includes("A $&amp; B"));
  assert.ok(idx.includes(`"Overview $'"`));
});
