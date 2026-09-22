#!/usr/bin/env node
// Mirrors iterate_visuals_for_papers papers into paper-<arxiv>/ course folders. Reads the source, never writes it.
import { readdirSync, readFileSync, writeFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { folderName, selectCards, explainerJson } from "./papers-lib.mjs";

const here = new URL(".", import.meta.url).pathname;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function renderIndex(paper, meta, cards) {
  const [first, ...rest] = cards;
  const groups = [["Start here", true, first ? [[first.file, first.title, ""]] : []],
                  ["Concepts", false, rest.map((c) => [c.file, c.title, ""])]];
  const src = meta.source ? ` · <a href="${esc(meta.source)}">${esc(meta.description.split(" · ").pop())}</a>` : "";
  const sub = `${esc((paper.authors || []).join(", "))}${src}. ${cards.length} cards: click any card to read it here; use ← → to page through.`;
  return readFileSync(join(here, "paper-template.html"), "utf8")
    .replaceAll("<!--TITLE-->", () => esc(paper.title))
    .replace("<!--SUB-->", () => sub)
    .replace("/*GROUPS*/[]", () => JSON.stringify(groups).replace(/</g, "\\u003c"));
}

// Writes only when bytes differ; returns true if anything changed.
function put(path, data) {
  if (existsSync(path) && readFileSync(path).equals(Buffer.from(data))) return false;
  writeFileSync(path, data); return true;
}

// Throws (without touching dest) when src is missing or holds no valid papers, so a
// missing or unmounted source never wipes the published folders. A paper-* folder is
// pruned only when no source paper maps to it; a paper with 0 ready cards is left as is.
export function syncPapers({ src, dest }) {
  if (!existsSync(src)) throw new Error(`papers source not found: ${src}`);
  const papers = [];
  for (const id of readdirSync(src).sort()) {
    const dir = join(src, id);
    if (!existsSync(join(dir, "paper.json")) || !existsSync(join(dir, "cards.json"))) continue;
    const paper = JSON.parse(readFileSync(join(dir, "paper.json"), "utf8"));
    const name = folderName(paper);
    if (!name) { console.warn(`papers: skipping ${id}: unsafe paper id`); continue; }
    papers.push({ dir, paper, name });
  }
  if (!papers.length) throw new Error(`papers source has no valid papers: ${src}`);

  const written = [], removed = [], known = new Set(papers.map((p) => p.name));
  for (const { dir, paper, name } of papers) {
    const cardsDir = join(dir, "cards");
    const cards = selectCards(JSON.parse(readFileSync(join(dir, "cards.json"), "utf8")), existsSync(cardsDir) ? readdirSync(cardsDir) : []);
    if (!cards.length) continue;
    const out = join(dest, name), meta = explainerJson(paper, cards);
    mkdirSync(out, { recursive: true });
    const want = new Set(["index.html", "explainer.json", ...cards.map((c) => c.file)]);
    let changed = false;
    for (const f of readdirSync(out)) if (!want.has(f)) { rmSync(join(out, f)); changed = true; }
    for (const c of cards) changed = put(join(out, c.file), readFileSync(join(cardsDir, c.file))) || changed;
    changed = put(join(out, "explainer.json"), JSON.stringify(meta, null, 2) + "\n") || changed;
    changed = put(join(out, "index.html"), renderIndex(paper, meta, cards)) || changed;
    if (changed) written.push(name);
  }
  for (const ent of readdirSync(dest, { withFileTypes: true })) {
    const f = ent.name;
    if (ent.isDirectory() && f.startsWith("paper-") && !known.has(f)) { rmSync(join(dest, f), { recursive: true }); removed.push(f); }
  }
  return { written, removed };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  let r;
  try { r = syncPapers({ src: join(homedir(), "Desktop/repos/iterate_visuals_for_papers/data/papers"), dest: here }); }
  catch (e) { console.error(`papers: ${e.message}`); process.exit(1); }
  console.log(`papers: ${r.written.length} updated${r.written.length ? " (" + r.written.join(", ") + ")" : ""}, ${r.removed.length} removed`);
}
