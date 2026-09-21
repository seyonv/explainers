#!/usr/bin/env node
// Builds index.html: one tile per explainer folder in this repo.
// A folder with index.html is a curriculum; a folder with one card is a single explainer.
// Optional <folder>/explainer.json overrides {"title","description","date"}.
import { readdirSync, readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { execFileSync } from "node:child_process";

const root = new URL(".", import.meta.url).pathname;
const text = (html, re) => (html.match(re)?.[1] || "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
const clip = (s, n) => (s.length > n ? s.slice(0, s.lastIndexOf(" ", n)) + "…" : s);

function firstCommitDate(dir) {
  try {
    const out = execFileSync("git", ["log", "--diff-filter=A", "--follow", "--format=%cs", "--", dir], { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim().split("\n");
    return out[out.length - 1] || null;
  } catch { return null; }
}

const entries = [];
for (const slug of readdirSync(root).sort()) {
  const dir = join(root, slug);
  if (slug.startsWith(".") || !statSync(dir).isDirectory()) continue;
  const htmls = readdirSync(dir).filter((f) => f.endsWith(".html")).sort();
  if (!htmls.length) continue;
  const isCurriculum = htmls.includes("index.html");
  const cards = htmls.filter((f) => f !== "index.html");
  const main = isCurriculum ? "index.html" : cards[0];
  const html = readFileSync(join(dir, main), "utf8");
  const preview = isCurriculum ? (cards.find((f) => f === "_overview.html") || cards.find((f) => f.startsWith("_")) || cards[0]) : main;
  const override = existsSync(join(dir, "explainer.json")) ? JSON.parse(readFileSync(join(dir, "explainer.json"), "utf8")) : {};
  entries.push({
    slug,
    href: `${slug}/${isCurriculum ? "" : main}`,
    preview: `${slug}/${preview}`,
    title: override.title || text(html, /<h1[^>]*>([\s\S]*?)<\/h1>/) || text(html, /<title>([\s\S]*?)<\/title>/) || slug,
    description: clip(override.description || text(html, /<p class="sub">([\s\S]*?)<\/p>/), 170),
    kind: override.kind === "paper" ? "paper" : isCurriculum ? "curriculum" : "single",
    cards: cards.length,
    date: override.date || firstCommitDate(slug) || new Date(statSync(join(dir, main)).mtimeMs).toISOString().slice(0, 10),
  });
}
entries.sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title));

const page = readFileSync(join(root, "hub-template.html"), "utf8")
  .replace("/*ENTRIES*/[]", JSON.stringify(entries, null, 1))
  .replace("<!--COUNT-->", `${entries.length} explainer${entries.length === 1 ? "" : "s"}, ${entries.reduce((n, e) => n + e.cards, 0)} cards`);
writeFileSync(join(root, "index.html"), page);
writeFileSync(join(root, "hub.json"), JSON.stringify({
  explainers: entries.length,
  papers: entries.filter((e) => e.kind === "paper").length,
  cards: entries.reduce((n, e) => n + e.cards, 0),
}) + "\n");
console.log(`index.html: ${entries.length} entries (${entries.map((e) => e.slug).join(", ")})`);
