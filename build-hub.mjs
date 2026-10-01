#!/usr/bin/env node
// Builds index.html: one tile per explainer folder in this repo.
// A folder with index.html is a curriculum; a folder with one card is a single explainer.
// Optional <folder>/explainer.json overrides {"title","description","date"}.
// --tracked: only folders with files in the git index (unattended publishes skip unstaged work).
import { readdirSync, readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { execFileSync } from "node:child_process";

const root = new URL(".", import.meta.url).pathname;
const text = (html, re) => (html.match(re)?.[1] || "").replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/\s+/g, " ").trim();
const clip = (s, n) => (s.length > n ? s.slice(0, s.lastIndexOf(" ", n)) + "…" : s);

function firstCommitDate(dir) {
  try {
    const out = execFileSync("git", ["log", "--diff-filter=A", "--follow", "--format=%cs", "--", dir], { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim().split("\n");
    return out[out.length - 1] || null;
  } catch { return null; }
}

const tracked = process.argv.includes("--tracked")
  ? new Set(execFileSync("git", ["ls-files"], { cwd: root, encoding: "utf8" }).split("\n").map((f) => f.split("/")[0]))
  : null;

const entries = [];
for (const slug of readdirSync(root).sort()) {
  const dir = join(root, slug);
  if (slug.startsWith(".") || !statSync(dir).isDirectory() || (tracked && !tracked.has(slug))) continue;
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
    tags: override.tags || [],
    date: override.date || firstCommitDate(slug) || new Date(statSync(join(dir, main)).mtimeMs).toISOString().slice(0, 10),
  });
}
entries.sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title));

// Each <prefix>-series.json folds its courses into one "series" tile on the map entry;
// the member courses stay in the list (search still finds them) but carry `series`.
for (const f of readdirSync(root).filter((f) => f.endsWith("-series.json"))) {
  const series = JSON.parse(readFileSync(join(root, f), "utf8"));
  const map = entries.find((e) => e.slug === series.map);
  if (!map) continue;
  map.kind = "series";
  map.title = series.title;
  map.courses = [];
  for (const c of series.courses) {
    const e = entries.find((x) => x.slug === c.slug);
    if (!e || e === map) continue;
    e.series = series.title;
    map.courses.push({ slug: e.slug, n: c.n, title: c.short, href: e.href, cards: e.cards });
  }
  map.total = map.courses.reduce((n, c) => n + c.cards, map.cards);
}
entries.sort((a, b) => (b.kind === "series") - (a.kind === "series"));

// study-path.json lays each path's day plans out as a section above the grid. Each day has a checklist and a
// one-page version; both stay in the list for search but leave the grid, like series members.
let paths = [];
if (existsSync(join(root, "study-path.json"))) {
  paths = JSON.parse(readFileSync(join(root, "study-path.json"), "utf8")).paths;
  for (const path of paths) {
    // A roadmap path groups an existing roadmap's stages into tiles instead of pointing at day plans. Each stage's
    // tick ids come from the data-ids on its link in the roadmap's index.html; all stages share one localStorage key.
    if (path.roadmap) {
      const html = readFileSync(join(root, path.roadmap, "index.html"), "utf8");
      const stages = Object.fromEntries([...html.matchAll(/href="([^"]+)" data-stage="([^"]+)" data-ids="([^"]+)"/g)]
        .map((m) => [m[2], { href: `${path.roadmap}/${m[1]}`, ids: m[3].split(" ") }]));
      for (const d of path.days) {
        d.stages = d.stages.map((s) => stages[s]).filter(Boolean);
        d.ids = d.stages.flatMap((s) => s.ids);
      }
      path.days = path.days.filter((d) => d.ids.length);
      continue;
    }
    path.days = path.days.filter((d) => entries.some((e) => e.slug === d.plan) && entries.some((e) => e.slug === d.flow));
    for (const d of path.days) {
      for (const e of entries.filter((x) => x.slug === d.plan || x.slug === d.flow)) e.series = path.title;
      const html = readFileSync(join(root, d.plan, d.plan + ".html"), "utf8");
      d.ids = [...new Set([...html.matchAll(/data-id="([^"]+)"/g)].map((m) => m[1]))];
    }
    for (const e of entries.filter((x) => x.slug === path.overview)) e.series = path.title;
  }
  paths = paths.filter((p) => p.days.length);
}

// Tell each course how much of it a study path already walks through, from the card links in the day plans,
// plus the day the path's roadmap assigns to the rest (path.later.courses), so nobody reads it twice.
const isCard = (f) => f.endsWith(".html") && f !== "index.html" && !f.startsWith("_") && !f.startsWith("checkpoint");
const dayRange = (ls) => ls.length === 1 ? ls[0] : `Days ${ls[0].replace(/^Day /, "")}–${ls[ls.length - 1].replace(/^Day /, "")}`;
for (const path of paths.filter((p) => !p.roadmap && p.short)) {
  const used = {};
  for (const d of path.days) {
    const html = readFileSync(join(root, d.plan, d.plan + ".html"), "utf8");
    for (const [, slug, card] of html.matchAll(/href="(?:\.\.\/)?([a-z0-9-]+)\/([^"\/#?]+\.html)/g)) {
      if (!isCard(card) || !existsSync(join(root, slug, card))) continue;
      const u = (used[slug] ||= { days: [], cards: new Set() });
      u.cards.add(card);
      if (!u.days.includes(d.label)) u.days.push(d.label);
    }
  }
  for (const e of entries.filter((x) => x.kind === "curriculum" && !x.inPath)) {
    const total = readdirSync(join(root, e.slug)).filter(isCard).length;
    const u = used[e.slug], later = path.later?.courses?.[e.slug];
    const p = (state, label, title) => (e.inPath = { path: path.id, state, label, title });
    if (u && u.cards.size >= total) p("all", `✓ ${path.short} · ${dayRange(u.days)}`, `All of it is in the ${path.short.toLowerCase()}, ${dayRange(u.days)}. No need to read it separately.`);
    else if (u && later) p("part", `${path.short} · ${u.cards.size}/${total} · rest ${later}`, `${u.cards.size} of ${total} cards are in the ${path.short.toLowerCase()}, ${dayRange(u.days)}; the rest is planned for ${later}.`);
    else if (later === "optional") p("optional", "Optional", `Not in the ${path.short.toLowerCase()}: read it if you want the extra depth.`);
    else if (later) p("later", `${path.short} · ${later}`, `Planned for ${later} of the ${path.short.toLowerCase()}.`);
  }
}
for (const map of entries.filter((e) => e.kind === "series")) {
  for (const c of map.courses) c.inPath = entries.find((e) => e.slug === c.slug)?.inPath;
}

// hub-topics.json files every folder under one topic (exact slug, or a prefix ending in *); the grid groups by it.
const topics = existsSync(join(root, "hub-topics.json")) ? JSON.parse(readFileSync(join(root, "hub-topics.json"), "utf8")).topics : [];
const topicOf = (slug) => topics.find((t) => t.slugs.some((p) => (p.endsWith("*") ? slug.startsWith(p.slice(0, -1)) : slug === p)))?.id || "other";
for (const e of entries) e.topic = topicOf(e.slug);

const page = readFileSync(join(root, "hub-template.html"), "utf8")
  .replace("/*ENTRIES*/[]", () => JSON.stringify(entries, null, 1).replace(/</g, "\\u003c"))
  .replace("/*TOPICS*/[]", () => JSON.stringify(topics.map(({ id, label, blurb }) => ({ id, label, blurb }))).replace(/</g, "\\u003c"))
  .replace("/*PATHS*/[]", () => JSON.stringify(paths).replace(/</g, "\\u003c"))
  .replace("<!--COUNT-->", () => `${entries.length} explainer${entries.length === 1 ? "" : "s"} · ${entries.reduce((n, e) => n + e.cards, 0)} cards · ${paths.length} study paths`);
writeFileSync(join(root, "index.html"), page);
writeFileSync(join(root, "hub.json"), JSON.stringify({
  explainers: entries.length,
  papers: entries.filter((e) => e.kind === "paper").length,
  cards: entries.reduce((n, e) => n + e.cards, 0),
}) + "\n");
console.log(`index.html: ${entries.length} entries (${entries.map((e) => e.slug).join(", ")})`);
