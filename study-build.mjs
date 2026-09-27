#!/usr/bin/env node
// Builds Study Mode into cards: for every <course>/_study/<slug>.json, injects the Prime block and the study
// stepper into <course>/<slug>.html (see study-lib.mjs). Cards whose JSON was removed get their blocks stripped.
// Usage: node study-build.mjs [course ...]   (default: every folder). Idempotent; publish.sh runs it before course-nav.
//        node study-build.mjs --card course/slug   touches only that card (safe for parallel card writers).
import { readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { inject, strip } from "./study-lib.mjs";

const root = new URL(".", import.meta.url).pathname;
const args = process.argv.slice(2);
const card = args[0] === "--card" ? args[1] : null;
const courses = card ? [card.split("/")[0]] : args.length ? args : readdirSync(root, { withFileTypes: true })
  .filter((e) => e.isDirectory() && !e.name.startsWith(".") && !["tasks", "test", "launchd", "node_modules"].includes(e.name)).map((e) => e.name);

const titleOf = (ref) => {
  const f = join(root, ref + ".html");
  const t = existsSync(f) && readFileSync(f, "utf8").match(/<h1[^>]*>([\s\S]*?)<\/h1>/)?.[1].replace(/<[^>]+>/g, "").trim();
  return t || ref.split("/")[1].replace(/-/g, " ");
};

let built = 0, stripped = 0, failed = 0;
for (const course of courses) {
  const dir = join(root, course);
  const sdir = join(dir, "_study");
  const jsons = new Set(existsSync(sdir) ? readdirSync(sdir).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5)) : []);
  for (const f of readdirSync(dir).filter((f) => f.endsWith(".html") && f !== "index.html" && (!card || f === card.split("/")[1] + ".html"))) {
    const slug = f.slice(0, -5), path = join(dir, f), html = readFileSync(path, "utf8");
    let out;
    try {
      out = jsons.has(slug) ? inject(html, JSON.parse(readFileSync(join(sdir, slug + ".json"), "utf8")), { course, slug, titleOf }) : strip(html);
    } catch (e) { console.error(`${course}/${f}: ${e.message}`); failed++; continue; }
    if (out !== html) { writeFileSync(path, out); jsons.has(slug) ? built++ : stripped++; }
    jsons.delete(slug);
  }
  if (card) jsons.clear();
  for (const orphan of jsons) { console.error(`${course}/_study/${orphan}.json: no card ${orphan}.html`); failed++; }
}
console.log(`study: ${built} built, ${stripped} stripped${failed ? `, ${failed} failed` : ""}`);
process.exit(failed ? 1 : 0);
