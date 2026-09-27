#!/usr/bin/env node
// Checks study JSONs: schema, causal-chain core model, retrieval mix, generic questions, links that resolve,
// and questions copy-pasted across cards of one course (the "boilerplate boxes" failure).
// Usage: node verify-study.mjs <course> [--only slug,slug] [--allow-missing]
//   --allow-missing: refs may point at cards that are listed in a course index / z2h cards.json but not written yet.
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { validate } from "./study-lib.mjs";

const root = new URL(".", import.meta.url).pathname;
const args = process.argv.slice(2);
const course = args.find((a) => !a.startsWith("--") && !args[args.indexOf(a) - 1]?.startsWith("--only"));
const only = args.includes("--only") ? args[args.indexOf("--only") + 1].split(",") : null;
const allowMissing = args.includes("--allow-missing");
if (!course) { console.error("usage: node verify-study.mjs <course> [--only a,b] [--allow-missing]"); process.exit(2); }

const known = new Set();
if (allowMissing) {
  const cards = join(root, "tasks/z2h-kit/cards.json");
  if (existsSync(cards)) for (const c of JSON.parse(readFileSync(cards, "utf8"))) known.add(`${c.course}/${c.slug}`);
  for (const d of readdirSync(root)) {
    const idx = join(root, d, "index.html");
    if (existsSync(idx)) for (const m of readFileSync(idx, "utf8").matchAll(/\["([\w.-]+)\.html"/g)) known.add(`${d}/${m[1]}`);
  }
}

const sdir = join(root, course, "_study");
const files = (existsSync(sdir) ? readdirSync(sdir).filter((f) => f.endsWith(".json")) : []).filter((f) => !only || only.includes(f.slice(0, -5)));
const problems = [], seen = new Map();
for (const f of files) {
  const slug = f.slice(0, -5);
  let s;
  try { s = JSON.parse(readFileSync(join(sdir, f), "utf8")); } catch (e) { problems.push(`${f}: invalid JSON (${e.message})`); continue; }
  if (!existsSync(join(root, course, slug + ".html"))) problems.push(`${f}: no card ${slug}.html`);
  for (const p of validate(s, { course, slug, root, known })) problems.push(`${f}: ${p}`);
  for (const q of [...(s.attend || []), ...(s.retrieve || []).map((r) => r.q)]) {
    const k = q.toLowerCase().replace(/<[^>]+>/g, "").replace(/\W+/g, " ").trim();
    if (seen.has(k)) problems.push(`${f}: question repeated from ${seen.get(k)}: "${q.slice(0, 60)}"`); else seen.set(k, f);
  }
}
if (!files.length) problems.push(`${course}: no study JSONs${only ? " matching --only" : ""}`);
console.log(problems.length ? `FAIL (${problems.length})\n` + problems.join("\n") : `PASS ${course}: ${files.length} study files`);
process.exit(problems.length ? 1 : 0);
