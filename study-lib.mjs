// Study Mode: turns a card's <course>/_study/<slug>.json into two marked blocks inside the card.
//   <!-- study:prime -->  after the subtitle: problem, where it fits, what you'll meet, questions to read with
//   <!-- study:panel -->  before the footer: a CSS-only stepper (core model → compress → retrieve → compare →
//                         predict → apply → reflect); each stage appears only once the previous one is opened
// Cards may not contain <script>, so everything here is static HTML + CSS. Used by study-build.mjs and verify-study.mjs.
import { existsSync } from "node:fs";
import { join } from "node:path";

const BLOCKS = ["prime", "panel"];
const PRINCIPLES = "../learning-principles/learning-principles.html";

export const refHref = (ref, course) => {
  const [c, slug] = ref.split("/");
  return c === course ? `${slug}.html` : `../${c}/${slug}.html`;
};

export function strip(html) {
  for (const b of BLOCKS) html = html.replace(new RegExp(`\\n?<!-- study:${b} -->[\\s\\S]*?<!-- /study:${b} -->`, "g"), "");
  return html.replace(/\n?\/\* study:css \*\/[\s\S]*?\/\* \/study:css \*\//g, "");
}

const li = (xs) => xs.map((x) => `<li>${x}</li>`).join("");
const reveal = (label, inner) => `<details class="ref"><summary>${label}</summary>${inner}</details>`;
const link = (ref, ctx) => `<a href="${refHref(ref, ctx.course)}">${ctx.titleOf(ref)}</a>`;

function prime(s, ctx) {
  const p = s.prime;
  const needs = (s.prereq || []).map((r) => link(r, ctx)).join(" · ");
  const back = (s.revisit || []).map((r) => link(r, ctx)).join(" · ");
  return `<!-- study:prime -->
<aside class="prime" aria-label="Prime: read this first">
<div class="prime-k">Prime · 2 minutes before you read</div>
<p><b>The problem:</b> ${p.problem}</p>
<p><b>Where it fits:</b> ${p.fits}</p>${needs ? `\n<p class="prime-links"><b>Needs first:</b> ${needs}</p>` : ""}${back ? `\n<p class="prime-links"><b>Builds on:</b> ${back}</p>` : ""}
<p><b>You'll meet:</b> ${p.expect.map((e) => `<span class="chip">${e}</span>`).join(" ")}</p>
<p><b>By the end:</b> ${p.goal}</p>
<p class="prime-q"><b>Read with these questions</b> (answer them in your head as you go; don't take notes line by line):</p>
<ol>${li(s.attend)}</ol>
</aside>
<!-- /study:prime -->`;
}

function panel(s, ctx) {
  const c = s.core;
  const stages = [
    ["Core model", "what to internalise, what to just look up",
      `<p class="st-rule">Keep it if it changes what you can <i>explain, predict, compare, choose, build or debug</i>. Otherwise it's reference.</p>
<div class="core"><div><h4>Understand and remember</h4><ul>${li(c.must)}${li((c.elevated || []).map((e) => `<span class="up">detail that matters</span> ${e}`))}</ul></div>
<div><h4>Reference only</h4><ul class="muted">${li(c.reference)}</ul></div></div>`],
    ["Compress", "write it yourself first",
      `<p>${s.compress.prompt}</p><p class="st-how">On paper or in your notes, in your own words, without scrolling up. Aim for the smallest thing that still explains it.</p>
${reveal("Then compare with this checklist", `<ul>${li(s.compress.checklist)}</ul><p class="st-how">Missing a point? Reread only the section about it, not the whole card.</p>`)}`],
    ["Retrieve", "from memory, card closed",
      `<ol class="qs">${s.retrieve.map((r) => `<li><span class="kind">${r.kind}</span> ${r.q}${reveal("Check", `<ul>${li(r.answer)}</ul>`)}</li>`).join("")}</ol>`],
    ["Compare", "against what you already know",
      s.compare.map((x) => `<p><b>vs ${link(x.with, ctx)}:</b> ${x.q}</p>${reveal("Check", `<ul>${li(x.answer)}</ul>`)}`).join("")],
    ["Predict", "commit to an answer before revealing",
      `<p>${s.predict.q}</p>${reveal("Reveal", `<p>${s.predict.answer}</p>`)}`],
    ["Apply", "use the model, don't restate it",
      `<p>${s.apply.task}</p><p><b>Done when:</b> ${s.apply.done}</p>${s.apply.hint ? reveal("Hint", `<p>${s.apply.hint}</p>`) : ""}`],
    ["Reflect", "and decide whether to move on",
      `<ul>${li(s.reflect)}</ul><p class="st-how">Repair what came up: reopen only that section, then retry the question you missed. Tutor mode: <code>/study ${ctx.course}/${ctx.slug}</code> in Claude Code grades your answers and walks you through each stage.</p>`],
  ];
  return `<!-- study:panel -->
<section class="study" aria-label="Study this card">
<div class="study-head"><h2>Study this card</h2><span>Open each step in order · <a href="${PRINCIPLES}">why this works</a></span></div>
<input type="checkbox" id="study-all" class="study-all"><label for="study-all" class="study-all-l">show all steps (for revisiting)</label>
${stages.map(([t, sub, inner], i) => `<details class="st"><summary><b>${i + 1} · ${t}</b> <span>${sub}</span></summary><div class="st-body">${inner}</div></details>`).join("\n")}
</section>
<!-- /study:panel -->`;
}

export const CSS = `/* study:css */
.prime{border:1px solid var(--accent-border,var(--border));background:var(--accent-bg);border-radius:10px;padding:10px 16px 6px;margin:12px 0 18px;font-size:14px}
.prime p{margin:4px 0}
.prime-k{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);font-weight:600;margin-bottom:4px}
.prime .chip{display:inline-block;border:1px solid var(--border);background:var(--surface);border-radius:999px;padding:0 8px;margin:1px 2px;font-size:12.5px}
.prime ol{margin:2px 0 6px;padding-left:22px}
.prime li{margin:2px 0}
.study{border:1px solid var(--border);border-radius:12px;padding:12px 16px;margin:26px 0 18px;background:var(--surface)}
.study-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:4px 12px}
.study-head h2{margin:0;font-size:18px}
.study-head span,.study-all-l{font-size:12.5px;color:var(--muted)}
.study-all{margin:8px 4px 8px 0;vertical-align:middle}
.study details.st{border-top:1px solid var(--border);padding:8px 0}
.study details.st>summary{cursor:pointer;font-size:15px}
.study details.st>summary span{color:var(--muted);font-size:13px}
.study-all:not(:checked)~details.st:not([open])~details.st{display:none}
.st-body{font-size:14px;padding:4px 0 2px 2px}
.st-body p{margin:6px 0}
.st-body ul,.st-body ol{margin:4px 0;padding-left:22px}
.st-body li{margin:3px 0}
.st-rule,.st-how{color:var(--muted);font-size:13px}
.core{display:grid;grid-template-columns:3fr 2fr;gap:6px 18px}
.core h4{margin:4px 0;font-size:13px}
.core .muted{color:var(--muted)}
.up{font-size:11px;border:1px solid var(--accent-border,var(--border));color:var(--accent);border-radius:4px;padding:0 4px;white-space:nowrap}
.study .kind{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--accent);margin-right:4px}
.study details.ref{margin:4px 0 8px;padding:4px 10px;border-left:3px solid var(--accent-border,var(--border));background:var(--surface2);border-radius:0 6px 6px 0}
.study details.ref>summary{cursor:pointer;font-size:13px;color:var(--accent)}
@media (max-width:620px){.core{grid-template-columns:1fr}}
/* /study:css */`;

export function inject(html, s, ctx) {
  html = strip(html);
  const sub = html.match(/<p class="sub"[\s\S]*?<\/p>/) || html.match(/<\/h1>/);
  if (!sub) throw new Error("no <h1> or subtitle to anchor Prime");
  const at = sub.index + sub[0].length;
  html = html.slice(0, at) + "\n" + prime(s, ctx) + html.slice(at);
  const foot = html.search(/<(div|footer|p) class="foot"/);
  const at2 = foot >= 0 ? foot : html.includes("<!-- series-nav:bottom -->") ? html.indexOf("<!-- series-nav:bottom -->") : html.lastIndexOf("</body>");
  html = html.slice(0, at2) + panel(s, ctx) + "\n" + html.slice(at2);
  const style = html.indexOf("</style>");
  return html.slice(0, style) + CSS + "\n" + html.slice(style);
}

// ---------- validation ----------
const KEYS = ["prime", "attend", "core", "compress", "retrieve", "compare", "predict", "apply", "reflect", "revisit", "prereq"];
// Questions that could sit on any card unchanged. Topic-specific versions of these are fine.
const GENERIC = [/^what is (it|this)\??$/i, /^why does (this|it) exist\??$/i, /^what does .{1,12} stand for\??$/i,
  /^(define|list) /i, /^what did you learn\??$/i, /^what is the definition of/i];

export function validate(s, ctx) {
  const out = [];
  const bad = (m) => out.push(m);
  for (const k of Object.keys(s)) if (!KEYS.includes(k)) bad(`unknown key: ${k}`);
  const p = s.prime || {};
  for (const k of ["problem", "fits", "goal"]) if (!p[k]) bad(`prime.${k}: missing`);
  if (!Array.isArray(p.expect) || p.expect.length < 2 || p.expect.length > 5) bad("prime.expect: 2–5 concepts");
  if (!Array.isArray(s.attend) || s.attend.length < 3 || s.attend.length > 6) bad("attend: 3–6 questions");
  const c = s.core || {};
  if (!Array.isArray(c.must) || c.must.length < 2) bad("core.must: at least 2 causal chains");
  for (const m of c.must || []) if (!/→|->/.test(m)) bad(`core.must is not a causal chain (use →): "${m.slice(0, 60)}"`);
  if (!Array.isArray(c.reference) || !c.reference.length) bad("core.reference: at least 1 item");
  if (!s.compress?.prompt || !(s.compress?.checklist?.length >= 2)) bad("compress: prompt + ≥2 checklist points");
  const r = s.retrieve || [];
  if (r.length < 3 || r.length > 5) bad("retrieve: 3–5 questions");
  for (const k of ["concept", "mechanism", "tradeoff"]) if (!r.some((x) => x.kind === k)) bad(`retrieve: needs a ${k} question`);
  for (const x of r) if (!x.answer?.length) bad(`retrieve: no answer for "${(x.q || "").slice(0, 40)}"`);
  if (!Array.isArray(s.compare) || !s.compare.length) bad("compare: at least 1");
  for (const x of s.compare || []) if (!x.with || !x.q || !x.answer?.length) bad("compare: each needs with, q, answer");
  if (!s.predict?.q || !s.predict?.answer) bad("predict: q + answer");
  if (!s.apply?.task || !s.apply?.done) bad("apply: task + done");
  if (!Array.isArray(s.reflect) || s.reflect.length < 3) bad("reflect: at least 3 prompts");
  const qs = [...(s.attend || []), ...r.map((x) => x.q), s.predict?.q].filter(Boolean);
  for (const q of qs) if (GENERIC.some((g) => g.test(q.trim()))) bad(`generic question: "${q}"`);
  const refs = [...(s.compare || []).map((x) => x.with), ...(s.revisit || []), ...(s.prereq || [])].filter(Boolean);
  for (const ref of refs) {
    if (!/^[\w.-]+\/[\w.-]+$/.test(ref)) bad(`bad ref (want course/slug): ${ref}`);
    else if (ctx.root && !existsSync(join(ctx.root, ref + ".html")) && !ctx.known?.has(ref)) bad(`unknown card ${ref}`);
  }
  // Every string field is an HTML fragment dropped into a card, so it must follow the card rules too.
  const walk = (v, path) => {
    if (typeof v === "string") {
      if (/<script/i.test(v)) bad(`${path}: no <script>`);
      if (/style=/i.test(v)) bad(`${path}: no inline style`);
      if (/#[0-9a-f]{3,8}\b/i.test(v) && /color|background/i.test(v)) bad(`${path}: no hard-coded colours`);
    } else if (Array.isArray(v)) v.forEach((x, i) => walk(x, `${path}[${i}]`));
    else if (v && typeof v === "object") for (const [k, x] of Object.entries(v)) walk(x, path ? `${path}.${k}` : k);
  };
  walk(s, "");
  return out;
}
