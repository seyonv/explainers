#!/usr/bin/env node
// Adds navigation to every course folder (any folder whose index.html has a GROUPS list):
// - every card: breadcrumbs + prev/next at the top, a big "next up" preview at the bottom
// - every course index: a contents sidebar in the reader
// Each <prefix>-series.json (perf-series.json, cs-series.json, …) makes its courses one series, which adds a
// course strip, cross-course next-up and "before this" prerequisites. Every other course is navigated on its own.
// Card order comes from each course's index.html GROUPS. Re-run after adding or reordering cards; it is idempotent.
import { readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

const root = new URL(".", import.meta.url).pathname;
const allSeries = readdirSync(root).filter((f) => f.endsWith("-series.json")).sort().reverse()
  .map((f) => ({ prefix: f.replace(/-series\.json$/, ""), ...JSON.parse(readFileSync(join(root, f), "utf8")) }));
const esc = (s) => s.replace(/&(?!amp;|lt;|gt;|quot;|#)/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const strip = (s) => s.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
const clip = (s, n) => (s.length > n ? s.slice(0, s.lastIndexOf(" ", n)) + "…" : s);
const slugOf = (f) => f.replace(/\.html$/, "");

// ---------- load courses and cards ----------
// Cards come from the GROUPS literal: either one entry per line (hand-written) or one-line JSON (generated papers).
function loadCourse(slug) {
  const idx = join(root, slug, "index.html");
  if (!existsSync(idx)) return null;
  const html = readFileSync(idx, "utf8");
  const src = html.match(/const GROUPS = (\[[\s\S]*?\]);\s*\n\s*const CARDS/)?.[1];
  if (!src) return null;
  const groups = new Function(`return ${src}`)();
  return { cards: groups.flatMap((g) => g[2].map(([file, title, desc]) => ({ file, title, desc: desc || "" }))), title: strip(html.match(/<h1[^>]*>([\s\S]*?)<\/h1>/)?.[1] || slug) };
}

// A track is a run of courses whose cards page into each other: one series, or one standalone course.
const seriesSlugs = new Set(allSeries.flatMap((s) => s.courses.map((c) => c.slug)));
const seriesTracks = allSeries.map((series) => ({ series, title: series.title, home: `../${series.map}/index.html`, courses: series.courses.map((c) => {
  const got = loadCourse(c.slug);
  return got ? { ...c, ...got, label: `${c.n} · ${c.short}`, built: true } : { ...c, label: `${c.n} · ${c.short}`, built: false, cards: [] };
}) }));
const standalone = readdirSync(root, { withFileTypes: true })
  .filter((e) => e.isDirectory() && !e.name.startsWith(".") && !seriesSlugs.has(e.name))
  .map((e) => { const got = loadCourse(e.name); return got && got.cards.length ? { slug: e.name, ...got, label: got.title, built: true } : null; })
  .filter(Boolean);
const tracks = [
  ...seriesTracks,
  ...standalone.map((c) => ({ series: null, title: "All explainers", home: "../index.html", courses: [c] })),
];
for (const t of tracks) {
  t.built = t.courses.filter((c) => c.built);
  for (const c of t.courses) c.track = t;
  t.flat = t.built.flatMap((c) => c.cards.map((card, i) => ({ ...card, course: c, i })));
}
const flat = tracks.flatMap((t) => t.flat);
const built = tracks.flatMap((t) => t.built);
const find = (courseSlug, file) => flat.find((x) => x.course.slug === courseSlug && x.file === file);
const href = (from, to, withHash = true) => {
  const base = from.course.slug === to.course.slug ? "index.html" : `../${to.course.slug}/index.html`;
  return withHash ? `${base}#${slugOf(to.file)}` : base;
};
const courseLabel = (c) => c.label;

function preview(card) {
  const html = readFileSync(join(root, card.course.slug, card.file), "utf8");
  const h1 = strip(html.match(/<h1[^>]*>([\s\S]*?)<\/h1>/)?.[1] || card.title);
  const sub = strip(html.match(/<p class="sub"[^>]*>([\s\S]*?)<\/p>/)?.[1] || "");
  const breath = strip(html.match(/<h2[^>]*>\s*In one breath\s*<\/h2>\s*<p[^>]*>([\s\S]*?)<\/p>/i)?.[1] || "");
  return { h1, sub: clip(sub, 200), breath: clip(breath, 330) };
}

function depsOf(card) {
  const list = card.course.track.series?.deps?.[`${card.course.slug}/${card.file}`] || [];
  return list.map((d) => {
    const m = d.match(/^\.\.\/([^/]+)\/(.+)$/);
    return m ? find(m[1], m[2]) : find(card.course.slug, d);
  }).filter(Boolean);
}

// ---------- card blocks ----------
const CARD_CSS = `/* series-nav */
.sn{max-width:820px;margin:0 auto;padding:16px 32px 0;font-family:-apple-system,"Inter",system-ui,sans-serif;font-size:13px;line-height:1.5;color:var(--muted)}
.sn a{color:inherit;text-decoration-color:var(--faint,var(--border));text-underline-offset:2px}
.sn .crumbs{display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 7px}
.sn .crumbs b{color:var(--text);font-weight:600}
.sn .sep{color:var(--faint,var(--border))}
.sn .step{margin-left:auto;white-space:nowrap}
.sn .dep{margin-top:5px;padding:6px 10px;border:1px solid var(--border);border-radius:8px;background:var(--surface)}
.sn .dep .ok{color:var(--accent);font-weight:600}
.snx{max-width:820px;margin:0 auto;padding:6px 32px 44px;font-family:-apple-system,"Inter",system-ui,sans-serif}
.snx .nx{display:grid;grid-template-columns:280px 1fr;text-decoration:none;color:var(--text);background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.snx .nx:hover,.snx .nx:focus-visible{border-color:var(--accent-border,var(--accent));outline:none}
.snx .mini{position:relative;height:210px;overflow:hidden;background:var(--bg);border-right:1px solid var(--border);padding:14px 16px}
.snx .mini span{display:block}
.snx .mini .m1{font-size:12.5px;font-weight:600;margin:0 0 4px;line-height:1.3}
.snx .mini .m2{font-size:9px;color:var(--muted);margin:0 0 8px;line-height:1.4}
.snx .mini .m3{font-size:8.5px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;margin:0 0 3px}
.snx .mini .m4{font-size:9.5px;margin:0;line-height:1.45}
.snx .tx{padding:16px 20px}
.snx .lab{font-size:12px;font-weight:600;color:var(--accent);letter-spacing:.04em;text-transform:uppercase}
.snx .tt{display:block;font-size:20px;font-weight:600;margin:5px 0 4px;line-height:1.25}
.snx .ds{display:block;font-size:14px;color:var(--muted)}
.snx .nd{display:block;font-size:13px;color:var(--muted);margin-top:10px}
.snx .go{display:block;margin-top:12px;font-size:14px;font-weight:600;color:var(--accent)}
.snx .aux{display:flex;justify-content:space-between;gap:12px;margin-top:10px;font-size:13px;color:var(--muted)}
.snx .aux a{color:inherit;text-decoration-color:var(--faint,var(--border));text-underline-offset:2px}
@media (max-width:560px){.sn,.snx{padding-left:16px;padding-right:16px}.snx .nx{grid-template-columns:1fr}.snx .mini{display:none}.sn .step{margin-left:0}}
/* /series-nav */`;

function topBlock(card, pos) {
  const c = card.course, t = c.track, flat = t.flat, px = t.series?.prefix;
  const deps = depsOf(card);
  let dep;
  if (!t.series) dep = "";
  else if (card.file === "_overview.html") dep = `<span class="ok">Overview.</span> Read it first as the map, or last as a recap.${c.needs.length ? ` This course needs ${c.needs.map((n) => `${px}-${n}`).join(" and ")}.` : ""}`;
  else if (!deps.length) dep = `<span class="ok">Stands alone.</span> You can jump in here without reading earlier cards.`;
  else dep = `<b>Before this:</b> ${deps.map((d) => `<a href="${href(card, d)}" target="_top">${esc(d.title)}</a>${d.course.slug !== c.slug ? ` (${px}-${d.course.n})` : ""}`).join(" · ")}`;
  const prev = pos > 0 ? flat[pos - 1] : null, next = flat[pos + 1];
  const step = [prev && `<a href="${href(card, prev)}" target="_top">← prev</a>`, next && `<a href="${href(card, next)}" target="_top">next →</a>`].filter(Boolean).join(" · ");
  return `<!-- series-nav:top -->
<nav class="sn" aria-label="Course navigation"><div class="crumbs"><a href="${t.home}" target="_top">${esc(t.title)}</a><span class="sep">›</span><a href="index.html" target="_top">${esc(courseLabel(c))}</a><span class="sep">›</span><b>${String(card.i + 1).padStart(2, "0")} / ${c.cards.length}</b><span class="step">${step}</span></div>
${dep ? `<div class="dep">${dep}</div>` : ""}</nav>
<!-- /series-nav:top -->`;
}

function bottomBlock(card, pos) {
  const t = card.course.track, flat = t.flat;
  const next = flat[pos + 1], prev = pos > 0 ? flat[pos - 1] : null;
  let box;
  if (next) {
    const p = preview(next);
    const crossing = next.course.slug !== card.course.slug;
    const lab = crossing ? `Next course · ${esc(courseLabel(next.course))}` : `Next up · ${String(next.i + 1).padStart(2, "0")} / ${next.course.cards.length}`;
    const nd = depsOf(next);
    const needs = !t.series ? "" : crossing ? next.course.note : nd.some((d) => d === card) ? "Builds directly on this card." : nd.length ? `Builds on: ${nd.map((d) => esc(d.title)).join(", ")}.` : "Stands alone.";
    box = `<a class="nx" href="${href(card, next)}" target="_top">
<span class="mini" aria-hidden="true"><span class="m1">${esc(p.h1)}</span><span class="m2">${esc(p.sub)}</span>${p.breath ? `<span class="m3">In one breath</span><span class="m4">${esc(p.breath)}</span>` : ""}</span>
<span class="tx"><span class="lab">${lab}</span><span class="tt">${esc(next.title)}</span><span class="ds">${esc(next.desc)}</span>${needs ? `<span class="nd">${needs}</span>` : ""}<span class="go">Continue →</span></span></a>`;
  } else if (!t.series) {
    box = `<a class="nx" href="../index.html" target="_top"><span class="mini" aria-hidden="true"><span class="m1">You've reached the end of ${esc(card.course.title)}</span></span>
<span class="tx"><span class="lab">End of the course</span><span class="tt">Back to all explainers</span><span class="ds">Every course and card on one page.</span><span class="go">Open all explainers →</span></span></a>`;
  } else {
    const soon = t.courses.find((c) => !c.built);
    box = `<a class="nx" href="${t.home}" target="_top"><span class="mini" aria-hidden="true"><span class="m1">You've reached the end of the published courses</span></span>
<span class="tx"><span class="lab">End of the series so far</span><span class="tt">Back to the map</span><span class="ds">${soon ? `Next to be built: ${esc(courseLabel(soon))}.` : ""}</span><span class="go">Open the map →</span></span></a>`;
  }
  const aux = `<div class="aux"><span>${prev ? `<a href="${href(card, prev)}" target="_top">← ${esc(prev.title)}</a>` : ""}</span><a href="index.html" target="_top">All cards in ${esc(courseLabel(card.course))}</a></div>`;
  return `<!-- series-nav:bottom -->
<div class="snx">${box}${aux}</div>
<!-- /series-nav:bottom -->`;
}

function patchCard(card) {
  const pos = card.course.track.flat.indexOf(card);
  const f = join(root, card.course.slug, card.file);
  let html = readFileSync(f, "utf8");
  html = html.replace(/\n?<!-- series-nav:top -->[\s\S]*?<!-- \/series-nav:top -->/, "").replace(/\n?<!-- series-nav:bottom -->[\s\S]*?<!-- \/series-nav:bottom -->/, "").replace(/\n?\/\* series-nav \*\/[\s\S]*?\/\* \/series-nav \*\//, "");
  html = html.replace(/<\/style>/, `${CARD_CSS}\n</style>`);
  html = html.replace(/<body([^>]*)>/, (m) => `${m}\n${topBlock(card, pos)}`);
  html = html.replace(/<\/body>/, `${bottomBlock(card, pos)}\n</body>`);
  writeFileSync(f, html);
}

// ---------- index blocks ----------
const INDEX_CSS = `/* series-nav */
.series{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 18px;font-size:13px}
.series a,.series span{padding:4px 10px;border:1px solid var(--border);border-radius:999px;color:var(--muted);text-decoration:none;background:var(--surface)}
.series a:hover{border-color:var(--accent-border,var(--accent))}
.series .cur{color:var(--accent);border-color:var(--accent-border,var(--accent));background:var(--accent-bg);font-weight:600}
.series span{opacity:.55}
.needs{font-size:13px;color:var(--muted);margin:4px 0 0}
.back{font-size:14px;color:var(--muted);text-decoration:none}
.rbody{flex:1;display:flex;min-height:0}
.rbody iframe{flex:1;min-width:0;width:auto}
.side{width:270px;flex:none;overflow:auto;border-right:1px solid var(--border);background:var(--surface);padding:10px 0 24px;font-size:13px}
.side[hidden]{display:none}
.side h3{font-size:11px;font-weight:600;color:var(--muted);letter-spacing:.04em;text-transform:uppercase;margin:14px 14px 4px}
.side a,.side button{display:block;width:100%;text-align:left;font:inherit;font-size:13px;color:var(--text);background:none;border:0;border-left:3px solid transparent;padding:4px 14px 4px 11px;cursor:pointer;text-decoration:none;line-height:1.35}
.side button:hover,.side a:hover{background:var(--surface2)}
.side button.on{border-left-color:var(--accent);background:var(--accent-bg);color:var(--accent);font-weight:600}
.side .n{color:var(--faint,var(--border));font-variant-numeric:tabular-nums;margin-right:6px}
.side .soon{color:var(--faint,var(--border));padding:4px 14px;display:block}
@media (max-width:900px){.side{position:absolute;top:0;bottom:0;left:0;z-index:2;box-shadow:none}.rbody{position:relative}}
/* /series-nav */`;

function patchIndex(c) {
  const t = c.track;
  const f = join(root, c.slug, "index.html");
  let html = readFileSync(f, "utf8");
  html = html.replace(/\n?\/\* series-nav \*\/[\s\S]*?\/\* \/series-nav \*\//, "").replace(/\n?<!-- series-nav -->[\s\S]*?<!-- \/series-nav -->/g, "").replace(/\n?\/\/ series-nav[\s\S]*?\/\/ \/series-nav/, "");
  html = html.replace(/<\/style>/, `${INDEX_CSS}\n</style>`);
  if (t.series) {
    const strip = t.courses.map((x) => x.built
      ? `<a href="../${x.slug}/index.html"${x.slug === c.slug ? ' class="cur" aria-current="page"' : ""}>${esc(courseLabel(x))}</a>`
      : `<span title="Coming later">${esc(courseLabel(x))}</span>`).join("");
    html = html.replace(/<main class="wrap">/, `<main class="wrap">\n<!-- series-nav -->\n<nav class="series" aria-label="Courses in the series">${strip}</nav>\n<!-- /series-nav -->`);
    html = html.replace(/(<p class="sub">[\s\S]*?<\/p>)/, `$1\n<!-- series-nav --><p class="needs"><b>Prerequisites:</b> ${esc(c.note)} Each card says what it builds on, so you can see where it's safe to jump around.</p><!-- /series-nav -->`);
  } else if (!html.includes('class="back"')) {
    html = html.replace(/<main class="wrap">/, `<main class="wrap">\n<!-- series-nav --><a class="back" href="../index.html">← All explainers</a><!-- /series-nav -->`);
  }
  if (!html.includes('class="rbody"')) {
    html = html.replace(/<iframe id="rframe" title="Card"><\/iframe>/, `<div class="rbody"><aside class="side" id="side" aria-label="Contents"></aside><iframe id="rframe" title="Card"></iframe></div>`);
    html = html.replace(/(<span class="t" id="rtitle">)/, `<button id="tog" aria-label="Toggle contents">☰</button>\n    $1`);
  }
  const others = t.series ? t.courses : [...allSeries.map((s) => ({ slug: s.map, label: s.title, built: true })), ...standalone];
  const data = JSON.stringify(others.map((x) => ({ slug: x.slug, label: courseLabel(x), built: x.built }))).replace(/</g, "\\u003c");
  const js = `// series-nav
const SERIES = ${data}, HERE = ${JSON.stringify(c.slug)};
const side = document.getElementById("side");
{
  let h = "", n = 0;
  for (const [name, , cards] of GROUPS) {
    h += "<h3>" + name + "</h3>";
    for (const [file, title] of cards) h += '<button data-go="' + (n++) + '"><span class="n">' + String(n).padStart(2, "0") + "</span>" + title + "</button>";
  }
  h += "<h3>Other courses</h3>";
  for (const s of SERIES) if (s.slug !== HERE) h += s.built ? "<a href='../" + s.slug + "/index.html'>" + s.label + "</a>" : '<span class="soon">' + s.label + " (coming)</span>";
  h += "<a href='../index.html'>All explainers →</a>";
  side.innerHTML = h;
}
if (matchMedia("(max-width:900px)").matches) side.hidden = true;
document.getElementById("tog").onclick = () => { side.hidden = !side.hidden; };
side.addEventListener("click", (e) => { if (e.target.closest("[data-go]") && matchMedia("(max-width:900px)").matches) side.hidden = true; });
function mark() { side.querySelectorAll("[data-go]").forEach((b) => b.classList.toggle("on", +b.dataset.go === cur)); side.querySelector(".on")?.scrollIntoView({ block: "nearest" }); }
new MutationObserver(mark).observe(document.getElementById("rpos"), { childList: true });
mark();
addEventListener("hashchange", () => { const i = CARDS.findIndex((x) => "#" + x[0].replace(".html", "") === location.hash); if (i >= 0 && i !== cur) show(i); });
// /series-nav`;
  html = html.replace(/<\/script>\s*<\/body>/, `${js}\n</script>\n</body>`);
  writeFileSync(f, html);
}

flat.forEach(patchCard);
built.forEach(patchIndex);
console.log(`course-nav: ${flat.length} cards in ${built.length} courses (${built.map((c) => c.slug).join(", ")})`);
