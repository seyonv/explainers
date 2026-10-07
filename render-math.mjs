// Typeset every \( … \) (inline) and \[ … \] (display) with KaTeX at build time, for any course that
// ships a katex/ folder (katex.min.js, katex.min.css, fonts/). Pages then load only the CSS and fonts.
// study-build.mjs calls this after injecting Study Mode; it can also be run on files directly:
//   node render-math.mjs <course>/<slug>.html [...]
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
const req = createRequire(import.meta.url);
const engines = {};
const katexFor = (course) => (engines[course] ??= req(resolve(course, "katex/katex.min.js")));

// One notation for the whole course (see llm-math/_facts.md, "Notation").
export const macros = {
  "\\Wq": "W_{q}", "\\Wk": "W_{k}", "\\Wv": "W_{v}", "\\Wo": "W_{o}",
  "\\Wout": "W_{\\text{out}}", "\\Wup": "W_{1}", "\\Wdown": "W_{2}",
  "\\softmax": "\\operatorname{softmax}", "\\LN": "\\operatorname{LN}", "\\GELU": "\\operatorname{GELU}",
  "\\ReLU": "\\operatorname{ReLU}", "\\RMS": "\\operatorname{RMS}", "\\mean": "\\operatorname{mean}",
  "\\ids": "\\text{ids}", "\\tgt": "\\text{target}",
  "\\tok": "\\text{#1}",
  // highlight one entry of a matrix: activation (blue), weight (violet), gradient (green), failure (red)
  "\\hl": "\\htmlClass{k-act}{#1}", "\\hw": "\\htmlClass{k-w}{#1}",
  "\\hg": "\\htmlClass{k-grad}{#1}", "\\hb": "\\htmlClass{k-bad}{#1}",
};

const opts = (displayMode) => ({ displayMode, macros: { ...macros }, trust: (c) => c.command === "\\htmlClass",
  strict: "ignore", throwOnError: true, output: "htmlAndMathml" });

export function render(html, course, where = "") {
  const katex = katexFor(course);
  const out = [];
  // leave <script>, <style>, <code> and <pre> alone
  const parts = html.split(/(<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>|<pre[\s\S]*?<\/pre>|<code[\s\S]*?<\/code>)/);
  let n = 0;
  for (const [i, part] of parts.entries()) {
    if (i % 2) { out.push(part); continue; }
    out.push(part
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, tex) => { n++; return wrap(katex, tex, true, where); })
      .replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => { n++; return wrap(katex, tex, false, where); }));
  }
  return [out.join(""), n];
}

function wrap(katex, tex, display, where) {
  const clean = tex.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">");
  try {
    return katex.renderToString(clean, opts(display));
  } catch (e) {
    throw new Error(`${where}: KaTeX failed on ${display ? "\\[" : "\\("}${tex}: ${e.message}`);
  }
}

export function renderPage(html, course, where) {
  const [out, n] = render(html, course, where);
  return [out.includes("katex/katex.min.css") ? out : out.replace("</head>", '<link rel="stylesheet" href="katex/katex.min.css">\n</head>'), n];
}

if (process.argv[1].endsWith("render-math.mjs")) {
  for (const f of process.argv.slice(2)) {
    let [html, n] = renderPage(readFileSync(f, "utf8"), dirname(f), f);
    writeFileSync(f, html);
    console.log(`${f}: ${n} formulas`);
  }
}
