import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inject, strip, validate, refHref } from "../study-lib.mjs";

const card = `<!doctype html><html><head><title>Regex FSM</title><style>
:root{--bg:#fff}
body{margin:0}
</style></head><body>
<!-- series-nav:top -->nav<!-- /series-nav:top -->
<h1>Regex and finite state machines</h1>
<p class="sub">Also called <code>FSM</code>.</p>

<p><b>In one breath.</b> body</p>
<h2>Worked example</h2>
<div class="foot">Watch · Code</div>
<!-- series-nav:bottom -->x<!-- /series-nav:bottom -->
</body></html>`;

const study = {
  prime: {
    problem: "Constrained decoding must know which tokens are legal at every step.",
    fits: "After token masking; before CFG/PDA engines.",
    expect: ["FSM state", "token index", "mask"],
    goal: "Trace a regex through states to a logits mask.",
  },
  attend: [
    "What must the FSM state encode about the text so far?",
    "How does one state turn into a set of allowed vocabulary tokens?",
    "Why does unbounded nesting break a finite set of states?",
  ],
  core: {
    must: [
      "Constraint → FSM state → allowed transitions → allowed tokens → logits mask → sampling",
      "Finite states → no stack → can't count unbounded nesting → JSON with arbitrary depth needs a PDA",
    ],
    reference: ["The internal index layout"],
    elevated: ["Precomputing state→token sets → per-step masking is a lookup, not a scan"],
  },
  compress: { prompt: "Explain FSM-constrained decoding in four sentences.", checklist: ["state", "mask", "limit"] },
  retrieve: [
    { q: "Walk from logits to the sampled token under an FSM constraint.", kind: "mechanism", answer: ["mask illegal logits to −∞"] },
    { q: "Why can a regex engine stay cheap per token?", kind: "concept", answer: ["precomputed index"] },
    { q: "When would you pay for a PDA instead?", kind: "tradeoff", answer: ["recursive schemas"] },
  ],
  compare: [{ with: "structured-outputs/cfg-pda", q: "FSM vs PDA: what changes?", answer: ["stack"] }],
  predict: { q: "Which JSON shape first exposes the FSM limit?", answer: "Arbitrarily nested arrays." },
  apply: { task: "Hand-trace the regex over 4 tokens and list masked tokens.", done: "Your masks match the card.", hint: "Start in state 0." },
  reflect: ["Which prediction did you get wrong?", "Could you explain the nesting limit architecturally?", "What will you revisit?"],
  revisit: ["structured-outputs/token-masking"],
  prereq: [],
};

const ctx = { course: "structured-outputs", slug: "regex-fsm", titleOf: (ref) => ref.split("/")[1] };

test("inject puts prime after the subtitle and the panel before the footer", () => {
  const out = inject(card, study, ctx);
  const sub = out.indexOf('<p class="sub">'), prime = out.indexOf("<!-- study:prime -->");
  const body = out.indexOf("In one breath"), panel = out.indexOf("<!-- study:panel -->"), foot = out.indexOf('<div class="foot">');
  assert.ok(sub < prime && prime < body, "prime between subtitle and body");
  assert.ok(body < panel && panel < foot, "panel before footer");
  assert.match(out, /\/\* study:css \*\/[\s\S]*\/\* \/study:css \*\/\n<\/style>/);
});

test("inject is idempotent", () => {
  const once = inject(card, study, ctx);
  assert.equal(inject(once, study, ctx), once);
});

test("strip removes every study block, even duplicates", () => {
  const twice = inject(card, study, ctx).replace("<!-- study:panel -->", "<!-- study:panel -->dup<!-- /study:panel --><!-- study:panel -->");
  assert.equal(strip(twice), card);
});

test("panel falls back to series-nav:bottom, then </body>", () => {
  const noFoot = card.replace('<div class="foot">Watch · Code</div>\n', "");
  const out = inject(noFoot, study, ctx);
  assert.ok(out.indexOf("<!-- study:panel -->") < out.indexOf("<!-- series-nav:bottom -->"));
  const bare = noFoot.replace(/<!-- series-nav:bottom -->x<!-- \/series-nav:bottom -->\n/, "");
  const out2 = inject(bare, study, ctx);
  assert.ok(out2.indexOf("<!-- /study:panel -->") < out2.indexOf("</body>"));
});

test("stepper is gated in order and has a show-all switch", () => {
  const out = inject(card, study, ctx);
  const stages = [...out.matchAll(/<details class="st"/g)].length;
  assert.equal(stages, 7);
  assert.ok(out.indexOf('class="study-all"') < out.indexOf('<details class="st"'));
  assert.doesNotMatch(out, /<script/);
});

test("refHref is relative to the card's course", () => {
  assert.equal(refHref("structured-outputs/cfg-pda", "structured-outputs"), "cfg-pda.html");
  assert.equal(refHref("z2h-2-micrograd/value-object", "z2h-8-gpt"), "../z2h-2-micrograd/value-object.html");
});

test("validate accepts a good study and flags the classic failures", () => {
  const root = mkdtempSync(join(tmpdir(), "study-"));
  mkdirSync(join(root, "structured-outputs"));
  for (const s of ["cfg-pda", "token-masking", "regex-fsm"]) writeFileSync(join(root, "structured-outputs", s + ".html"), "x");
  assert.deepEqual(validate(study, { ...ctx, root }), []);

  const bad = structuredClone(study);
  bad.attend = ["What is it?", "Why does this exist?"];
  bad.core.must = ["FSMs are used"];
  bad.retrieve = bad.retrieve.filter((r) => r.kind !== "tradeoff");
  bad.compare[0].with = "structured-outputs/nope";
  bad.panel = "<script>x</script>";
  const p = validate(bad, { ...ctx, root }).join("\n");
  assert.match(p, /attend: 3–6/);
  assert.match(p, /generic question/);
  assert.match(p, /must.*causal chain/);
  assert.match(p, /retrieve: needs a tradeoff/);
  assert.match(p, /unknown card structured-outputs\/nope/);
  assert.match(p, /unknown key: panel/);
});

test("validate rejects script and hard-coded colours inside fields", () => {
  const root = mkdtempSync(join(tmpdir(), "study-"));
  mkdirSync(join(root, "structured-outputs"));
  for (const s of ["cfg-pda", "token-masking", "regex-fsm"]) writeFileSync(join(root, "structured-outputs", s + ".html"), "x");
  const bad = structuredClone(study);
  bad.prime.goal = '<span style="color:#f00">x</span><script>1</script>';
  const p = validate(bad, { ...ctx, root }).join("\n");
  assert.match(p, /prime.goal: no <script>/);
  assert.match(p, /prime.goal: no inline style/);
});

test("each stage links to its principle, and field-guide backlinks render in Prime", () => {
  const out = inject(card, study, { ...ctx, usedIn: [{ guide: "Agents", title: "p2 coding tools", url: "https://example.com/#p2" }] });
  for (const id of ["layers", "compress", "retrieve", "interleave", "predict", "apply", "reflect", "prime"])
    assert.ok(out.includes(`learning-principles.html#${id}"`), id);
  assert.match(out, /Used in the field guides:<\/b> <a href="https:\/\/example.com\/#p2">p2 coding tools<\/a> \(Agents\)/);
  assert.doesNotMatch(inject(card, study, ctx), /Used in the field guides/);
});

test("inject keeps its CSS where it was, so another script's CSS appended after it doesn't make them swap", () => {
  const once = inject(card, study, ctx);
  const withNav = once.replace("</style>", "/* series-nav */\n.sn{}\n/* /series-nav */\n</style>");
  assert.equal(inject(withNav, study, ctx), withNav);
});

test("inject is idempotent when the footer is indented rather than on a fresh line", () => {
  const indented = card.replace('<div class="foot">', '  <div class="foot">').replace("\n  <div", "\n\n  <div");
  const once = inject(indented, study, ctx);
  assert.equal(inject(once, study, ctx), once);
  assert.equal(strip(once), indented);
});

test("a card without a footer anchors the panel inside </article>, clear of course-nav's bottom block", () => {
  const paper = card.replace('<div class="foot">Watch · Code</div>\n', "").replace("<h1>", "<article><h1>").replace("<!-- series-nav:bottom -->", "</article>\n<!-- series-nav:bottom -->");
  const once = inject(paper, study, ctx);
  assert.ok(once.indexOf("<!-- /study:panel -->") < once.indexOf("</article>"));
  const renav = once.replace(/\n?<!-- series-nav:bottom -->[\s\S]*?<!-- \/series-nav:bottom -->/, "").replace("</body>", "<!-- series-nav:bottom -->y<!-- /series-nav:bottom -->\n</body>");
  assert.equal(inject(renav, study, ctx), renav);
});
